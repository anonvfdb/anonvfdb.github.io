#!/usr/bin/env python3
"""Generate response/index.html — the reviewer-response page.

Emits two pages:
  index.html   — A. informed consent (header + links out), B. qualitative samples
  consent.html — the two consent PDFs from ../consent-forms/, embedded in full

Section B renders the samples listed in PICKS, one row each: the full audio-visual
run on the left and the same model's audio-only run on the same source clip on the
right, so the pair isolates what the visual channel changes. Generation items get a
single centred column — the agent renders its own video, so there is no audio-only
counterpart. Composites come from vfdb-study/data/composites.

    python3 build_response.py            # clips served via the symlink
    python3 build_response.py --final    # copy just the referenced clips in
"""
import argparse
import base64
import csv
import html
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.abspath(os.path.join(HERE, "..", "..", "vfdb-study"))
sys.path.insert(0, STUDY)

from rubrics import AXIS_LABELS, AXIS_HINTS, axes_for  # noqa: E402

ITEM_KEY = os.path.join(STUDY, "data", "item_key.csv")
MANIFEST = os.path.join(STUDY, "data", "manifest.jsonl")
RATINGS_DIR = os.path.join(STUDY, "ratings")
COMPOSITES = os.path.join(STUDY, "data", "composites")
CONSENT_DIR = os.path.join(HERE, "..", "consent-forms")
SITE_CSS = os.path.join(HERE, "..", "style.css")
PAGE_CSS = os.path.join(HERE, "response.css")
ENCODED = os.path.join(HERE, ".encoded")      # display-resolution clips, gitignored
ENCODE_SCALE = "854:-2"
ENCODE_CRF = "30"

OUT_HTML = os.path.join(HERE, "index.html")
OUT_CONSENT = os.path.join(HERE, "consent.html")
# --inline writes these instead; staticrypt turns them into the shipped .html files.
SRC_HTML = os.path.join(HERE, "index.src.html")
SRC_CONSENT = os.path.join(HERE, "consent.src.html")
# Whisper transcripts of the agent channel for composites with no manifest entry,
# produced by transcribe_counterparts.py. Absent file = feature simply off.
TRANSCRIPTS = os.path.join(HERE, "transcripts.json")
ASR = json.load(open(TRANSCRIPTS)) if os.path.exists(TRANSCRIPTS) else {}

# The samples shown on the page, in display order. Each renders one row: AV in the
# left column, the matching audio-only run in the right.
PICKS = ["g0005", "g0006", "g0009", "g0010",
         "p0001", "p0002", "p0005", "p0017", "p0028", "p0032",
         "p0036", "p0037", "p0046", "p0047", "p0048"]

# Raters are pseudonymous in the study logs; the page shows them as R1/R2 in a
# stable order so a reviewer can tell "same rater" across items.
RATER_DISPLAY = {"user_2": "Rater A", "user-3": "Rater B", "user_3": "Rater B"}

# item_key.csv stores internal run slugs for the cascaded generation pipelines.
# Map them onto the names the paper uses, keeping the run slug visible underneath
# so a reviewer can line the sample up with the released model outputs.
MODEL_DISPLAY = {
    "MiniOmni2": ("Mini-Omni2", None),
    "gemini-livekit-avatar-anam-gen-full":
        ("Gemini 2.5 Flash Native + Anam", "gemini-livekit-avatar-anam-gen-full"),
    "gemini-livekit-avatar-anam-gen-pilot":
        ("Gemini 2.5 Flash Native + Anam", "gemini-livekit-avatar-anam-gen-pilot"),
    "gemini-livekit-avatar-keyframe-gen-full":
        ("Gemini 2.5 Flash Native + Keyframe", "gemini-livekit-avatar-keyframe-gen-full"),
    "gemini-livekit-avatar-keyframe-gen-pilot":
        ("Gemini 2.5 Flash Native + Keyframe", "gemini-livekit-avatar-keyframe-gen-pilot"),
}

MODE_LABEL = {
    "av": ("AV", "full audio-visual input"),
    "ao": ("Audio-only", "audio-only input, video withheld"),
    "gen": ("Generation", "agent produces speech + avatar video"),
}

CONSENT_FORMS = [
    {
        "file": "Project Form (Audio and Video Projects).pdf",
        "anchor": "project-form",
        "title": "Project Form — Audio and Video Projects",
        "blurb": (
            "The per-project disclosure shown to every contributor before recording: what the "
            "session captures, what the recordings are used for, how long they are retained, and "
            "the contributor's withdrawal rights."
        ),
    },
    {
        "file": "Content Contribution Agreement.pdf",
        "anchor": "contribution-agreement",
        "title": "Content Contribution Agreement",
        "blurb": (
            "The signed agreement under which contributors grant rights in the recorded audio and "
            "video. Executed by every participant appearing in the dataset prior to their session."
        ),
    },
]


def load_items():
    key = {r["item_id"]: r for r in csv.DictReader(open(ITEM_KEY))}
    man = {}
    with open(MANIFEST) as fh:
        for line in fh:
            d = json.loads(line)
            man[d["item_id"]] = d

    ratings = defaultdict(list)
    for fn in sorted(os.listdir(RATINGS_DIR)):
        if not fn.endswith(".jsonl"):
            continue
        for line in open(os.path.join(RATINGS_DIR, fn)):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ratings[d["item_id"]].append(d)

    items = []
    for iid in sorted(key, key=lambda s: (s[0], s)):
        k, m = key[iid], man.get(iid, {})
        rs = sorted(ratings.get(iid, []), key=lambda d: RATER_DISPLAY.get(d["rater_id"], d["rater_id"]))
        items.append({"id": iid, "key": k, "man": m, "ratings": rs})
    return items


def esc(s):
    return html.escape(str(s if s is not None else ""))


def fmt_score(v):
    """Judge/human scores are on an integer 0-5 scale; the CSV stores them as floats."""
    if v in (None, "", "nan"):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return str(int(f)) if f == int(f) else f"{f:.1f}"


def score_class(v):
    if v is None:
        return "na"
    f = float(v)
    if f <= 1:
        return "s-low"
    if f <= 3:
        return "s-mid"
    return "s-high"


def model_label(model):
    """Paper-facing model name. Cascaded generation runs also carry pilot/full, which
    is the only thing distinguishing two runs of the same pipeline."""
    name, run = MODEL_DISPLAY.get(model, (model, None))
    if run and run.rsplit("-", 1)[-1] in ("pilot", "full"):
        name += f" ({run.rsplit('-', 1)[-1]})"
    return name


def score_table(item):
    """One row per axis that applies to this (bucket, dynamic): judge vs each human."""
    k, man = item["key"], item["man"]
    axes = man.get("axes") or axes_for(k["bucket"], k["dynamic_type"])
    if not axes:
        return '<p class="muted small">No scoring axes defined for this dynamic.</p>'

    raters = [(RATER_DISPLAY.get(r["rater_id"], r["rater_id"]), r) for r in item["ratings"]]

    head = "".join(f"<th>{esc(n)}</th>" for n, _ in raters)
    rows = []
    for ax in axes:
        jv = fmt_score(k.get(f"judge_{ax}"))
        cells = [
            f'<td class="ax"><span class="ax-name">{esc(AXIS_LABELS.get(ax, ax))}</span>'
            f'<span class="ax-hint">{esc(AXIS_HINTS.get(ax, ""))}</span></td>',
            f'<td class="sc {score_class(jv)}">{jv if jv is not None else "&mdash;"}</td>',
        ]
        for _, r in raters:
            hv = fmt_score(r.get("scores", {}).get(ax))
            cells.append(f'<td class="sc {score_class(hv)}">{hv if hv is not None else "&mdash;"}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")

    note = ""
    notes = [(n, r.get("note", "").strip()) for n, r in raters if r.get("note", "").strip()]
    if notes:
        note = '<div class="rater-notes">' + "".join(
            f'<p><span class="who">{esc(n)}</span>{esc(t)}</p>' for n, t in notes
        ) + "</div>"

    empty = "" if raters else (
        '<p class="muted small nohuman">Not covered in the human-validation subset '
        "(judge score only).</p>"
    )

    return (
        '<table class="scores"><thead><tr><th>Axis</th><th>LM judge</th>'
        + head
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + empty
        + note
    )


def transcript_block(item):
    ag = item["man"].get("agent") or {}
    t = (ag.get("transcript") or "").strip()
    if not t:
        if ag.get("missing"):
            return '<p class="muted small">Agent produced no response for this item.</p>'
        return (
            '<p class="muted small">No text transcript &mdash; this run emits native audio; '
            "the response is in the clip above.</p>"
        )
    long = " long" if len(t) > 900 else ""
    return f'<blockquote class="transcript{long}">{esc(t)}</blockquote>'


def _base_clip(c):
    """av and ao runs on the same source clip differ only by an export suffix."""
    return c.replace("__perception", "").replace("__generation", "")


def rendered_counterpart(item_id, mode):
    """Filename of a post-hoc counterpart composite, if one has been rendered."""
    fn = f"{item_id}-{mode}.mp4"
    return fn if os.path.exists(os.path.join(COMPOSITES, fn)) else None


def find_partner(item, items_by_id):
    """The same model's run on the same source clip in the opposite input mode.

    The study sampled items stratified over the judge's score range, not as matched
    av/ao pairs, so most picks have no partner rendered. Returns None in that case
    rather than substituting a different model's run, which would not be an AV-delta
    comparison at all.
    """
    k = item["key"]
    want = "ao" if k["mode"] == "av" else "av"
    for other in items_by_id.values():
        ok = other["key"]
        if (ok["item_id"] != k["item_id"]
                and ok["model"] == k["model"]
                and ok["mode"] == want
                and _base_clip(ok["clip_id"]) == _base_clip(k["clip_id"])):
            return other
    return None


def pair_cell(item, mode, rendered=None, row_key=None):
    """One column of a pair row.

    Three cases: a study item (has metadata + transcript), a `rendered` counterpart
    composite produced after the fact (video only — these were never sampled into the
    study, so there is no manifest entry to draw a transcript from), or nothing.
    """
    label, tip = MODE_LABEL.get(mode, (mode, ""))

    if item is None and rendered:
        asr = ASR.get(rendered[:-4], {})
        # A silent agent channel is a real result, not a missing transcript — leave it blank.
        block = ""
        if asr.get("text"):
            long = " long" if len(asr["text"]) > 900 else ""
            block = (f'<h4>Agent transcript</h4>'
                     f'<blockquote class="transcript{long}">{esc(asr["text"])}</blockquote>')
        return f"""
      <div class="pair-col">
        <div class="col-head">
          <span class="pill pill-mode" title="{esc(tip)}">{esc(label)}</span>
        </div>
        <video controls preload="metadata" playsinline src="clips/{esc(rendered)}"></video>
        {block}
      </div>"""

    if item is None:
        # A human reference has no opposite-mode run by construction, not by omission:
        # the human interlocutor was on a video call, so "video withheld" is not a
        # condition that exists for them. Say that rather than implying a gap.
        if row_key is not None and row_key.get("source") == "human":
            title = f"Human reference &mdash; no {label.lower()} run available"
            body = ("The human interlocutor was on a video call, so withholding video is not "
                    "a condition that exists for them.")
        else:
            title = f"No matched {esc(label)} run rendered"
            body = (f"This model was not sampled in {label} mode on this source clip, "
                    "so no composite exists for it.")
        return f"""
      <div class="pair-col is-missing">
        <div class="col-head"><span class="pill pill-mode">{esc(label)}</span></div>
        <div class="missing-box">
          <p class="missing-title">{title}</p>
          <p class="muted small">{body}</p>
        </div>
      </div>"""

    k = item["key"]
    iid = k["item_id"]
    return f"""
      <div class="pair-col">
        <div class="col-head">
          <span class="pill pill-mode" title="{esc(tip)}">{esc(label)}</span>
        </div>
        <video controls preload="metadata" playsinline src="clips/{esc(iid)}.mp4"></video>
        <h4>Agent transcript</h4>
        {transcript_block(item)}
      </div>"""


def pair_row(item, partner):
    """One pick = one row. Left column av, right column ao, whichever the pick is.

    Generation items have no av/ao split — the agent renders its own video, so there
    is no 'video withheld' counterpart — and get a single full-width cell instead.
    """
    k = item["key"]
    win = item["man"].get("event_window") or [None, None]
    win_s = f"{win[0]:g}&ndash;{win[1]:g}s" if win[0] is not None else "&mdash;"
    is_gen = k["bucket"] == "generation"

    if is_gen:
        cols, status = pair_cell(item, "gen"), "solo"
    else:
        av = item if k["mode"] == "av" else partner
        ao = item if k["mode"] == "ao" else partner
        # counterparts rendered after the study land as <pick_id>-<mode>.mp4
        av_r = None if av else rendered_counterpart(k["item_id"], "av")
        ao_r = None if ao else rendered_counterpart(k["item_id"], "ao")
        cols = pair_cell(av, "av", av_r, k) + pair_cell(ao, "ao", ao_r, k)
        status = "paired" if (partner or av_r or ao_r) else "unpaired"

    return f"""
  <article class="pair {status}" id="pair-{esc(k['item_id'])}">
    <div class="pair-head">
      <div class="pills">
        <span class="pill pill-id">{esc(k['item_id'])}</span>
        <span class="pill pill-model">{esc(model_label(k['model']))}</span>
        <span class="pill pill-bucket">{esc(k['dynamic_type'])}</span>
        <span class="pill">{win_s}</span>
        {'<span class="pill pill-human">human reference</span>' if k['source'] == 'human' else ''}
        {'<span class="pill pill-paired">AV/AO pair</span>' if partner else ''}
      </div>
    </div>
    <div class="pair-cols{' one-col' if is_gen else ''}">{cols}</div>
  </article>"""


def b64_file(path):
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def encoded_clip(name):
    """Path to the display-resolution copy of a composite, encoding it on first use.

    The gated pages carry their media inline, so every byte is paid for twice (base64
    inflates ~33%, staticrypt then roughly doubles). 854x480 / CRF 30 keeps gaze,
    gesture and the event box legible at a quarter of the source size. Cached by
    source mtime so repeat builds are free.
    """
    os.makedirs(ENCODED, exist_ok=True)
    src = os.path.join(COMPOSITES, name)
    dst = os.path.join(ENCODED, name)
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return dst
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
         "-vf", f"scale={ENCODE_SCALE}", "-c:v", "libx264", "-crf", ENCODE_CRF,
         "-preset", "veryfast", "-movflags", "+faststart",
         "-c:a", "aac", "-b:a", "48k", "-ac", "2", dst],
        check=True)
    return dst


def inline_assets(html, clips):
    """Fold every external reference into the document so nothing leaks past the gate."""
    # Stylesheets: the site's shared sheet plus this page's.
    css = "\n".join(open(p).read() for p in (SITE_CSS, PAGE_CSS))
    html = re.sub(r'\s*<link rel="stylesheet" href="[^"]+">', "", html, count=2)
    html = html.replace("</head>", f"<style>\n{css}\n</style>\n</head>", 1)

    # Clips: swap each clips/<name> reference for the encoded bytes.
    for name in clips:
        uri = "data:video/mp4;base64," + b64_file(encoded_clip(name))
        html = html.replace(f'src="clips/{name}"', f'src="{uri}"')
    return html


def sample_rows(items_by_id):
    """The qualitative-samples section body: one row per pick, in PICKS order."""
    rows, paired, pairable, gen = [], 0, 0, 0
    for iid in PICKS:
        item = items_by_id.get(iid)
        if item is None:
            print(f"  warning: {iid} not in the study set — skipped")
            continue
        if item["key"]["bucket"] == "generation":
            gen += 1
            rows.append(pair_row(item, None))
            continue
        pairable += 1
        partner = find_partner(item, items_by_id)
        other = "ao" if item["key"]["mode"] == "av" else "av"
        if partner or rendered_counterpart(iid, other):
            paired += 1
        rows.append(pair_row(item, partner))

    # Reported to the terminal, not the page.
    print(f"  samples: {len(rows)} rows — {paired}/{pairable} perception rows paired, "
          f"{gen} generation single-column")
    return "".join(rows)


def referenced_clips(items_by_id):
    """Every composite the page actually loads — what --final needs to copy in."""
    want = set()
    for iid in PICKS:
        item = items_by_id.get(iid)
        if item is None:
            continue
        want.add(iid + ".mp4")
        if item["key"]["bucket"] == "generation":
            continue
        partner = find_partner(item, items_by_id)
        if partner:
            want.add(partner["id"] + ".mp4")
        for m in ("av", "ao"):
            r = rendered_counterpart(iid, m)
            if r:
                want.add(r)
    return sorted(want)


def consent_section():
    """Section A on the response page: header + links out to consent.html."""
    links = "".join(
        f'<li><a href="consent.html#{c["anchor"]}">{esc(c["title"])}</a></li>'
        for c in CONSENT_FORMS
    )
    return f"""
    <p>Every person appearing in VideoFDB recorded their session through data capture co-authors
    under a two-part consent instrument. Contributors were compensated, were told before recording that the
    sessions would be released as a research evaluation dataset, and retain the right to withdraw.
    No session was recorded before both documents were completed.</p>
    <p>Both instruments are reproduced in full on a separate page:</p>
    <ul class="consent-index">{links}</ul>
    <p class="consent-cta"><a class="btn btn-sm" href="consent.html">View consent forms &rarr;</a></p>"""


def consent_page(inline=False):
    """Standalone page carrying both consent PDFs.

    With inline=True the PDF bytes are embedded and rebuilt client-side as blob
    URLs. Data: URIs are unreliable for the PDF viewer across browsers; blobs are
    same-origin and render consistently — and, unlike a linked file, they cannot be
    fetched around the password gate.
    """
    cards, blobs = [], []
    for c in CONSENT_FORMS:
        anchor = esc(c["anchor"])
        if inline:
            b64 = b64_file(os.path.join(CONSENT_DIR, c["file"]))
            blobs.append(f'<script type="text/plain" data-pdf="{anchor}">{b64}</script>')
            viewer = f'<div class="pdf" id="pdf-{anchor}"></div>'
            link = (f'<a class="btn btn-sm" id="dl-{anchor}" '
                    f'download="{esc(c["file"])}">Open PDF &rarr;</a>')
        else:
            href = "../consent-forms/" + c["file"].replace(" ", "%20")
            viewer = (f'<object class="pdf" data="{href}#view=FitH" type="application/pdf">'
                      '<p class="muted small">Inline PDF preview is unavailable in this browser '
                      '&mdash; use the link below.</p></object>')
            link = f'<a class="btn btn-sm" href="{href}">Open PDF &rarr;</a>'

        cards.append(f"""
    <div class="consent-card" id="{anchor}">
      <h2 class="consent-title">{esc(c['title'])}</h2>
      <p>{esc(c['blurb'])}</p>
      {viewer}
      <p class="consent-links">
        {link}
        <span class="muted small filename">{esc(c['file'])}</span>
      </p>
    </div>""")

    pdf_script = ""
    if inline:
        pdf_script = "".join(blobs) + """
<script>
document.querySelectorAll('script[data-pdf]').forEach(function (s) {
  var bin = atob(s.textContent.trim());
  var buf = new Uint8Array(bin.length);
  for (var i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  var url = URL.createObjectURL(new Blob([buf], {type: 'application/pdf'}));
  var id = s.getAttribute('data-pdf');
  document.getElementById('pdf-' + id).innerHTML =
    '<object class="pdf" type="application/pdf" data="' + url + '#view=FitH">' +
    '<p class="muted small">Inline PDF preview is unavailable in this browser &mdash; ' +
    'use the link below.</p></object>';
  document.getElementById('dl-' + id).href = url;
});
</script>"""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>VideoFDB &mdash; Informed Consent</title>
<link rel="stylesheet" href="../style.css">
<link rel="stylesheet" href="response.css">
</head>
<body>
<main class="wide">
  <h1>VideoFDB &mdash; Informed Consent</h1>
  <p class="subtitle">Anonymous submission, NeurIPS 2026 Datasets &amp; Benchmarks track.
  The two-part consent instrument completed by every contributor appearing in the dataset,
  reproduced in full.</p>

  <nav class="pagenav">
    <a href="index.html">&larr; Response to reviewers</a>
    <a href="#{esc(CONSENT_FORMS[0]['anchor'])}">{esc(CONSENT_FORMS[0]['title'])}</a>
    <a href="#{esc(CONSENT_FORMS[1]['anchor'])}">{esc(CONSENT_FORMS[1]['title'])}</a>
  </nav>

  <p>No session was recorded before both documents below were completed. Contributors were
  compensated, were told before recording that the sessions would be released as a research
  evaluation dataset, and retain the right to withdraw.</p>
  <p>Downstream use of the released clips is separately bound by the
  <a href="../LICENSE">NVIDIA Evaluation License</a>, which prohibits re-identification, likeness
  generation, biometric processing, and any commercial use &mdash; see the
  <a href="../index.html">terms-of-use page</a>.</p>
{"".join(cards)}
</main>
{pdf_script}
</body>
</html>
"""


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>VideoFDB &mdash; Response to Reviewers</title>
<link rel="stylesheet" href="../style.css">
<link rel="stylesheet" href="response.css">
</head>
<body>
<main class="wide">
  <h1>VideoFDB &mdash; Response to Reviewers</h1>
  <p class="subtitle">Anonymous submission, NeurIPS 2026 Datasets &amp; Benchmarks track.
  This page hosts supplementary materials requested during review: the informed consent forms with
  contributor terms, and a set of qualitative samples with their model attribution and
  transcripts.</p>

  <nav class="pagenav">
    <a href="#consent">A. Informed consent</a>
    <a href="#samples">B. Qualitative samples</a>
    <a href="consent.html">Consent forms &rarr;</a>
    <a href="../index.html">&larr; Terms of use</a>
  </nav>

  <section id="consent">
    <h2>A. Informed consent</h2>
    __CONSENT__
  </section>

  <section id="samples">
    <h2>B. Qualitative samples</h2>
    <p>Each sample is a side-by-side composite: the <b>user</b> on the left, the <b>agent</b> on the
    right &mdash; as a waveform for perception items, as the generated avatar for generation items.
    Where a sample has two columns, both are the <em>same model on the same source clip</em>, run
    once with full audio-visual input and once with the video withheld; the pair isolates what the
    visual channel changes. Items marked <em>human reference</em> are the original human
    interlocutor's response, run through the identical pipeline.</p>

    __CARDS__
  </section>
</main>

<script>
// Only one clip audible at a time — these are all conversational audio.
document.addEventListener('play', function (e) {
  if (e.target.tagName !== 'VIDEO') return;
  document.querySelectorAll('video').forEach(function (v) {
    if (v !== e.target) v.pause();
  });
}, true);
</script>
</body>
</html>
"""


def looks_encrypted(path):
    """True if `path` is a staticrypt output — never silently overwrite one."""
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8", errors="replace") as fh:
        return "staticryptEncryptedMsgUniqueVariableName" in fh.read(400_000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true",
                    help="copy the referenced clips into response/clips/ as real files, "
                         "replacing the symlink to the full composite set")
    ap.add_argument("--inline", action="store_true",
                    help="write index.src.html/consent.src.html with all media, styles and "
                         "PDFs embedded — the input staticrypt encrypts for the gated deploy")
    args = ap.parse_args()

    items_by_id = {it["id"]: it for it in load_items()}
    missing = [i for i in PICKS if i not in items_by_id]
    if missing:
        sys.exit(f"unknown item ids in PICKS: {missing}")

    if not args.inline:
        for p in (OUT_HTML, OUT_CONSENT):
            if looks_encrypted(p):
                sys.exit(f"{os.path.basename(p)} is staticrypt output — refusing to overwrite it "
                         "with a plain build. Use --inline, then re-encrypt.")

    if args.final:
        # Replace the whole-composites symlink with just the clips this page loads,
        # so the directory is self-contained and deployable (tens of MB, not 456).
        dest = os.path.join(HERE, "clips")
        if os.path.islink(dest):
            os.unlink(dest)
        os.makedirs(dest, exist_ok=True)
        want = referenced_clips(items_by_id)
        for fn in want:
            shutil.copy2(os.path.join(COMPOSITES, fn), os.path.join(dest, fn))
        for stale in set(os.listdir(dest)) - set(want):
            os.remove(os.path.join(dest, stale))
        print(f"  copied {len(want)} clips into response/clips/")

    page = (PAGE
            .replace("__CONSENT__", consent_section())
            .replace("__CARDS__", sample_rows(items_by_id)))
    consent = consent_page(inline=args.inline)

    if args.inline:
        clips = referenced_clips(items_by_id)
        page = inline_assets(page, clips)
        consent = inline_assets(consent, [])
        out_page, out_consent = SRC_HTML, SRC_CONSENT
    else:
        out_page, out_consent = OUT_HTML, OUT_CONSENT

    for path, body in ((out_page, page), (out_consent, consent)):
        with open(path, "w") as fh:
            fh.write(body)

    print(f"wrote {out_page} — {len(PICKS)} qualitative samples "
          f"({os.path.getsize(out_page)/1e6:.1f} MB)")
    print(f"wrote {out_consent} — {len(CONSENT_FORMS)} consent forms "
          f"({os.path.getsize(out_consent)/1e6:.1f} MB)")
    if args.inline:
        # Anything still pointing at a file on disk would be fetchable around the gate.
        # Links between pages (consent.html, ../index.html, ../LICENSE) are meant to stay.
        leak = re.compile(r'(?:src|href|data)="(?!data:|blob:)'
                          r'([^"]*\.(?:mp4|css|pdf|wav|png|jpe?g|svg))"')
        for path in (out_page, out_consent):
            found = sorted(set(leak.findall(open(path).read())))
            if found:
                print(f"  WARNING: {os.path.basename(path)} still references {found}")
        print("  next: staticrypt these two files (see DEPLOY.md)")


if __name__ == "__main__":
    main()
