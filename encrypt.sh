#!/usr/bin/env bash
# Re-encrypt every staticrypt-gated page behind a single shared password.
#
#   python3 response/build_response.py --inline   # produces the response/*.src.html inputs
#   ./encrypt.sh                                  # this script
#
# All three gated pages use one salt and one password, so they must always be
# encrypted together — rotating only some of them leaves the site answering to two
# different passwords.
#
# The password is prompted for, never a shell argument and never in history. The
# script refuses an empty password, refuses a password that is already the current
# one (a no-op rotation that would otherwise look like a successful one), and
# verifies that what it wrote actually decrypts before moving anything into place.
# An empty or mistyped -p otherwise produces valid-looking output that nobody can
# open, which is worth catching here rather than in a browser.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"              # neurips/, where .staticrypt.json lives
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

SRCS=(preview.src.html response/index.src.html response/consent.src.html)
DSTS=(preview.html     response/index.html     response/consent.html)

[[ -f .staticrypt.json ]] || { echo "no .staticrypt.json here — wrong directory?" >&2; exit 1; }
for f in "${SRCS[@]}"; do
  [[ -f $f ]] || { echo "missing $f — run build_response.py --inline first" >&2; exit 1; }
done

# Verify a password against already-encrypted pages, without needing to decrypt them.
# usage: check_pw <password> <file>...   — exit 0 only if every file opens with it.
check_pw() {
  local pw=$1; shift
  STATICRYPT_PW=$pw python3 - "$@" <<'PY'
import hashlib, hmac, os, re, sys

pw = os.environ["STATICRYPT_PW"].encode()
strip = os.environ.get("STATICRYPT_STRIP", "")
quiet = os.environ.get("STATICRYPT_QUIET") == "1"
ok = True

for path in sys.argv[1:]:
    h = open(path, encoding="utf-8", errors="replace").read()
    m_salt = re.search(r'"staticryptSaltUniqueVariableName"\s*:\s*"([0-9a-f]+)"', h)
    m_msg = re.search(r'"staticryptEncryptedMsgUniqueVariableName"\s*:\s*"([0-9a-zA-Z+/=]+)"', h)
    if not (m_salt and m_msg):
        print(f"  FAIL {path} — not a staticrypt page?")
        ok = False
        continue
    salt, signed = m_salt.group(1), m_msg.group(1)
    mac, msg = signed[:64], signed[64:]
    s = salt.encode()
    # staticrypt derives the key in three widening rounds, for backward compatibility.
    k = hashlib.pbkdf2_hmac("sha1",   pw,          s,   1000, 32).hex()
    k = hashlib.pbkdf2_hmac("sha256", k.encode(),  s,  14000, 32).hex()
    k = hashlib.pbkdf2_hmac("sha256", k.encode(),  s, 585000, 32).hex()
    good = hmac.compare_digest(
        hmac.new(bytes.fromhex(k), msg.encode(), hashlib.sha256).hexdigest(), mac)
    ok &= good
    if not quiet:
        label = path[len(strip):].lstrip("/") if strip and path.startswith(strip) else path
        print(f"  {'OK  ' if good else 'FAIL'} {label}  "
              f"({os.path.getsize(path)/1e6:.1f} MB, salt {salt[:8]}…)")

sys.exit(0 if ok else 1)
PY
}

read -rsp 'new staticrypt password: ' PW; echo
read -rsp 'confirm password       : ' PW2; echo
[[ -n $PW ]]        || { echo "password is empty — aborting" >&2; exit 1; }
[[ $PW == "$PW2" ]] || { echo "passwords do not match — aborting" >&2; exit 1; }

# Catch a no-op rotation: if the new password already opens the live page, nothing
# would actually change and every check below would still pass.
if [[ -f ${DSTS[0]} ]] && STATICRYPT_QUIET=1 check_pw "$PW" "${DSTS[0]}" 2>/dev/null; then
  echo "that is already the current password — pick a different one" >&2
  exit 1
fi

mkdir -p "$OUT/response"
for i in "${!SRCS[@]}"; do
  src=${SRCS[$i]} dst=${DSTS[$i]}
  echo "encrypting $src ..."
  npx -y staticrypt "$src" \
    -p "$PW" \
    --short --remember 0 \
    --template-title "VideoFDB — Dataset Access" \
    --template-instructions "Enter the access password from the submission materials." \
    --template-button "Unlock" \
    --template-placeholder "Password" \
    --template-error "Incorrect password." \
    --template-color-primary "#1a4480" \
    --template-color-secondary "#fafaf7" \
    -d "$OUT" >/dev/null
  # staticrypt flattens its output: no response/ subdirectory is recreated.
  mv "$OUT/$(basename "$src")" "$OUT/$dst"
done

# Verify in the staging directory, so a bad password never overwrites a working site.
echo
STATICRYPT_STRIP=$OUT check_pw "$PW" "${DSTS[@]/#/$OUT/}"

for i in "${!DSTS[@]}"; do mv "$OUT/${DSTS[$i]}" "${DSTS[$i]}"; done

echo
echo "All ${#DSTS[@]} pages re-encrypted and verified against the password you entered."
echo "Sanity-check in a browser, then commit (the .src.html stay ignored)."
