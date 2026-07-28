#!/usr/bin/env bash
# Encrypt the inlined response pages behind the same staticrypt gate as preview.html.
#
#   python3 response/build_response.py --inline   # produces the .src.html inputs
#   response/encrypt.sh                           # this script
#
# Prompts for the password (never a shell argument, never in history), refuses to
# run on an empty one, and verifies afterwards that the entered password actually
# decrypts what was written — an empty -p silently produces valid-looking output
# that nobody can open, which is worth catching here rather than in a browser.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."          # neurips/, where .staticrypt.json lives
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

[[ -f .staticrypt.json ]] || { echo "no .staticrypt.json here — wrong directory?" >&2; exit 1; }
for f in response/index.src.html response/consent.src.html; do
  [[ -f $f ]] || { echo "missing $f — run build_response.py --inline first" >&2; exit 1; }
done

read -rsp 'staticrypt password: ' PW; echo
read -rsp 'confirm password    : ' PW2; echo
[[ -n $PW ]]      || { echo "password is empty — aborting" >&2; exit 1; }
[[ $PW == "$PW2" ]] || { echo "passwords do not match — aborting" >&2; exit 1; }

for f in index consent; do
  echo "encrypting response/$f.src.html ..."
  npx -y staticrypt "response/$f.src.html" \
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
  mv "$OUT/$f.src.html" "response/$f.html"
done

echo
PW="$PW" python3 - response/index.html response/consent.html <<'PY'
import hashlib, hmac, os, re, sys

pw = os.environ["PW"].encode()
ok = True
for path in sys.argv[1:]:
    h = open(path, encoding="utf-8", errors="replace").read()
    salt = re.search(r'"staticryptSaltUniqueVariableName"\s*:\s*"([0-9a-f]+)"', h).group(1)
    signed = re.search(r'"staticryptEncryptedMsgUniqueVariableName"\s*:\s*"([0-9a-zA-Z+/=]+)"', h).group(1)
    mac, msg = signed[:64], signed[64:]
    s = salt.encode()
    # staticrypt derives the key in three widening rounds, for backward compatibility.
    k = hashlib.pbkdf2_hmac("sha1",   pw,          s,   1000, 32).hex()
    k = hashlib.pbkdf2_hmac("sha256", k.encode(),  s,  14000, 32).hex()
    k = hashlib.pbkdf2_hmac("sha256", k.encode(),  s, 585000, 32).hex()
    good = hmac.compare_digest(
        hmac.new(bytes.fromhex(k), msg.encode(), hashlib.sha256).hexdigest(), mac)
    ok &= good
    print(f"  {'OK  ' if good else 'FAIL'} {path}  ({os.path.getsize(path)/1e6:.1f} MB, salt {salt[:8]}…)")
sys.exit(0 if ok else 1)
PY

echo
echo "Both pages encrypt and verify against the password you entered."
echo "Sanity-check in a browser, then commit response/*.html (the .src.html stay ignored)."
