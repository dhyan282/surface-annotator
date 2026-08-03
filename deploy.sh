#!/usr/bin/env bash
# deploy.sh -- one-shot deploy to a free Hugging Face Space (Docker SDK).
#
# What this does:
#   1. Creates a new public Space on Hugging Face under YOUR account.
#   2. Adds the Space as a git remote.
#   3. Pushes main -> Space, which builds the Docker image and starts the app.
#
# Prerequisites:
#   * Hugging Face account (free): https://huggingface.co/join
#   * HF write token: https://huggingface.co/settings/tokens
#       (Role: Write. Save it somewhere -- you'll paste it below.)
#   * git is installed and your local repo is committed.
#
# Usage (Git Bash on Windows, or any bash on Linux/Mac):
#     bash deploy.sh
#
# The script will prompt you for:
#   - HF username
#   - Space name (the URL will be https://huggingface.co/spaces/<user>/<space>)
#   - HF write token (kept in memory only; not written to disk)
#
# You can override any of those via env vars to make it fully non-interactive:
#     HF_USERNAME=dhyan282 SPACE_NAME=surface-auto-annotator HF_TOKEN=hf_xxx bash deploy.sh
set -euo pipefail

if [ -z "${HF_USERNAME:-}" ]; then
  read -r -p "Hugging Face username: " HF_USERNAME
fi
if [ -z "${SPACE_NAME:-}" ]; then
  read -r -p "Space name (URL: huggingface.co/spaces/${HF_USERNAME}/<name>): " SPACE_NAME
fi
SPACE_NAME="${SPACE_NAME:-surface-auto-annotator}"
if [ -z "${HF_TOKEN:-}" ]; then
  read -r -s -p "HF write token (hf_xxx...): " HF_TOKEN
  echo
fi

if [ -z "$HF_USERNAME" ] || [ -z "$SPACE_NAME" ] || [ -z "$HF_TOKEN" ]; then
  echo "ERROR: username, space name, and token are all required." >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

echo
echo "==> Target: https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
echo "==> Repo:   $REPO_DIR"
echo

# --- 1. Sanity check the local repo ---
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: $REPO_DIR is not a git repo. Run 'git init' first." >&2
  exit 1
fi

# Make sure there is something to push.
if ! git rev-parse HEAD >/dev/null 2>&1; then
  echo "==> No commits yet. Creating initial commit on 'main'..."
  git checkout -B main
  git add -A
  git -c user.name="surface-annotator" -c user.email="noreply@example.com" \
      commit -m "Initial commit: Surface Auto-Annotator"
fi

# --- 2. Create the Space via HF API ---
echo "==> Creating Space (idempotent: skips if it already exists)..."
python - "$HF_USERNAME" "$SPACE_NAME" "$HF_TOKEN" <<'PY'
import json, sys, urllib.request, urllib.error

user, name, token = sys.argv[1], sys.argv[2], sys.argv[3]
url = f"https://huggingface.co/api/spaces/{user}/{name}"
body = json.dumps({
    "name": name,
    "sdk": "docker",
    "private": False,
    "hardware": "cpu-basic",   # free CPU tier
    "storage": "small",        # free storage tier
}).encode("utf-8")
req = urllib.request.Request(
    url, data=body, method="POST",
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    },
)
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        print("    Space created:", resp.status, json.loads(resp.read()).get("id"))
except urllib.error.HTTPError as e:
    if e.code == 409:
        print("    Space already exists -- continuing.")
    else:
        print(f"    HTTP {e.code}: {e.read().decode('utf-8', 'replace')}", file=sys.stderr)
        sys.exit(1)
PY

# --- 3. Wire up the Space as a git remote ---
REMOTE="hf"
REMOTE_URL="https://${HF_USER}:${HF_TOKEN}@huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
# The actual URL form HF accepts is:
REMOTE_URL="https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"

if git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo "==> Remote '$REMOTE' already exists -- updating URL."
  git remote set-url "$REMOTE" "$REMOTE_URL"
else
  echo "==> Adding remote '$REMOTE' -> $REMOTE_URL"
  git remote add "$REMOTE" "$REMOTE_URL"
fi

# Authenticate the push without persisting the token. We use a temporary
# GIT_ASKPASS helper so the token is only kept in process memory.
echo "==> Pushing to Space (this can take a few minutes for the first build)..."
export HF_TOKEN
export GIT_ASKPASS="$REPO_DIR/.hf_askpass.sh"
cat > "$GIT_ASKPASS" <<'EOF'
#!/usr/bin/env bash
case "$1" in
  Username*) echo "$HF_USERNAME" ;;
  Password*) echo "$HF_TOKEN" ;;
esac
EOF
chmod +x "$GIT_ASKPASS"

# Make sure the branch is named 'main' (HF expects main by default).
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$CURRENT_BRANCH" != "main" ]; then
  echo "==> Renaming current branch '$CURRENT_BRANCH' -> 'main' for HF."
  git branch -M main
fi

# Push. We push with --force the first time so the Space gets the right
# initial history even if it was created with a default README.
git push --force "$REMOTE" main

# Clean up the askpass helper.
rm -f "$GIT_ASKPASS"

echo
echo "==> Done! Your Space will start building now."
echo "    URL:      https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
echo "    Logs:     https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}/logs"
echo "    It usually takes 3-5 minutes to build the Docker image and start the app."
echo "    The first request will take ~30s while SegFormer downloads (~44 MB)."
