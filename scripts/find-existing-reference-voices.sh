#!/usr/bin/env bash
# Discover private reference WAVs on the existing Linux runner, or clone them
# with *already configured* Git credentials. Never prompts, requests a new PAT,
# or exports the private corpus to this public repository.
set -euo pipefail
: "${RUNNER_TEMP:?RUNNER_TEMP is required}"
: "${GITHUB_ENV:?GITHUB_ENV is required}"

valid_speech_dir() {
  local dir="$1" n
  [[ -d "$dir" ]] || return 1
  for n in 1 2 3 4 5; do
    [[ -s "$dir/reference_$n.wav" && -s "$dir/reference_$n.txt" ]] || return 1
  done
}
export_source() {
  local dir="$1" kind="$2"
  valid_speech_dir "$dir" || return 1
  # This path stays inside runner-local GITHUB_ENV; no audio, transcript,
  # identifier, or sensitive path is echoed to Actions logs.
  printf 'PRIVATE_REFERENCE_SPEECH_DIR=%s\n' "$dir" >> "$GITHUB_ENV"
  printf 'REFERENCE_CORPUS_FOUND source=%s count=5\n' "$kind"
}
if [[ -n "${POCKET_TTS_REFERENCE_DIR:-}" ]]; then
  if export_source "$POCKET_TTS_REFERENCE_DIR" existing; then exit 0; fi
  if export_source "$POCKET_TTS_REFERENCE_DIR/speech" existing; then exit 0; fi
fi
# Known possible local checkouts or mounted reference directories. This does
# not copy, chmod, modify, or delete any user-owned reference file.
for root in "$HOME/reference" "$HOME/references" "$HOME/repos/reference" \
            "$HOME/git/reference" "$HOME/projects/reference" \
            "$HOME/Projects/reference" "$HOME/Documents/reference" \
            "$HOME/.cache/reference" "$HOME/.cache/references" \
            "${GITHUB_WORKSPACE:-}/../reference"; do
  if export_source "$root/speech" existing; then exit 0; fi
done
# Small, bounded discovery in the runner user's home. Avoid scanning all
# mounted drives, exposing private path names or traversing other users.
while IFS= read -r -d '' wav; do
  dir="$(dirname "$wav")"
  if export_source "$dir" discovered; then exit 0; fi
done < <(find "$HOME" -maxdepth 5 -type f -name 'reference_1.wav' \
            -not -path '*/.git/*' -print0 2>/dev/null)

# If the machine already has gh authentication or a configured Git/SSH key,
# reuse it noninteractively to fetch the private repository. No extra token
# is created, requested or embedded in a command line. Never print Git logs.
target="$RUNNER_TEMP/pocket-private-reference-${GITHUB_RUN_ID}"
rm -rf -- "$target"
clone_success=false
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  if timeout 60s gh repo clone unistorm10/reference "$target" -- --depth 1 \
       </dev/null >/dev/null 2>&1; then
    clone_success=true
  fi
fi
if [[ "$clone_success" == false ]]; then
  rm -rf -- "$target"
  if GIT_TERMINAL_PROMPT=0 timeout 60s \
      git -c credential.interactive=never clone --quiet --depth 1 \
      https://github.com/unistorm10/reference.git "$target" \
      </dev/null >/dev/null 2>&1; then
    clone_success=true
  fi
fi
if [[ "$clone_success" == false ]]; then
  rm -rf -- "$target"
  if GIT_TERMINAL_PROMPT=0 \
     GIT_SSH_COMMAND='ssh -oBatchMode=yes -oStrictHostKeyChecking=yes -oConnectTimeout=8' \
     timeout 40s git clone --quiet --depth 1 \
       git@github.com:unistorm10/reference.git "$target" \
       </dev/null >/dev/null 2>&1; then
    clone_success=true
  fi
fi
if [[ "$clone_success" == true ]] && export_source "$target/speech" existing_git_auth; then
  exit 0
fi
rm -rf -- "$target"
echo "::error::Five reference WAVs are not installed on this existing runner, and no existing gh/Git/SSH credential can read unistorm10/reference. The test was NOT run. No new runner or token was requested." >&2
exit 1
