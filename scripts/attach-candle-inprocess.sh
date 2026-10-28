#!/usr/bin/env bash
# Prepare the audited patched Candle crate for the isolated in-process adapter.
# Cargo path dependencies inherit the nearest containing workspace unless
# package.workspace explicitly points at the original upstream manifest.
set -euo pipefail
if (( $# != 2 )); then
  echo "usage: bash scripts/attach-candle-inprocess.sh PATCHED_SOURCE VENDOR_DEST" >&2
  exit 2
fi
source_dir="$(realpath "$1")"
dest="$2"
pinned="dbf78c816cd83c89f11dbeb9d87290ad0a3dccd0"
if [[ ! -f "$source_dir/Cargo.toml" ]] || [[ ! -f "$source_dir/crates/pocket-tts/Cargo.toml" ]]; then
  echo "missing original Candle workspace manifests" >&2
  exit 1
fi
if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$pinned" ]]; then
  echo "refusing unpinned upstream Candle source" >&2
  exit 1
fi
if [[ -e "$dest" || -L "$dest" ]]; then
  echo "destination already exists: $dest" >&2
  exit 2
fi
mkdir -p "$(dirname "$dest")"
cp -a "$source_dir" "$dest"
python3 - "$dest/crates/pocket-tts/Cargo.toml" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
data = path.read_text()
# The patched upstream's [package] currently relies on workspace inheritance.
# This explicit relative path is from crates/pocket-tts to the upstream root.
marker = "[package]\n"
if data.count(marker) != 1:
    raise SystemExit("unexpected upstream Candle package manifest shape")
section = data.split("[dependencies]", 1)[0]
if "workspace = " in section:
    raise SystemExit("upstream already sets a workspace root; review manually")
path.write_text(data.replace(marker, '[package]\nworkspace = "../.."\n', 1))
print("Candle package inherits its OWN upstream workspace, not the adapter workspace")
PY
test "$(git -C "$dest" rev-parse HEAD)" = "$pinned"
