#!/usr/bin/env bash
# Reproducibly assemble the audited Pocket TTS Candle v4 candidate.
# This is a build-time source checkout, NOT yet vendored/in-process inference.
set -euo pipefail
if (( $# != 1 )); then
  echo "usage: bash scripts/prepare-candle-v4.sh DESTINATION" >&2
  exit 2
fi
dest="$1"
if [[ -e "$dest" ]]; then
  echo "destination already exists: $dest" >&2
  exit 2
fi
mkdir -p "$(dirname "$dest")"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pinned="dbf78c816cd83c89f11dbeb9d87290ad0a3dccd0"
git clone --quiet https://github.com/stepupgaming/pocket-tts.git "$dest"
if [[ "$(git -C "$dest" rev-parse HEAD)" != "$pinned" ]]; then
  git -C "$dest" checkout --quiet "$pinned"
fi
if [[ "$(git -C "$dest" rev-parse HEAD)" != "$pinned" ]]; then
  echo "audited source revision mismatch" >&2
  exit 1
fi
crate="$dest/crates/pocket-tts"
mkdir -p "$crate/config" "$crate/examples"
cp "$root/crates/pocket-tts-core/tests/fixtures/english-v4.yaml" "$crate/config/probe-english.yaml"
cp "$root/crates/pocket-tts-core/tests/fixtures/english-7m-v4.yaml" "$crate/config/probe-7m.yaml"
cp "$root/crates/pocket-tts-core/tests/fixtures/english-18m-refl-v4.yaml" "$crate/config/probe-18m.yaml"
cp "$root/benchmarks/candle_v4_probe.rs" "$crate/examples/candle_v4_probe.rs"
for patch in \
  0001_mimi_downsample.py \
  0002_speaker_projection.py \
  0003_low_rank_lookup.py \
  0004_drifting_sampler.py \
  0005_z_projection.py \
  0006_asymmetric_mimi.py \
  0007_mimi_projection_bridge.py \
  0008_grouped_query_attention.py \
  0009_voice_bos_prefix.py; do
  python3 "$root/patches/candle/$patch" "$dest"
done
cargo fmt --manifest-path "$dest/Cargo.toml" --all
printf 'CANDLE_SOURCE=%s\nCANDLE_BASE_REVISION=%s\n' "$dest" "$pinned"
