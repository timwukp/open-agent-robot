#!/usr/bin/env bash
# Fetch the Unitree G1 (with hands) model from MuJoCo Menagerie (BSD/Apache
# licensed models) into body/assets/ via sparse checkout — the full Menagerie
# is ~1GB; we need one robot.
set -euo pipefail
cd "$(dirname "$0")"
if [ -d assets/unitree_g1 ]; then
  echo "assets/unitree_g1 already present"; exit 0
fi
tmp=$(mktemp -d)
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/google-deepmind/mujoco_menagerie.git "$tmp"
git -C "$tmp" sparse-checkout set unitree_g1
mkdir -p assets
cp -R "$tmp/unitree_g1" assets/
rm -rf "$tmp"
echo "fetched: $(ls assets/unitree_g1/*.xml | tr '\n' ' ')"
