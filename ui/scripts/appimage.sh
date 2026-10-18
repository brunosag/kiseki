#!/usr/bin/env bash
# Builds ui/release/kiseki-<version>-x86_64.AppImage: the UI, the Python payload (the files
# Remote.deploy ships, plus ui/dist) under a BUILD id that names its directory on the user's
# machine, and a pinned uv. Needs git, curl, sha256sum, pnpm and the installed packages.
set -euo pipefail
cd "$(dirname "$0")/.."
UV_VERSION=0.13.1
UV_SHA256=6c86040a8f045ab9adc9e3ec4cdafa03fd00e9a988ed2bf28971c7fc2d0debd0

pnpm build
rm -rf build
mkdir -p build/kiseki/ui
git -C .. ls-files -z src pyproject.toml uv.lock .python-version README.md |
  (cd .. && xargs -0 cp --parents -t ui/build/kiseki)
cp -r dist build/kiseki/ui/dist
version="$(node -p 'require("./package.json").version')"
digest="$(cd build/kiseki && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-12)"
echo "$version-$digest" >build/kiseki/BUILD

curl -fsSL -o build/uv.tar.gz \
  "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-musl.tar.gz"
echo "$UV_SHA256  build/uv.tar.gz" | sha256sum -c --quiet
tar -xzf build/uv.tar.gz -C build --strip-components=1 uv-x86_64-unknown-linux-musl/uv

pnpm exec electron-builder --linux AppImage --config electron-builder.config.cjs
