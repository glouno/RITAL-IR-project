#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/submission}"
ZIP_NAME="${2:-rital-ir-project-submission.zip}"
STAGING_DIR="$OUT_DIR/rital-ir-project"

rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR" "$OUT_DIR"

rsync -a "$ROOT_DIR"/ "$STAGING_DIR"/ \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude ".env" \
  --exclude ".runs/" \
  --exclude ".pytest_cache/" \
  --exclude ".mypy_cache/" \
  --exclude ".ruff_cache/" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude "data/" \
  --exclude "tmp/" \
  --exclude "legacy/" \
  --exclude "submission/" \
  --exclude ".DS_Store"

(
  cd "$OUT_DIR"
  rm -f "$ZIP_NAME"
  zip -qr "$ZIP_NAME" "rital-ir-project"
)

echo "$OUT_DIR/$ZIP_NAME"
