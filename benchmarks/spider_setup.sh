#!/usr/bin/env bash
# Download and prepare the Spider 1.0 dev set for benchmarking.
#
# Usage:
#   bash benchmarks/spider_setup.sh
#
# Creates:
#   benchmarks/spider/
#     dev.json          # 1,034 (question, gold SQL, db_id) triples
#     database/         # 166 SQLite databases
#     tables.json       # schema metadata for every database
#
# Size: ~95 MB compressed, ~300 MB uncompressed.
# Only the dev set is used — the training set is not downloaded.

set -euo pipefail

DEST="$(cd "$(dirname "$0")" && pwd)/spider"
ZIP_URL="https://drive.usercontent.google.com/download?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&confirm=t"

if [ -f "$DEST/dev.json" ] && [ -d "$DEST/database" ]; then
    echo "Spider dev set already present at $DEST — skipping download."
    echo "To force re-download, remove $DEST and re-run."
    exit 0
fi

mkdir -p "$DEST"
echo "Downloading Spider dataset (~95 MB) ..."

# Google Drive may serve a virus-scan warning page for large files.
# The &confirm=t parameter bypasses it for the direct download link.
TMPZIP="$DEST/_spider.zip"
if command -v wget &>/dev/null; then
    wget -q --show-progress -O "$TMPZIP" "$ZIP_URL"
elif command -v curl &>/dev/null; then
    curl -L -o "$TMPZIP" "$ZIP_URL"
else
    echo "Error: wget or curl required." >&2
    exit 1
fi

echo "Extracting ..."
unzip -q -o "$TMPZIP" -d "$DEST/_tmp"

# The zip may contain a top-level spider/ directory — flatten it.
INNER=$(find "$DEST/_tmp" -maxdepth 2 -name "dev.json" -print -quit)
if [ -z "$INNER" ]; then
    echo "Error: dev.json not found in downloaded archive." >&2
    exit 1
fi
INNER_DIR="$(dirname "$INNER")"

mv "$INNER_DIR/dev.json" "$DEST/dev.json"
mv "$INNER_DIR/tables.json" "$DEST/tables.json" 2>/dev/null || true

# The database directory may be called database/ or databases/
if [ -d "$INNER_DIR/database" ]; then
    mv "$INNER_DIR/database" "$DEST/database"
elif [ -d "$INNER_DIR/databases" ]; then
    mv "$INNER_DIR/databases" "$DEST/database"
fi

rm -rf "$DEST/_tmp" "$TMPZIP"

N=$(python3 -c "import json; print(len(json.load(open('$DEST/dev.json'))))")
echo "Done. $N dev examples in $DEST/dev.json"
echo "Databases in $DEST/database/"
