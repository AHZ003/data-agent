#!/usr/bin/env bash
# Download and prepare BIRD mini-dev (500 questions, 11 SQLite databases).
#
# Usage:
#   bash benchmarks/bird_setup.sh
#
# Creates:
#   benchmarks/bird/
#     mini_dev_sqlite.json   # question, evidence, SQL, db_id, difficulty
#     dev_databases/<db_id>/<db_id>.sqlite
#
# Size: ~800 MB download. Source: https://github.com/bird-bench/mini_dev

set -euo pipefail

DEST="$(cd "$(dirname "$0")" && pwd)/bird"
URL="https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"

if [ -f "$DEST/mini_dev_sqlite.json" ] && [ -d "$DEST/dev_databases" ]; then
    echo "BIRD mini-dev already present at $DEST — skipping download."
    exit 0
fi

mkdir -p "$DEST"
TMPZIP="$DEST/_minidev.zip"
echo "Downloading BIRD mini-dev (~800 MB) ..."
curl -L --fail -o "$TMPZIP" "$URL"

echo "Extracting ..."
unzip -q -o "$TMPZIP" -d "$DEST/_tmp"

JSON=$(find "$DEST/_tmp" -name "mini_dev_sqlite.json" -print -quit)
DBS=$(find "$DEST/_tmp" -type d -name "dev_databases" -print -quit)
if [ -z "$JSON" ] || [ -z "$DBS" ]; then
    echo "Error: expected mini_dev_sqlite.json and dev_databases/ in the archive." >&2
    exit 1
fi
mv "$JSON" "$DEST/mini_dev_sqlite.json"
mv "$DBS" "$DEST/dev_databases"
# Some releases ship the databases as nested zips.
find "$DEST/dev_databases" -name "*.zip" -exec sh -c 'unzip -q -o "$1" -d "$(dirname "$1")" && rm "$1"' _ {} \;

rm -rf "$DEST/_tmp" "$TMPZIP"
N=$(python3 -c "import json; print(len(json.load(open('$DEST/mini_dev_sqlite.json'))))")
echo "Done. $N questions in $DEST/mini_dev_sqlite.json"
