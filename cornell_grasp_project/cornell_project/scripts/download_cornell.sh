#!/bin/bash
# scripts/download_cornell.sh
# ----------------------------
# Download and extract the Cornell Grasping Dataset.
# No registration required — publicly available.
#
# Usage:
#     bash scripts/download_cornell.sh
#
# What it does:
#   1. Downloads CornellGraspingDataset.zip (~500 MB)
#   2. Extracts into data/raw/cornell/
#   3. Verifies the structure

set -e

DATA_DIR="data/raw/cornell"
ZIP_FILE="data/raw/CornellGraspingDataset.zip"
URL="https://www.cs.cornell.edu/home/ahadi/CornellGraspingDataset.zip"

echo "================================================="
echo "  Cornell Grasping Dataset Downloader"
echo "================================================="
echo ""

mkdir -p data/raw

# ── Check if already downloaded ──────────────────────────────
if [ -d "$DATA_DIR" ] && [ "$(ls -A $DATA_DIR)" ]; then
    DEPTH_COUNT=$(find "$DATA_DIR" -name "*d.tiff" -o -name "*d.png" 2>/dev/null | wc -l)
    LABEL_COUNT=$(find "$DATA_DIR" -name "*cpos.txt" 2>/dev/null | wc -l)
    if [ "$DEPTH_COUNT" -gt 0 ] && [ "$LABEL_COUNT" -gt 0 ]; then
        echo "  ✓ Cornell data already present:"
        echo "    Depth images : $DEPTH_COUNT"
        echo "    Label files  : $LABEL_COUNT"
        echo ""
        echo "  Skipping download. To re-download, delete data/raw/cornell/ first."
        exit 0
    fi
fi

# ── Download ──────────────────────────────────────────────────
echo "  Downloading (~500 MB)..."
echo "  URL: $URL"
echo ""

if command -v wget &> /dev/null; then
    wget -c -q --show-progress "$URL" -O "$ZIP_FILE"
elif command -v curl &> /dev/null; then
    curl -L -C - --progress-bar "$URL" -o "$ZIP_FILE"
else
    echo "  ERROR: Neither wget nor curl found. Install one and retry."
    exit 1
fi

echo ""
echo "  ✓ Download complete."

# ── Extract ───────────────────────────────────────────────────
echo "  Extracting..."
mkdir -p "$DATA_DIR"
unzip -q "$ZIP_FILE" -d data/raw/

# The zip extracts to a folder — rename it to 'cornell' if needed
EXTRACTED=$(find data/raw -maxdepth 1 -mindepth 1 -type d | grep -v cornell | head -1)
if [ -n "$EXTRACTED" ] && [ "$EXTRACTED" != "$DATA_DIR" ]; then
    mv "$EXTRACTED" "$DATA_DIR"
fi

# Remove zip file to save space
rm -f "$ZIP_FILE"

echo "  ✓ Extracted to $DATA_DIR"

# ── Verify ────────────────────────────────────────────────────
echo ""
echo "  Verifying..."
DEPTH_COUNT=$(find "$DATA_DIR" -name "*d.tiff" -o -name "*d.png" 2>/dev/null | wc -l)
LABEL_COUNT=$(find "$DATA_DIR" -name "*cpos.txt" 2>/dev/null | wc -l)
RGB_COUNT=$(find "$DATA_DIR" -name "*r.png" 2>/dev/null | wc -l)

echo "    RGB images   : $RGB_COUNT"
echo "    Depth images : $DEPTH_COUNT"
echo "    Label files  : $LABEL_COUNT"

if [ "$DEPTH_COUNT" -gt 0 ] && [ "$LABEL_COUNT" -gt 0 ]; then
    echo ""
    echo "  ✓ Cornell dataset ready."
    echo ""
    echo "  Next step: python train.py --config configs/baseline.yaml"
else
    echo ""
    echo "  ✗ WARNING: Dataset may be incomplete."
    echo "    Expected ~885 depth images and ~885 label files."
    echo "    Try running this script again."
fi

echo "================================================="
