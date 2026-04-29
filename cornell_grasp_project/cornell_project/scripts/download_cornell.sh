#!/usr/bin/env bash
# scripts/download_cornell.sh
# -----------------------------------------
# Robust Cornell Grasping Dataset downloader.
#
# What it does:
#   1. Tries multiple download mirrors
#   2. Downloads the archive to data/raw/
#   3. Extracts into data/raw/cornell/
#   4. Verifies expected Cornell files exist
#
# Usage:
#   bash scripts/download_cornell.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${ROOT_DIR}/data/raw"
TARGET_DIR="${RAW_DIR}/cornell"
TMP_DIR="${RAW_DIR}/_cornell_tmp"
ZIP_FILE="${RAW_DIR}/cornell_dataset.zip"

echo "================================================="
echo "  Cornell Grasping Dataset Downloader"
echo "================================================="
echo

mkdir -p "${RAW_DIR}"

# ── Check whether dataset already exists ─────────────────────
if [ -d "${TARGET_DIR}" ] && [ "$(find "${TARGET_DIR}" -type f | head -n 1)" ]; then
    RGB_COUNT="$(find "${TARGET_DIR}" -type f -name "*r.png" 2>/dev/null | wc -l)"
    DEPTH_COUNT="$(find "${TARGET_DIR}" -type f \( -name "*d.tiff" -o -name "*d.png" \) 2>/dev/null | wc -l)"
    POS_COUNT="$(find "${TARGET_DIR}" -type f -name "*cpos.txt" 2>/dev/null | wc -l)"
    NEG_COUNT="$(find "${TARGET_DIR}" -type f -name "*cneg.txt" 2>/dev/null | wc -l)"

    if [ "${DEPTH_COUNT}" -gt 0 ] && [ "${POS_COUNT}" -gt 0 ]; then
        echo "  Cornell dataset already found:"
        echo "    RGB images      : ${RGB_COUNT}"
        echo "    Depth images    : ${DEPTH_COUNT}"
        echo "    Positive labels : ${POS_COUNT}"
        echo "    Negative labels : ${NEG_COUNT}"
        echo
        echo "  Skipping download."
        echo "  Delete '${TARGET_DIR}' first if you want to re-download."
        exit 0
    fi
fi

# ── Candidate mirrors ────────────────────────────────────────
URLS=(
    "https://github.com/skumra/robotic-grasping/raw/master/data/cornell/data.zip"
)

download_with_curl() {
    local url="$1"
    echo "  Trying curl: ${url}"
    curl -L --fail --retry 3 --retry-delay 2 -o "${ZIP_FILE}" "${url}"
}

download_with_wget() {
    local url="$1"
    echo "  Trying wget: ${url}"
    wget -O "${ZIP_FILE}" "${url}"
}

download_ok=0

echo "  Attempting download..."
echo

for url in "${URLS[@]}"; do
    rm -f "${ZIP_FILE}"

    if command -v curl >/dev/null 2>&1; then
        if download_with_curl "${url}"; then
            download_ok=1
            break
        else
            echo "  curl failed for this mirror."
        fi
    fi

    if command -v wget >/dev/null 2>&1; then
        if download_with_wget "${url}"; then
            download_ok=1
            break
        else
            echo "  wget failed for this mirror."
        fi
    fi
done

if [ "${download_ok}" -ne 1 ]; then
    echo
    echo "ERROR: Could not download the Cornell dataset from any configured mirror."
    echo
    echo "Please download it manually and extract it into:"
    echo "  ${TARGET_DIR}"
    echo
    echo "Expected files include:"
    echo "  pcd0100r.png"
    echo "  pcd0100d.tiff"
    echo "  pcd0100cpos.txt"
    echo "  pcd0100cneg.txt"
    exit 1
fi

echo
echo "  Download complete."

# ── Clean previous partial extraction ────────────────────────
rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"
rm -rf "${TARGET_DIR}"

# ── Extract ──────────────────────────────────────────────────
if ! command -v unzip >/dev/null 2>&1; then
    echo "ERROR: 'unzip' is not installed."
    echo "Install unzip and re-run the script."
    exit 1
fi

echo "  Extracting..."
unzip -q "${ZIP_FILE}" -d "${TMP_DIR}"

# ── Find extracted Cornell content ───────────────────────────
# We look for the directory that contains Cornell annotation files.
CANDIDATE_FILE="$(find "${TMP_DIR}" -type f -name "*cpos.txt" | head -n 1 || true)"

if [ -z "${CANDIDATE_FILE}" ]; then
    echo
    echo "ERROR: Extraction finished, but no Cornell annotation files were found."
    echo "Please inspect:"
    echo "  ${TMP_DIR}"
    exit 1
fi

CANDIDATE_DIR="$(dirname "${CANDIDATE_FILE}")"

# If files are already at top level under TMP_DIR, use TMP_DIR.
# Otherwise use the highest directory under TMP_DIR that appears to be the dataset root.
# We want the final TARGET_DIR to directly contain pcd*. files.
while [ "$(dirname "${CANDIDATE_DIR}")" != "${TMP_DIR}" ] && [ "${CANDIDATE_DIR}" != "${TMP_DIR}" ]; do
    PARENT="$(dirname "${CANDIDATE_DIR}")"
    if find "${PARENT}" -maxdepth 1 -type f -name "*cpos.txt" | grep -q .; then
        CANDIDATE_DIR="${PARENT}"
    else
        break
    fi
done

mkdir -p "${TARGET_DIR}"

shopt -s dotglob nullglob
cp -r "${CANDIDATE_DIR}/"* "${TARGET_DIR}/"
shopt -u dotglob nullglob

# ── Remove archive to save space ─────────────────────────────
rm -f "${ZIP_FILE}"
rm -rf "${TMP_DIR}"

# ── Verify ───────────────────────────────────────────────────
echo
echo "  Verifying dataset structure..."

RGB_COUNT="$(find "${TARGET_DIR}" -type f -name "*r.png" 2>/dev/null | wc -l)"
DEPTH_COUNT="$(find "${TARGET_DIR}" -type f \( -name "*d.tiff" -o -name "*d.png" \) 2>/dev/null | wc -l)"
POS_COUNT="$(find "${TARGET_DIR}" -type f -name "*cpos.txt" 2>/dev/null | wc -l)"
NEG_COUNT="$(find "${TARGET_DIR}" -type f -name "*cneg.txt" 2>/dev/null | wc -l)"

echo "    RGB images      : ${RGB_COUNT}"
echo "    Depth images    : ${DEPTH_COUNT}"
echo "    Positive labels : ${POS_COUNT}"
echo "    Negative labels : ${NEG_COUNT}"

if [ "${DEPTH_COUNT}" -gt 0 ] && [ "${POS_COUNT}" -gt 0 ]; then
    echo
    echo "  Cornell dataset is ready at:"
    echo "    ${TARGET_DIR}"
    echo
    echo "  Quick sanity check:"
    find "${TARGET_DIR}" -maxdepth 1 -type f | head -n 8 || true
else
    echo
    echo "WARNING: Dataset extraction may be incomplete."
    echo "Expected Cornell files were not detected correctly."
    echo "Please inspect:"
    echo "  ${TARGET_DIR}"
    exit 1
fi

echo
echo "================================================="
