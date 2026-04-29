#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data/raw"
TARGET_DIR="${DATA_DIR}/cornell"
TMP_DIR="${DATA_DIR}/_tmp_cornell_download"
ZIP_PATH="${TMP_DIR}/cornell.zip"

mkdir -p "${DATA_DIR}"
rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"

if [ -d "${TARGET_DIR}" ]; then
  echo "[INFO] Cornell dataset directory already exists:"
  echo "       ${TARGET_DIR}"
  echo "[INFO] Remove it first if you want to re-download."
  exit 0
fi

download_ok=0

try_download() {
  local url="$1"
  echo "[INFO] Trying: ${url}"
  if command -v curl >/dev/null 2>&1; then
    if curl -L --fail --retry 3 --retry-delay 2 -o "${ZIP_PATH}" "${url}"; then
      return 0
    fi
  fi
  if command -v wget >/dev/null 2>&1; then
    if wget -O "${ZIP_PATH}" "${url}"; then
      return 0
    fi
  fi
  return 1
}

# Candidate mirrors.
# Keep this list short and easy to maintain.
URLS=(
  "https://github.com/skumra/robotic-grasping/raw/master/data/cornell/data.zip"
)

for url in "${URLS[@]}"; do
  if try_download "${url}"; then
    download_ok=1
    break
  else
    echo "[WARN] Failed: ${url}"
    rm -f "${ZIP_PATH}"
  fi
done

if [ "${download_ok}" -ne 1 ]; then
  echo
  echo "[ERROR] Could not download the Cornell dataset automatically."
  echo
  echo "Please do one of the following:"
  echo "  1. Download it manually from a working mirror."
  echo "  2. Place the extracted dataset under:"
  echo "       ${TARGET_DIR}"
  echo
  echo "Expected files look like:"
  echo "  pcd0100r.png"
  echo "  pcd0100d.tiff"
  echo "  pcd0100cpos.txt"
  echo "  pcd0100cneg.txt"
  echo
  exit 1
fi

echo "[INFO] Download finished."

if ! command -v unzip >/dev/null 2>&1; then
  echo "[ERROR] unzip is not installed."
  echo "Install it, then extract manually:"
  echo "  unzip ${ZIP_PATH} -d ${TMP_DIR}"
  exit 1
fi

echo "[INFO] Extracting archive..."
unzip -q "${ZIP_PATH}" -d "${TMP_DIR}"

# Try to locate the extracted dataset root.
# We look for a directory containing Cornell-style files.
CANDIDATE_DIR="$(find "${TMP_DIR}" -type f \( -name '*cpos.txt' -o -name '*cneg.txt' \) | head -n 1 | xargs -r dirname)"

if [ -z "${CANDIDATE_DIR}" ]; then
  echo "[ERROR] Downloaded archive does not look like a Cornell dataset."
  echo "Please inspect: ${TMP_DIR}"
  exit 1
fi

# Move the nearest useful parent into final location.
# If archive extracted to data/, rename it to cornell.
DATA_PARENT="${TMP_DIR}/data"
if [ -d "${DATA_PARENT}" ]; then
  mv "${DATA_PARENT}" "${TARGET_DIR}"
else
  mkdir -p "${TARGET_DIR}"
  cp -r "${TMP_DIR}/." "${TARGET_DIR}/"
fi

# Flatten one nested level if needed.
if [ -d "${TARGET_DIR}/data" ] && [ ! -f "${TARGET_DIR}/pcd0100cpos.txt" ]; then
  shopt -s dotglob nullglob
  mv "${TARGET_DIR}/data/"* "${TARGET_DIR}/"
  rmdir "${TARGET_DIR}/data" || true
  shopt -u dotglob nullglob
fi

rm -rf "${TMP_DIR}"

echo "[INFO] Cornell dataset prepared at:"
echo "       ${TARGET_DIR}"
echo

echo "[INFO] Quick verification:"
find "${TARGET_DIR}" -maxdepth 1 -type f | head -n 10 || true
echo
echo "[INFO] To verify annotation files:"
echo "  find ${TARGET_DIR} -name '*cpos.txt' | head"
