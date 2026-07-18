#!/usr/bin/env bash
# Fetch the open datasets Fisura uses into the local data vault (idempotent; nothing enters git).
# Root: $FISURA_DATA_ROOT, else .env FISURA_DATA_ROOT, else <repo>/data/raw-local (gitignored).
# Sources, licenses and redistribution rulings: data/README.md (the registry). URLs verified 2026-07-18.
# Gated sets (MVTec AD/AD2, GAPs, OmniCrack30k, Severstal, CrackVision12K) need a one-time manual step.
set -u

repo="$(cd "$(dirname "$0")/.." && pwd)"
root="${FISURA_DATA_ROOT:-}"
if [ -z "$root" ] && [ -f "$repo/.env" ]; then
  root="$(sed -n 's/^FISURA_DATA_ROOT=//p' "$repo/.env" | head -1)"
fi
[ -z "$root" ] && root="$repo/data/raw-local"
raw="$root/raw"
mkdir -p "$raw"
echo "fetch-data: vault root = $root"

has_content() { [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }
get() { # url dir name
  mkdir -p "$2"
  local dest="$2/$3"
  if [ -s "$dest" ]; then echo "SKIP $3"; return 0; fi
  echo "GET  $3"
  curl -sS -L --fail --retry 3 --retry-delay 10 -C - -o "$dest" "$1" || echo "FAIL $3"
}

# --- Direct downloads ---
has_content "$raw/bcl"          || get 'https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/RURXSH' "$raw/bcl" 'bcl_dataverse.zip'
has_content "$raw/crackseg9k"   || get 'https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/EGIEBY' "$raw/crackseg9k" 'crackseg9k_dataverse.zip'
has_content "$raw/deepcrack537" || get 'https://github.com/yhlleo/DeepCrack/raw/master/dataset/DeepCrack.zip' "$raw/deepcrack537" 'DeepCrack.zip'
has_content "$raw/visa"         || get 'https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar' "$raw/visa" 'VisA_20220922.tar'
has_content "$raw/kolektorsdd"  || get 'https://go.vicos.si/kolektorsdd' "$raw/kolektorsdd" 'KolektorSDD.zip'
has_content "$raw/kolektorsdd2" || get 'https://go.vicos.si/kolektorsdd2' "$raw/kolektorsdd2" 'KolektorSDD2.zip'
if ! has_content "$raw/dacl10k"; then
  get 'https://dacl10k.s3.eu-central-1.amazonaws.com/dacl10k-challenge/dacl10k_v2_devphase.zip' "$raw/dacl10k" 'dacl10k_v2_devphase.zip'
  get 'https://dacl10k.s3.eu-central-1.amazonaws.com/dacl10k-challenge/dacl10k_v2_testchallenge.zip' "$raw/dacl10k" 'dacl10k_v2_testchallenge.zip'
fi
if ! has_content "$raw/codebrim"; then
  get 'https://zenodo.org/records/2620293/files/CODEBRIM_classification_balanced_dataset.zip?download=1' "$raw/codebrim" 'CODEBRIM_classification_balanced_dataset.zip'
  get 'https://zenodo.org/records/2620293/files/CODEBRIM_original_images.zip?download=1' "$raw/codebrim" 'CODEBRIM_original_images.zip'
fi
has_content "$raw/rdd2022" || get 'https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/RDD2022.zip' "$raw/rdd2022" 'RDD2022.zip'

# SDNET2018 (CC BY 4.0): resolve the zip link from the landing page
if ! has_content "$raw/sdnet2018"; then
  href="$(curl -sSL 'https://digitalcommons.usu.edu/all_datasets/48/' | grep -oE 'href="[^"]*(viewcontent|\.zip)[^"]*"' | head -1 | sed 's/href="//; s/"$//')"
  case "$href" in
    http*) get "$href" "$raw/sdnet2018" 'SDNET2018.zip' ;;
    /*)    get "https://digitalcommons.usu.edu$href" "$raw/sdnet2018" 'SDNET2018.zip' ;;
    *)     echo 'FAIL sdnet2018: no download link on landing page' ;;
  esac
fi

# Ozgenel METU (CC BY 4.0): Mendeley public API signed URLs (expire; fetched per run; needs python3 for JSON)
if ! has_content "$raw/ozgenel-metu"; then
  python3 - "$raw/ozgenel-metu" <<'PY' || echo 'FAIL ozgenel (python3 required for the Mendeley API route)'
import json, pathlib, subprocess, sys, urllib.request
dest = pathlib.Path(sys.argv[1]); dest.mkdir(parents=True, exist_ok=True)
url = "https://data.mendeley.com/public-api/datasets/5y9wdsg2zt/files?folder_id=root&version=2"
files = json.load(urllib.request.urlopen(url))
for f in files:
    out = dest / f["filename"]
    if out.exists() and out.stat().st_size > 0:
        print(f"SKIP {f['filename']}"); continue
    print(f"GET  {f['filename']}")
    subprocess.run(["curl", "-sS", "-L", "--fail", "-o", str(out), f["content_details"]["download_url"]], check=False)
PY
fi

# --- Git-clone datasets ---
for pair in \
  'crackforest-cfd https://github.com/cuilimeng/CrackForest-dataset.git' \
  'uav75 https://github.com/ben-z-original/uav75.git' \
  'magnetic-tile https://github.com/abin24/Magnetic-tile-defect-datasets..git' \
  'masonry-dais https://github.com/dimitrisdais/crack_detection_CNN_masonry.git'; do
  name="${pair%% *}"; url="${pair#* }"
  if has_content "$raw/$name"; then echo "SKIP $name"; else echo "CLONE $name"; git clone --depth 1 "$url" "$raw/$name"; fi
done

# --- Kaggle mirrors (need kaggle CLI + auth) ---
if command -v kaggle >/dev/null 2>&1; then
  for pair in \
    'khanhha11k lakshaymiddha/crack-segmentation-dataset' \
    'neu-det kaustubhdikshit/neu-surface-defect-database' \
    'dagm2007 mhskjelvareid/dagm-2007-competition-dataset-optical-inspection'; do
    name="${pair%% *}"; slug="${pair#* }"
    if has_content "$raw/$name"; then echo "SKIP $name"; else mkdir -p "$raw/$name"; echo "KAGGLE $slug"; kaggle datasets download -d "$slug" -p "$raw/$name" --unzip; fi
  done
else
  echo 'NOTE kaggle CLI not found: skipping Kaggle mirrors (khanhha11k, neu-det, dagm2007)'
fi

# --- Google Drive (needs gdown) ---
if command -v gdown >/dev/null 2>&1; then
  if has_content "$raw/pavement-bundle-fyangneil"; then echo 'SKIP pavement-bundle-fyangneil'; else
    mkdir -p "$raw/pavement-bundle-fyangneil"
    echo 'GDOWN fyangneil pavement bundle'
    gdown '13_vDYl54Mrd34dddX9w4ppAEiuWv4MlD' -O "$raw/pavement-bundle-fyangneil/pavement_bundle.zip"
  fi
else
  echo 'NOTE gdown not found: skipping the fyangneil pavement bundle'
fi

echo 'fetch-data: done. Gated sets (manual, once): MVTec AD/AD2 (form), GAPs (academic form), OmniCrack30k (email), Severstal (accept Kaggle rules), CrackVision12K (browser). See data/README.md.'
