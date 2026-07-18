# Fetch the open datasets Fisura uses into the local data vault (idempotent; nothing enters git).
# Root: $env:FISURA_DATA_ROOT, else .env FISURA_DATA_ROOT, else <repo>/data/raw-local (gitignored).
# Sources, licenses and redistribution rulings: data/README.md (the registry). URLs verified 2026-07-18.
# Gated sets (MVTec AD/AD2, GAPs, OmniCrack30k, Severstal, CrackVision12K) need a one-time manual step,
# documented in the registry; this script covers every direct source.
# Optional tools: kaggle CLI (Kaggle mirrors; KAGGLE_API_TOKEN or kaggle.json) and gdown (Google Drive).

$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'

$repo = Split-Path $PSScriptRoot -Parent
$root = $env:FISURA_DATA_ROOT
if (-not $root -and (Test-Path "$repo\.env")) {
    $line = Select-String -Path "$repo\.env" -Pattern '^FISURA_DATA_ROOT=(.+)$' | Select-Object -First 1
    if ($line) { $root = $line.Matches[0].Groups[1].Value.Trim() }
}
if (-not $root) { $root = Join-Path $repo 'data\raw-local' }
$raw = Join-Path $root 'raw'
New-Item -ItemType Directory -Force $raw | Out-Null
Write-Host "fetch-data: vault root = $root"

function Has-Content($dir) { (Test-Path $dir) -and ((Get-ChildItem $dir -Force -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0) }
function Get-Archive {
    param($Url, $Dir, $Name)
    New-Item -ItemType Directory -Force $Dir | Out-Null
    $dest = Join-Path $Dir $Name
    if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 0)) { Write-Host "SKIP $Name"; return $dest }
    Write-Host "GET  $Name"
    curl.exe -sS -L --fail --retry 3 --retry-delay 10 -C - -o $dest $Url
    if ($LASTEXITCODE -eq 0) { return $dest }
    Write-Host "FAIL $Name (curl exit $LASTEXITCODE)"; return $null
}

# --- Direct downloads (segmentation / classification / multi-class / anomaly) ---
$d = "$raw\bcl"           # Bridge Crack Library, CC0 (Ye et al. 2021)
if (-not (Has-Content $d)) { Get-Archive 'https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/RURXSH' $d 'bcl_dataverse.zip' | Out-Null }
$d = "$raw\crackseg9k"    # CrackSeg9k aggregation (CC0 label; component licenses apply)
if (-not (Has-Content $d)) { Get-Archive 'https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/EGIEBY' $d 'crackseg9k_dataverse.zip' | Out-Null }
$d = "$raw\deepcrack537"  # DeepCrack-537 (cite-only)
if (-not (Has-Content $d)) { Get-Archive 'https://github.com/yhlleo/DeepCrack/raw/master/dataset/DeepCrack.zip' $d 'DeepCrack.zip' | Out-Null }
$d = "$raw\visa"          # VisA (CC BY 4.0)
if (-not (Has-Content $d)) { Get-Archive 'https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar' $d 'VisA_20220922.tar' | Out-Null }
$d = "$raw\kolektorsdd"   # KolektorSDD (CC BY-NC-SA: local only)
if (-not (Has-Content $d)) { Get-Archive 'https://go.vicos.si/kolektorsdd' $d 'KolektorSDD.zip' | Out-Null }
$d = "$raw\kolektorsdd2"  # KolektorSDD2 (CC BY-NC-SA: local only)
if (-not (Has-Content $d)) { Get-Archive 'https://go.vicos.si/kolektorsdd2' $d 'KolektorSDD2.zip' | Out-Null }
$d = "$raw\dacl10k"       # dacl10k (CC BY-NC 4.0: local only)
if (-not (Has-Content $d)) {
    Get-Archive 'https://dacl10k.s3.eu-central-1.amazonaws.com/dacl10k-challenge/dacl10k_v2_devphase.zip' $d 'dacl10k_v2_devphase.zip' | Out-Null
    Get-Archive 'https://dacl10k.s3.eu-central-1.amazonaws.com/dacl10k-challenge/dacl10k_v2_testchallenge.zip' $d 'dacl10k_v2_testchallenge.zip' | Out-Null
}
$d = "$raw\codebrim"      # CODEBRIM (non-commercial: local only)
if (-not (Has-Content $d)) {
    Get-Archive 'https://zenodo.org/records/2620293/files/CODEBRIM_classification_balanced_dataset.zip?download=1' $d 'CODEBRIM_classification_balanced_dataset.zip' | Out-Null
    Get-Archive 'https://zenodo.org/records/2620293/files/CODEBRIM_original_images.zip?download=1' $d 'CODEBRIM_original_images.zip' | Out-Null
}
$d = "$raw\rdd2022"       # RDD2022 (CC BY-SA 4.0)
if (-not (Has-Content $d)) { Get-Archive 'https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/RDD2022.zip' $d 'RDD2022.zip' | Out-Null }

# SDNET2018 (CC BY 4.0): resolve the zip link from the Digital Commons landing page
$d = "$raw\sdnet2018"
if (-not (Has-Content $d)) {
    try {
        $landing = Invoke-WebRequest -Uri 'https://digitalcommons.usu.edu/all_datasets/48/' -UseBasicParsing
        $href = ($landing.Links | Where-Object { $_.href -match 'viewcontent|\.zip' } | Select-Object -First 1).href
        if ($href) {
            if ($href -notmatch '^https?://') { $href = "https://digitalcommons.usu.edu$href" }
            Get-Archive $href $d 'SDNET2018.zip' | Out-Null
        } else { Write-Host 'FAIL sdnet2018: no download link on landing page' }
    } catch { Write-Host "FAIL sdnet2018: $($_.Exception.Message)" }
}

# Ozgenel METU (CC BY 4.0): Mendeley public API signed URL (expires; fetched per run)
$d = "$raw\ozgenel-metu"
if (-not (Has-Content $d)) {
    try {
        $files = Invoke-RestMethod -Uri 'https://data.mendeley.com/public-api/datasets/5y9wdsg2zt/files?folder_id=root&version=2' -UseBasicParsing
        foreach ($f in $files) { Get-Archive $f.content_details.download_url $d $f.filename | Out-Null }
    } catch { Write-Host "FAIL ozgenel: $($_.Exception.Message)" }
}

# --- Git-clone datasets ---
foreach ($c in @(
    @{ name = 'crackforest-cfd'; url = 'https://github.com/cuilimeng/CrackForest-dataset.git' },   # non-commercial
    @{ name = 'uav75';           url = 'https://github.com/ben-z-original/uav75.git' },            # GPL-3.0: local only
    @{ name = 'magnetic-tile';   url = 'https://github.com/abin24/Magnetic-tile-defect-datasets..git' }, # cite-only
    @{ name = 'masonry-dais';    url = 'https://github.com/dimitrisdais/crack_detection_CNN_masonry.git' } # GPL-3.0 repo
)) {
    $d = "$raw\$($c.name)"
    if (-not (Has-Content $d)) { Write-Host "CLONE $($c.name)"; git clone --depth 1 $c.url $d }
    else { Write-Host "SKIP $($c.name)" }
}

# --- Kaggle mirrors (need kaggle CLI + auth; skipped cleanly otherwise) ---
$kaggle = Get-Command kaggle -ErrorAction SilentlyContinue
if ($kaggle) {
    foreach ($k in @(
        @{ name = 'khanhha11k'; slug = 'lakshaymiddha/crack-segmentation-dataset' },  # aggregation, cite-only components
        @{ name = 'neu-det';    slug = 'kaustubhdikshit/neu-surface-defect-database' }, # cite-only
        @{ name = 'dagm2007';   slug = 'mhskjelvareid/dagm-2007-competition-dataset-optical-inspection' } # CC BY 4.0
    )) {
        $d = "$raw\$($k.name)"
        if (-not (Has-Content $d)) { New-Item -ItemType Directory -Force $d | Out-Null; Write-Host "KAGGLE $($k.slug)"; kaggle datasets download -d $k.slug -p $d --unzip }
        else { Write-Host "SKIP $($k.name)" }
    }
} else { Write-Host 'NOTE kaggle CLI not found: skipping Kaggle mirrors (khanhha11k, neu-det, dagm2007)' }

# --- Google Drive (needs gdown; skipped cleanly otherwise) ---
$gdown = Get-Command gdown -ErrorAction SilentlyContinue
if ($gdown) {
    $d = "$raw\pavement-bundle-fyangneil"  # Crack500 + GAPs384 + CFD + AEL + CrackTree200 (cite-only mix: local only)
    if (-not (Has-Content $d)) {
        New-Item -ItemType Directory -Force $d | Out-Null
        Write-Host 'GDOWN fyangneil pavement bundle'
        gdown '13_vDYl54Mrd34dddX9w4ppAEiuWv4MlD' -O "$d\pavement_bundle.zip"
    } else { Write-Host 'SKIP pavement-bundle-fyangneil' }
} else { Write-Host 'NOTE gdown not found: skipping the fyangneil pavement bundle' }

Write-Host 'fetch-data: done. Gated sets (manual, once): MVTec AD/AD2 (form), GAPs (academic form), OmniCrack30k (email), Severstal (accept Kaggle rules), CrackVision12K (browser). See data/README.md.'
