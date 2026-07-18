#!/usr/bin/env bash
set -euo pipefail

# Define target directory preserving original conventions
CHKPT_DIR="$(git rev-parse --show-toplevel)/cv-multitask-learning-project/checkpoints"
WEIGHT_NAME="ExpKITTI_joint.ckpt"
FILE_PATH="${CHKPT_DIR}/${WEIGHT_NAME}"

# Placeholder URL: In an actual deployment, this would be your AWS S3 bucket 
# or a direct link containing the Nekrasov et al. weights.
WEIGHT_URL="https://s3.eu-central-1.amazonaws.com/hydranet-project-bucket/${WEIGHT_NAME}"

mkdir -p "$CHKPT_DIR"

echo "=> Initializing pretrained MobileNetV2 & RefineNet weights..."

# 1. Idempotency Check: Skip download if weights already exist
if [ -f "$FILE_PATH" ]; then
    echo "=> [INFO] Weights already exist at ${FILE_PATH}. Skipping download."
else
    echo "=> [START] Fetching ${WEIGHT_NAME} via curl..."
    
    # 2. Resumable Download using curl
    if curl -sLC - -o "$FILE_PATH" "$WEIGHT_URL"; then
        echo "=> [DONE] Successfully downloaded to ${FILE_PATH}"
    else
        echo "=> [ERROR] Failed to download weights. Check network or URL." >&2
        exit 1
    fi
fi

# 3. Integrity Verification (Optional but highly recommended for HPC workflows)
# EXPECTED_MD5="e4d909c290d0fb1ca068ffaddf22cbd0" 
# ACTUAL_MD5=$(md5sum "$FILE_PATH" | awk '{print $1}')
# if [ "$ACTUAL_MD5" == "$EXPECTED_MD5" ]; then
#     echo "=> [INFO] MD5 checksum verified."
# else
#     echo "=> [WARNING] MD5 mismatch! The .ckpt file might be corrupted." >&2
# fi

echo "Weight initialization sequence completed."