#!/usr/bin/env bash

# ==============================================================================
# HydraNet - Official Benchmarks Downloader 
# (Modified for 5 Minimal Scenes, Flat Structure, and Custom Train/Val/Test Split)
# ==============================================================================

set -uo pipefail

# ==============================================================================
# DOWNLOAD LINKS 
# ==============================================================================
URL_OBJ_IMG="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
URL_OBJ_LBL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
URL_SEG="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_semantics.zip"
URL_DEPTH="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip"

RAW_BASE_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"

RAW_CATEGORIES=(city residential road campus person)
RAW_FIRST_DRIVE=(
    "2011_09_26_drive_0001"   # City
    "2011_09_26_drive_0019"   # Residential
    "2011_09_26_drive_0015"   # Road
    "2011_09_28_drive_0016"   # Campus
    "2011_09_28_drive_0053"   # Person
)

# Custom Splits Configuration for Raw and Depth Data
TRAIN_SCENES=("2011_09_26_drive_0001" "2011_09_26_drive_0019" "2011_09_26_drive_0015")
VAL_SCENES=("2011_09_28_drive_0016")
TEST_SCENES=("2011_09_28_drive_0053")

BASE_DIR="./data"
mkdir -p "$BASE_DIR"

MAX_RETRIES=8
RETRY_DELAY=15
FAILED_TASKS=()

# ------------------------------------------------------------------
# Download with Retry Module
# Continuously retries the download if the connection drops.
# Uses 'curl -C -' to resume from the last downloaded byte.
# ------------------------------------------------------------------
download_with_retry() {
    local url="$1" out="$2" attempt=1
    while (( attempt <= MAX_RETRIES )); do
        if (( attempt > 1 )); then
            echo "   retry ${attempt}/${MAX_RETRIES} in ${RETRY_DELAY}s (resuming, not restarting)..."
            sleep "$RETRY_DELAY"
        fi
        if curl -L -C - --fail --connect-timeout 30 -o "$out" "$url"; then
            return 0
        fi
        attempt=$((attempt + 1))
    done
    return 1
}

# ------------------------------------------------------------------
# Fetch and Extract Module
# Downloads a specific zip file, verifies its integrity, 
# and unzips it into the specified directory.
# ------------------------------------------------------------------
fetch_and_extract() {
    local label="$1" url="$2" dir="$3" zipname="$4"
    local zip="${dir}/${zipname}"
    local marker="${dir}/.done_${zipname}"

    mkdir -p "$dir"

    # Skip if the completion marker exists
    if [ -f "$marker" ]; then
        echo "   [SKIP] ${label} already downloaded."
        return 0
    fi

    if download_with_retry "$url" "$zip"; then
        echo "   Verifying archive integrity..."
        if unzip -tq "$zip" >/dev/null 2>&1; then
            echo "   Unzipping ${label}..."
            unzip -q -o "$zip" -d "$dir"
            rm -f "$zip"
            touch "$marker"
            return 0
        fi
        echo "   archive failed integrity check (possibly corrupted)."
    fi

    echo "❌ Failed to download/verify ${label}."
    FAILED_TASKS+=("$label")
    return 1
}

# ------------------------------------------------------------------
# Split Object & Semantics Datasets (80% Train / 20% Val)
# This function handles the physical separation of the default KITTI
# 'training' folder into 'train' and 'val' to match the dataloader.
# ------------------------------------------------------------------
split_object_and_semantics() {
    echo " -> ✂️  Splitting Object & Semantics datasets into Train and Val (80/20 split)..."
    
    for dataset in "kitti_object" "kitti_semantics"; do
        local ds_dir="${BASE_DIR}/${dataset}"
        local marker="${ds_dir}/.done_split"
        
        # If the 'training' folder is missing or the split is already done, skip to the next dataset
        if [ ! -d "${ds_dir}/training" ] || [ -f "$marker" ]; then
            continue
        fi

        # Step 1: Rename the official 'training' directory to 'train'
        # At this point, 'train' contains 100% of the labeled data.
        mv "${ds_dir}/training" "${ds_dir}/train"
        
        # Step 2: Create the corresponding 'val' directory structure
        # This mirrors all subfolders (e.g., image_2, label_2, semantic_rgb) from 'train' to 'val'
        mkdir -p "${ds_dir}/val"
        for subfolder in "${ds_dir}/train"/*; do
            if [ -d "$subfolder" ]; then
                local sub_name=$(basename "$subfolder")
                mkdir -p "${ds_dir}/val/${sub_name}"
            fi
        done

        # Step 3: Determine the reference folder to count the total number of files
        # kitti_object uses 'image_2', kitti_semantics uses 'images'
        if [ -d "${ds_dir}/train/image_2" ]; then
            local ref_folder="image_2"
        elif [ -d "${ds_dir}/train/images" ]; then
            local ref_folder="images"
        else
            continue
        fi

        # Count total files in the reference folder and calculate 20% for the validation set
        local files=($(ls -1 "${ds_dir}/train/${ref_folder}" | grep -E '\.png|\.jpg'))
        local total=${#files[@]}
        local val_count=$((total * 20 / 100))
        
        echo "      Processing ${dataset}: Moving ${val_count} of ${total} files to val..."

        # Step 4: Move the exact 20% of files from 'train' to 'val'
        # By doing this, 'train' is left with 80%, and 'val' acquires the remaining 20%
        for (( i=${total}-${val_count}; i<${total}; i++ )); do
            local filename="${files[$i]}"
            local base_name="${filename%.*}"
            
            # Loop through all subfolders (images, labels, etc.) and move the matching file ID
            for subfolder in "${ds_dir}/train"/*; do
                if [ -d "$subfolder" ]; then
                    local sub_name=$(basename "$subfolder")
                    # Check and move .png (images/masks) or .txt (labels) if they exist
                    if [ -f "${subfolder}/${base_name}.png" ]; then
                        mv "${subfolder}/${base_name}.png" "${ds_dir}/val/${sub_name}/"
                    elif [ -f "${subfolder}/${base_name}.txt" ]; then
                        mv "${subfolder}/${base_name}.txt" "${ds_dir}/val/${sub_name}/"
                    fi
                fi
            done
        done
        
        # Step 5: Rename the official 'testing' directory (which has no labels) to 'test'
        if [ -d "${ds_dir}/testing" ]; then
            mv "${ds_dir}/testing" "${ds_dir}/test"
        fi
        
        # Place a marker to prevent re-splitting on future runs
        touch "$marker"
        echo "      ✅ ${dataset} split completed."
    done
}

# ------------------------------------------------------------------
# Depth Data Cleanup, Flattening & Splitting Module
# Removes unnecessary nested folders and applies the custom Train/Val/Test split.
# ------------------------------------------------------------------
cleanup_depth_data() {
    echo " -> 🧹 Optimizing Depth GT: Flattening, removing '_sync', and applying Train/Val/Test split..."
    local depth_marker="${DEPTH_DIR}/.done_depth_cleanup"

    if [ -f "$depth_marker" ]; then
        echo "   [SKIP] Depth data already optimized and split."
        return 0
    fi

    # Create target split directories temporarily with different names to avoid conflicts 
    mkdir -p "${DEPTH_DIR}/split_train" "${DEPTH_DIR}/split_val" "${DEPTH_DIR}/split_test"

    for subset in "train" "val"; do
        if [ -d "${DEPTH_DIR}/${subset}" ]; then
            for dir in "${DEPTH_DIR}/${subset}"/*; do
                if [ -d "$dir" ]; then
                    local dirname=$(basename "$dir")
                    local clean_name="${dirname%_sync}" # Strip the '_sync' suffix
                    local target_folder=""

                    # Match the scene name against our predefined arrays to determine its destination
                    for s in "${TRAIN_SCENES[@]}"; do
                        if [ "$clean_name" == "$s" ]; then target_folder="split_train"; break; fi
                    done
                    for s in "${VAL_SCENES[@]}"; do
                        if [ "$clean_name" == "$s" ]; then target_folder="split_val"; break; fi
                    done
                    for s in "${TEST_SCENES[@]}"; do
                        if [ "$clean_name" == "$s" ]; then target_folder="split_test"; break; fi
                    done

                    if [ -n "$target_folder" ]; then
                        # Flatten the directory structure by moving images up from 'proj_depth/groundtruth/'
                        if [ -d "$dir/proj_depth/groundtruth" ]; then
                            mv "$dir/proj_depth/groundtruth/"* "$dir/" 2>/dev/null || true
                            rm -rf "$dir/proj_depth"
                        fi
                        # Move the flattened folder to its assigned train/val/test directory
                        mv "$dir" "${DEPTH_DIR}/${target_folder}/${clean_name}"
                    else
                        # Discard any scenes that are not part of our 5 minimal scenes
                        rm -rf "$dir"
                    fi
                fi
            done
        fi
    done
    
    # Replace the original KITTI folders with our strictly organized split folders
    rm -rf "${DEPTH_DIR}/train" "${DEPTH_DIR}/val"
    mv "${DEPTH_DIR}/split_train" "${DEPTH_DIR}/train"
    mv "${DEPTH_DIR}/split_val" "${DEPTH_DIR}/val"
    mv "${DEPTH_DIR}/split_test" "${DEPTH_DIR}/test"
    
    touch "$depth_marker"
    echo "   ✅ Cleanup & splitting complete! Check data/kitti_depth/ for train/val/test folders."
}

# ==============================================================================
# PIPELINE START
# ==============================================================================
echo "==========================================================="
echo "🚀 Starting KITTI Benchmarks Download Pipeline"
echo "==========================================================="

OBJ_DIR="${BASE_DIR}/kitti_object"
SEG_DIR="${BASE_DIR}/kitti_semantics"
DEPTH_DIR="${BASE_DIR}/kitti_depth"
RAW_DIR="${BASE_DIR}/kitti_raw"

echo "-> [1/5] Downloading Object Detection Images (12GB)..."
fetch_and_extract "Object Detection Images" "$URL_OBJ_IMG" "$OBJ_DIR" "data_object_image_2.zip" || true

echo "-> [2/5] Downloading Object Detection Labels (5MB)..."
fetch_and_extract "Object Detection Labels" "$URL_OBJ_LBL" "$OBJ_DIR" "data_object_label_2.zip" || true

echo "-> [3/5] Downloading Semantic Segmentation (298MB)..."
fetch_and_extract "Semantic Segmentation" "$URL_SEG" "$SEG_DIR" "data_semantics.zip" || true

# Trigger the physical 80/20 split for Object and Segmentation immediately after downloading
split_object_and_semantics

echo "-> [4/5] Downloading Depth Estimation Ground Truth (15GB)..."
if fetch_and_extract "Depth Estimation" "$URL_DEPTH" "$DEPTH_DIR" "data_depth_annotated.zip"; then
    # Clean up and split the massive 15GB depth dataset into our 5 minimal scenes
    cleanup_depth_data
else
    FAILED_TASKS+=("Depth Estimation Download/Extraction")
fi

echo "-> [5/5] Downloading and Restructuring Raw Data (Applying Train/Val/Test Split)..."

# Ensure the root RAW_DIR and the three split folders are created
mkdir -p "${RAW_DIR}/train" "${RAW_DIR}/val" "${RAW_DIR}/test"

# Verify that our category array matches the chosen sequence array
if [ "${#RAW_CATEGORIES[@]}" -ne "${#RAW_FIRST_DRIVE[@]}" ]; then
    echo "❌ RAW_CATEGORIES and RAW_FIRST_DRIVE are out of sync."
    FAILED_TASKS+=("Raw Data (config error)")
else
    for i in "${!RAW_CATEGORIES[@]}"; do
        cat_name="${RAW_CATEGORIES[$i]}"
        drive="${RAW_FIRST_DRIVE[$i]}"
        
        # Determine which split folder this specific drive should be placed in
        split_folder=""
        for s in "${TRAIN_SCENES[@]}"; do
            if [ "$drive" == "$s" ]; then split_folder="train"; break; fi
        done
        if [ -z "$split_folder" ]; then
            for s in "${VAL_SCENES[@]}"; do
                if [ "$drive" == "$s" ]; then split_folder="val"; break; fi
            done
        fi
        if [ -z "$split_folder" ]; then
            for s in "${TEST_SCENES[@]}"; do
                if [ "$drive" == "$s" ]; then split_folder="test"; break; fi
            done
        fi

        # Skip if the drive was not defined in any of the split arrays
        if [ -z "$split_folder" ]; then
            echo "   -> [SKIP] ${drive} is not assigned to train/val/test splits."
            continue
        fi
        
        # Extract the date string (e.g., "2011_09_26" from "2011_09_26_drive_0001") for internal extraction routing
        date_str="${drive:0:10}"
        
        target_dir="${RAW_DIR}/${split_folder}/${drive}"
        marker="${RAW_DIR}/.done_${drive}"
        
        echo "   -> [$(( i + 1 ))/${#RAW_CATEGORIES[@]}] Extracting ${drive} directly to kitti_raw/${split_folder}/ ..."

        # Skip extraction if this specific scene has already been fully processed
        if [ -f "$marker" ] || [ -d "$target_dir" ]; then
            echo "      [SKIP] ${drive} already structured in ${split_folder}/."
            continue
        fi

        zip_url="${RAW_BASE_URL}/${drive}/${drive}_sync.zip"
        zip_file="${RAW_DIR}/${drive}_sync.zip"
        tmp_dir="${RAW_DIR}/tmp_${drive}"

        if download_with_retry "$zip_url" "$zip_file"; then
            echo "      Unzipping..."
            unzip -q -o "$zip_file" -d "$tmp_dir"
            
            # The internal KITTI structure is nested: tmp_dir/2011_09_26/2011_09_26_drive_0001_sync
            # Move the innermost folder out to the correct split directory and rename it (removing '_sync')
            mv "${tmp_dir}/${date_str}/${drive}_sync" "$target_dir"
            
            # Clean up the temporary extraction folders and the downloaded zip file
            rm -rf "$tmp_dir"
            rm -f "$zip_file"
            touch "$marker"
            echo "      ✅ Restructured to ${target_dir}"
        else
            echo "❌ Failed to download/verify Raw Data: ${drive}"
            FAILED_TASKS+=("Raw Data - ${drive}")
        fi
    done
fi

echo "==========================================================="
if [ ${#FAILED_TASKS[@]} -eq 0 ]; then
    echo "✅ All downloads, extractions, and restructurings completed!"
    echo "Your benchmarks are now located in ${BASE_DIR}/"
else
    echo "⚠️  Finished with ${#FAILED_TASKS[@]} failure(s):"
    for t in "${FAILED_TASKS[@]}"; do
        echo "   - $t"
    done
    echo "Re-run this script to retry."
fi
echo "==========================================================="

if [ ${#FAILED_TASKS[@]} -ne 0 ]; then
    exit 1
fi