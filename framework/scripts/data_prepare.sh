#!/usr/bin/env bash

# ==============================================================================
# HydraNet - Official Benchmarks Downloader
# (5 Minimal Raw/Depth Scenes, merged into one RGB+depth tree, pruned to an
#  exact 1:1 images<->depth pairing, and flattened to <split>/{images,depth};
#  Object Detection self-split 7:2:1 with images/labels renaming; Semantic
#  Segmentation now also self-split 7:2:1 with the unlabeled official
#  testing/ discarded, matching Object Detection.)
#
# Assumptions made below that are easy to get wrong reading this cold -
# flagged again in the chat reply, search for "ASSUMPTION" to find them
# in this file.
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
# ------------------------------------------------------------------
fetch_and_extract() {
    local label="$1" url="$2" dir="$3" zipname="$4"
    local zip="${dir}/${zipname}"
    local marker="${dir}/.done_${zipname}"

    mkdir -p "$dir"

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
# split_dataset BASE_DIR REF_MODALITY RATIOS KEEP_TESTING [RENAMES]
#
# Generic splitter for the official "training/<modality>/..." (+ optional
# "testing/<modality>/...") layout used by both kitti_object and
# kitti_semantics.
#
#   RATIOS       colon-separated weights, "8:2" (train:val) or "7:2:1"
#                (train:val:test). Applied ONLY to training/ - files are
#                assigned round-robin by sorted index (file 1 -> bucket 0,
#                file 2 -> bucket 1, ...), not "first/last N%", so the
#                held-out portion is spread across the id range instead of
#                clustered at one end.
#   KEEP_TESTING 1 = official testing/ (no public labels) is kept, just
#                renamed to test/.
#                0 = official testing/ is deleted outright - use this when
#                RATIOS already carves a labeled test/ out of training/,
#                which makes the unlabeled official testing/ dead weight.
#   RENAMES      optional "old1=new1,old2=new2" - modality subfolders are
#                written under the new name (e.g. "image_2=images").
#
# Modality subfolders under training/ are auto-discovered (not hardcoded),
# and for each reference file its counterpart in every other modality is
# matched by id with the extension wildcarded (image_2 is .png, label_2 is
# .txt - matching the literal filename would silently miss every label).
# Idempotent via a per-dataset marker file.
# ------------------------------------------------------------------
split_dataset() {
    local base_dir="$1" ref_modality="$2" ratios="$3" keep_testing="$4" renames="${5:-}"
    local marker="${base_dir}/.done_split"

    if [ -f "$marker" ]; then
        echo "   [SKIP] ${base_dir} already split."
        return 0
    fi
    if [ ! -d "${base_dir}/training/${ref_modality}" ]; then
        echo "❌ ${base_dir}/training/${ref_modality} not found - can't split."
        FAILED_TASKS+=("Split - ${base_dir} (missing ${ref_modality})")
        return 1
    fi

    local split_names=(train val test)
    local weights=()
    IFS=':' read -r -a weights <<< "$ratios"
    local n_splits=${#weights[@]}
    local total_weight=0 w
    for w in "${weights[@]}"; do total_weight=$((total_weight + w)); done

    local -A rename_of=()
    if [ -n "$renames" ]; then
        local pairs=() pair old new
        IFS=',' read -r -a pairs <<< "$renames"
        for pair in "${pairs[@]}"; do
            old="${pair%%=*}"; new="${pair#*=}"
            rename_of["$old"]="$new"
        done
    fi

    local modalities=() m
    for m in "${base_dir}/training"/*/; do
        [ -d "$m" ] || continue
        modalities+=("$(basename "$m")")
    done

    local -A dest_name=()
    for m in "${modalities[@]}"; do
        if [ -n "${rename_of[$m]:-}" ]; then
            dest_name["$m"]="${rename_of[$m]}"
        else
            dest_name["$m"]="$m"
        fi
    done

    local i s
    for (( i=0; i<n_splits; i++ )); do
        s="${split_names[$i]}"
        mkdir -p "${base_dir}/${s}"
        for m in "${modalities[@]}"; do
            mkdir -p "${base_dir}/${s}/${dest_name[$m]}"
        done
    done

    local counts=()
    for (( i=0; i<n_splits; i++ )); do counts+=(0); done

    local idx=0 f ref_name id_noext bucket cum which mod src
    for f in "${base_dir}/training/${ref_modality}"/*; do
        [ -f "$f" ] || continue
        ref_name="$(basename "$f")"
        id_noext="${ref_name%.*}"
        bucket=$(( idx % total_weight ))
        idx=$((idx + 1))

        cum=0
        which=$((n_splits - 1))
        for (( i=0; i<n_splits; i++ )); do
            cum=$((cum + weights[i]))
            if (( bucket < cum )); then which=$i; break; fi
        done
        s="${split_names[$which]}"
        counts[$which]=$(( counts[$which] + 1 ))

        for mod in "${modalities[@]}"; do
            for src in "${base_dir}/training/${mod}/${id_noext}".*; do
                [ -f "$src" ] || continue
                mv "$src" "${base_dir}/${s}/${dest_name[$mod]}/$(basename "$src")"
            done
        done
    done

    rm -rf "${base_dir}/training"

    if [ "$keep_testing" = "1" ] && [ -d "${base_dir}/testing" ]; then
        rm -rf "${base_dir}/test"
        mv "${base_dir}/testing" "${base_dir}/test"
    else
        rm -rf "${base_dir}/testing"
    fi

    touch "$marker"
    local summary=""
    for (( i=0; i<n_splits; i++ )); do
        summary="${summary}${split_names[$i]}=${counts[$i]} "
    done
    echo "   ✅ Split complete (ratio ${ratios}): ${summary}"
}

# ------------------------------------------------------------------
# Depth Data Cleanup, Flattening & Splitting Module
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
                        mv "$dir" "${DEPTH_DIR}/${target_folder}/${clean_name}"
                    else
                        rm -rf "$dir"
                    fi
                fi
            done
        fi
    done
    
    rm -rf "${DEPTH_DIR}/train" "${DEPTH_DIR}/val"
    mv "${DEPTH_DIR}/split_train" "${DEPTH_DIR}/train"
    mv "${DEPTH_DIR}/split_val" "${DEPTH_DIR}/val"
    mv "${DEPTH_DIR}/split_test" "${DEPTH_DIR}/test"
    
    touch "$depth_marker"
    echo "   ✅ Cleanup & splitting complete!"
}

# ------------------------------------------------------------------
# verify_depth_scenes
# cleanup_depth_data() silently drops any configured scene that never had
# official depth annotations to begin with - this just makes that visible.
# ------------------------------------------------------------------
verify_depth_scenes() {
    local all_scenes=("${TRAIN_SCENES[@]}" "${VAL_SCENES[@]}" "${TEST_SCENES[@]}")
    local s split found missing=0
    for s in "${all_scenes[@]}"; do
        found=0
        for split in train val test; do
            [ -d "${DEPTH_DIR}/${split}/${s}" ] && { found=1; break; }
        done
        if [ "$found" -eq 0 ]; then
            echo "   ⚠️  ${s}: no depth ground truth found after cleanup (may not be part of the official depth release)."
            missing=$((missing + 1))
        fi
    done
    [ "$missing" -eq 0 ] && echo "   ✅ All ${#all_scenes[@]} configured scenes present in depth train/val/test."
    return 0
}

# ------------------------------------------------------------------
# merge_raw_into_depth
# ASSUMPTION: only image_02 is kept from each side, and since depth's own
# flattened folder is already called "image_02" (that's a depth map, not a
# photo), keeping both under that name would collide. So this renames
# depth's image_02 -> depth/, drops depth's image_03 entirely, and the RGB
# image_02 migrated from kitti_raw becomes images/ instead. End state per
# scene: <split>/<scene>/{images/, depth/}. If you wanted different names
# here, this is the one line to change.
#
# A raw camera folder is itself image_02/{data/*.png, timestamps.txt} (KITTI's
# own raw layout), not flat PNGs - only the frames inside data/ are pulled up
# into images/, timestamps.txt is intentionally dropped, we don't need it.
#
# Only merges a raw scene if a matching depth scene folder already exists
# for it (same split, same scene name) - otherwise the raw scene is left
# untouched and reported, never silently deleted. kitti_raw/ is removed
# only once every scene under it has actually been consumed: each merged
# scene's own ".done_<scene>" marker is deleted the moment it's consumed
# (otherwise that leftover marker is exactly what keeps the final rmdir
# from ever succeeding, even after every real data folder is gone), and
# rmdir (not rm -rf) is used throughout, so anything genuinely left behind
# still survives.
# ------------------------------------------------------------------
merge_raw_into_depth() {
    local split scene raw_scene_dir depth_scene_dir
    local merged=0 skipped=0

    for split in train val test; do
        [ -d "${RAW_DIR}/${split}" ] || continue
        for raw_scene_dir in "${RAW_DIR}/${split}"/*/; do
            [ -d "$raw_scene_dir" ] || continue
            scene="$(basename "$raw_scene_dir")"
            depth_scene_dir="${DEPTH_DIR}/${split}/${scene}"

            if [ ! -d "$depth_scene_dir" ]; then
                echo "   ⚠️  ${scene}: no matching depth scene in ${split}/ yet - leaving raw data at ${raw_scene_dir} untouched."
                skipped=$((skipped + 1))
                continue
            fi
            if [ ! -d "${raw_scene_dir}image_02/data" ]; then
                echo "   ⚠️  ${scene}: no image_02/data under raw data - nothing to migrate."
                skipped=$((skipped + 1))
                continue
            fi

            rm -rf "${depth_scene_dir}/image_03"
            if [ -d "${depth_scene_dir}/image_02" ]; then
                rm -rf "${depth_scene_dir}/depth"
                mv "${depth_scene_dir}/image_02" "${depth_scene_dir}/depth"
            fi

            rm -rf "${depth_scene_dir}/images"
            mkdir -p "${depth_scene_dir}/images"
            mv "${raw_scene_dir}image_02/data/"* "${depth_scene_dir}/images/" 2>/dev/null || true

            rm -rf "$raw_scene_dir"
            rm -f "${RAW_DIR}/.done_${scene}"
            merged=$((merged + 1))
        done
        # Only removes a split folder that merging actually fully emptied;
        # a no-op (fails silently) if anything was skipped and left behind.
        rmdir "${RAW_DIR}/${split}" 2>/dev/null || true
    done
    rmdir "$RAW_DIR" 2>/dev/null || true

    echo "   ✅ Merged ${merged} scene(s) into ${DEPTH_DIR}; ${skipped} skipped (see warnings above, if any)."
}

# ------------------------------------------------------------------
# finalize_depth_dataset
# Runs after merge_raw_into_depth has produced <split>/<scene>/{images,depth}.
# Per scene:
#   0. Defensive cleanup for scenes coming from an older run of this script:
#      if images/ still has a nested data/ (+ timestamps.txt) instead of
#      flat frames, flatten it first.
#   1. Prune to an exact 1:1 pairing: depth ground truth doesn't cover every
#      raw frame, so images/ always starts out as a superset - any filename
#      present on only one side of images/ vs depth/ is deleted.
#   2. Dissolve the scene folder: paired files move up into a single
#      <split>/images/ and <split>/depth/, renamed "<scene>__<filename>" so
#      frames from different scenes sharing a split can never collide.
# Idempotent via a marker file; safe to re-run.
# ------------------------------------------------------------------
finalize_depth_dataset() {
    local marker="${DEPTH_DIR}/.done_depth_finalize"
    if [ -f "$marker" ]; then
        echo "   [SKIP] Depth data already paired and flattened."
        return 0
    fi

    local split scene_dir scene f base kept=0 dropped=0

    for split in train val test; do
        [ -d "${DEPTH_DIR}/${split}" ] || continue

        # Capture the scene list BEFORE creating the split-level images/depth/
        # destination folders below - otherwise those two folders would show
        # up in the glob on the very next loop and get mistaken for scenes.
        local scene_dirs=()
        for scene_dir in "${DEPTH_DIR}/${split}"/*/; do
            [ -d "$scene_dir" ] || continue
            scene_dirs+=("$scene_dir")
        done

        mkdir -p "${DEPTH_DIR}/${split}/images" "${DEPTH_DIR}/${split}/depth"

        for scene_dir in "${scene_dirs[@]}"; do
            scene="$(basename "$scene_dir")"

            if [ -d "${scene_dir}images/data" ]; then
                mv "${scene_dir}images/data/"* "${scene_dir}images/" 2>/dev/null || true
                rm -rf "${scene_dir}images/data" "${scene_dir}images/timestamps.txt"
            fi

            if [ ! -d "${scene_dir}images" ] || [ ! -d "${scene_dir}depth" ]; then
                echo "   ⚠️  ${scene}: missing images/ or depth/ - leaving as-is, not flattened."
                continue
            fi

            for f in "${scene_dir}images/"*; do
                [ -f "$f" ] || continue
                base="$(basename "$f")"
                if [ -f "${scene_dir}depth/${base}" ]; then
                    kept=$((kept + 1))
                else
                    rm -f "$f"
                    dropped=$((dropped + 1))
                fi
            done
            for f in "${scene_dir}depth/"*; do
                [ -f "$f" ] || continue
                base="$(basename "$f")"
                if [ ! -f "${scene_dir}images/${base}" ]; then
                    rm -f "$f"
                    dropped=$((dropped + 1))
                fi
            done

            for f in "${scene_dir}images/"*; do
                [ -f "$f" ] || continue
                mv "$f" "${DEPTH_DIR}/${split}/images/${scene}__$(basename "$f")"
            done
            for f in "${scene_dir}depth/"*; do
                [ -f "$f" ] || continue
                mv "$f" "${DEPTH_DIR}/${split}/depth/${scene}__$(basename "$f")"
            done

            rm -rf "$scene_dir"
        done
    done

    touch "$marker"
    echo "   ✅ Paired + flattened depth data: ${kept} matched frame(s) kept, ${dropped} unpaired file(s) dropped."
}

echo "==========================================================="
echo "🚀 Starting KITTI Benchmarks Download Pipeline"
echo "==========================================================="

OBJ_DIR="${BASE_DIR}/kitti_object"
SEG_DIR="${BASE_DIR}/kitti_semantics"
DEPTH_DIR="${BASE_DIR}/kitti_depth"
RAW_DIR="${BASE_DIR}/kitti_raw"

echo "-> [1/6] Downloading Object Detection Images (12GB)..."
fetch_and_extract "Object Detection Images" "$URL_OBJ_IMG" "$OBJ_DIR" "data_object_image_2.zip" || true

echo "-> [2/6] Downloading Object Detection Labels (5MB)..."
fetch_and_extract "Object Detection Labels" "$URL_OBJ_LBL" "$OBJ_DIR" "data_object_label_2.zip" || true

echo "   Splitting kitti_object 7:2:1 (train:val:test) from training/ only -"
echo "   official testing/ has no public labels, so it's discarded rather than kept."
split_dataset "$OBJ_DIR" "image_2" "7:2:1" "0" "image_2=images,label_2=labels" || true

echo "-> [3/6] Downloading Semantic Segmentation (298MB)..."
fetch_and_extract "Semantic Segmentation" "$URL_SEG" "$SEG_DIR" "data_semantics.zip" || true

echo "   Splitting kitti_semantics 7:2:1 (train:val:test) from training/ only -"
echo "   official testing/ has no public labels, so it's discarded rather than kept."
split_dataset "$SEG_DIR" "image_2" "7:2:1" "0" "image_2=images" || true

echo "-> [4/6] Downloading Depth Estimation Ground Truth (15GB)..."
if fetch_and_extract "Depth Estimation" "$URL_DEPTH" "$DEPTH_DIR" "data_depth_annotated.zip"; then
    cleanup_depth_data
    verify_depth_scenes
else
    FAILED_TASKS+=("Depth Estimation Download/Extraction")
fi

echo "-> [5/6] Downloading and Restructuring Raw Data (Applying Train/Val/Test Split)..."

mkdir -p "${RAW_DIR}/train" "${RAW_DIR}/val" "${RAW_DIR}/test"

if [ "${#RAW_CATEGORIES[@]}" -ne "${#RAW_FIRST_DRIVE[@]}" ]; then
    echo "❌ RAW_CATEGORIES and RAW_FIRST_DRIVE are out of sync."
    FAILED_TASKS+=("Raw Data (config error)")
else
    for i in "${!RAW_CATEGORIES[@]}"; do
        cat_name="${RAW_CATEGORIES[$i]}"
        drive="${RAW_FIRST_DRIVE[$i]}"

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

        if [ -z "$split_folder" ]; then
            echo "   -> [SKIP] ${drive} is not assigned to train/val/test splits."
            continue
        fi

        date_str="${drive:0:10}"
        target_dir="${RAW_DIR}/${split_folder}/${drive}"
        marker="${RAW_DIR}/.done_${drive}"
        tmp_dir="${RAW_DIR}/tmp_${drive}"

        echo "   -> [$(( i + 1 ))/${#RAW_CATEGORIES[@]}] Extracting ${drive} directly to kitti_raw/${split_folder}/ ..."

        if [ -f "$marker" ] || [ -d "$target_dir" ]; then
            echo "      [SKIP] ${drive} already structured in ${split_folder}/."
            continue
        fi

        zip_url="${RAW_BASE_URL}/${drive}/${drive}_sync.zip"

        # Routed through fetch_and_extract (into a scratch tmp_dir) instead of
        # a bare download_with_retry + unzip, so this gets the same
        # unzip -tq integrity check and retry/resume as every other download.
        if fetch_and_extract "Raw Data - ${cat_name} (${drive})" "$zip_url" "$tmp_dir" "${drive}_sync.zip"; then
            src="${tmp_dir}/${date_str}/${drive}_sync"
            if [ -d "$src" ]; then
                mv "$src" "$target_dir"
                rm -rf "$tmp_dir"
                touch "$marker"
                echo "      ✅ Restructured to ${target_dir}"
            else
                # Deliberately NOT touching $marker and NOT deleting tmp_dir:
                # a layout mismatch becomes a loud, retryable failure instead
                # of a silent "success" that's actually empty.
                echo "❌ ${drive}: expected ${src} after unzip but it's not there - inspect ${tmp_dir} manually."
                FAILED_TASKS+=("Raw Data - ${drive} (unexpected zip layout)")
            fi
        fi
    done
fi

echo "-> [6/6] Merging Raw RGB (image_02) into the matching Depth scenes..."
merge_raw_into_depth

echo "   Pairing images<->depth 1:1 and flattening scene folders..."
finalize_depth_dataset

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