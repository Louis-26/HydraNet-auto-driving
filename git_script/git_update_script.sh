#!/usr/bin/env bash
# git_update_script.sh
#
# Replace a local folder (default: git_script) with the same-named folder
# pulled fresh from a remote git repo's specific branch. Intended to sync
# shared tooling (like git_script/) across many repos from one source of truth.
#
# Any flag you don't provide silently falls back to the default — there are
# no interactive prompts. This makes the script safe to call from other
# scripts (like git_search.sh) and keeps bare-no-args invocation fast.
#
# Usage:
#   git_update_script.sh                              # all defaults
#   git_update_script.sh --target-repo URL --folder-name NAME ...   # override any subset
#
# Flags:
#   --target-repo / --target_repo   URL       Remote repo URL, no trailing .git required
#   --folder-name / --folder_name   NAME      Folder inside the repo to grab
#   --branch-name / --branch_name   NAME      Branch to pull from
#   --save-folder / --save_folder   PATH      Scratch dir to stage the download
#   -h / --help                               Show this message

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# -------- defaults --------

DEFAULT_REPO_URL="https://github.com/Louis-26/git_script_template"
DEFAULT_FOLDER_NAME="git_script"
DEFAULT_BRANCH_NAME="main"
DEFAULT_SAVE_DIR="${HOME//\\//}/Downloads"

# -------- cli parsing --------

REPO_URL="$DEFAULT_REPO_URL"
FOLDER_NAME="$DEFAULT_FOLDER_NAME"
BRANCH_NAME="$DEFAULT_BRANCH_NAME"
SAVE_DIR="$DEFAULT_SAVE_DIR"

print_help() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

need_value() {
    local flag="$1"
    local value="${2:-}"

    if [ -z "$value" ] || [[ "$value" == --* ]]; then
        echo "ERROR: $flag requires a value." >&2
        exit 2
    fi
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --target-repo|--target_repo)
            need_value "$1" "${2:-}"
            REPO_URL="$2"
            shift 2
            ;;

        --folder-name|--folder_name)
            need_value "$1" "${2:-}"
            FOLDER_NAME="$2"
            shift 2
            ;;

        --branch-name|--branch_name)
            need_value "$1" "${2:-}"
            BRANCH_NAME="$2"
            shift 2
            ;;

        --save-folder|--save_folder)
            need_value "$1" "${2:-}"
            SAVE_DIR="$2"
            shift 2
            ;;

        -h|--help)
            print_help
            ;;

        *)
            echo "Unknown argument: $1" >&2
            echo "Run with --help for usage." >&2
            exit 2
            ;;
    esac
done

# Normalize Windows-style backslashes to forward slashes
SAVE_DIR="${SAVE_DIR//\\//}"

# -------- sanity checks --------

if [ ! -d "$SAVE_DIR" ]; then
    echo "Creating scratch directory: $SAVE_DIR"
    mkdir -p "$SAVE_DIR" || {
        echo "Failed to create $SAVE_DIR" >&2
        exit 1
    }
fi

ORIGINAL_DIR="$(pwd)"

echo
echo "Update plan:"
echo "  From:   $REPO_URL  (branch: $BRANCH_NAME)"
echo "  Folder: $FOLDER_NAME"
echo "  Stage:  $SAVE_DIR"
echo "  Target: $ORIGINAL_DIR/$FOLDER_NAME"
echo

# -------- 1. sparse-checkout the folder into a temp dir --------

TMP_DIR="$(mktemp -d "$SAVE_DIR/update_script_XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

cd "$TMP_DIR"

git init -q
git remote add origin "${REPO_URL%.git}.git"
git sparse-checkout init --no-cone
git sparse-checkout set "$FOLDER_NAME"

echo "Fetching $FOLDER_NAME from $BRANCH_NAME ..."

if ! git pull --quiet "${REPO_URL%.git}.git" "$BRANCH_NAME"; then
    echo "Pull failed. Check the URL/branch and try again." >&2
    exit 1
fi

if [ ! -d "$FOLDER_NAME" ]; then
    echo "The folder '$FOLDER_NAME' was not found at the top level of $BRANCH_NAME." >&2
    exit 1
fi

# -------- 2. replace the local folder with the downloaded one --------

cd "$ORIGINAL_DIR"

rm -rf -- "$ORIGINAL_DIR/$FOLDER_NAME"
mv -- "$TMP_DIR/$FOLDER_NAME" "$ORIGINAL_DIR/"

echo
echo "Done! '$FOLDER_NAME' was updated in $ORIGINAL_DIR/$FOLDER_NAME"
