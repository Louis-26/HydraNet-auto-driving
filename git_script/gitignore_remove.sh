# in this file, it will iteratively prompt the user to enter the file from .gitignore to remove, meaning it will
# be tracked by git again, if that file doesn't exist, throw a warning and prompt the user to enter again
# meanwhile, if it the folder/file is directly there in .gitignore, it will be removed from .gitignore directly
# if that folder/file is included in some folder in .gitignore, decompose the folder into the same level of the folder/file, and then remove the file/folder

#!/usr/bin/env bash

# Interactively un-ignore files or folders from .gitignore.
#
# In a loop:
#   - prompt user for a path to un-ignore
#   - if the path doesn't exist on disk, warn and prompt again
#   - if the path is listed explicitly in .gitignore, remove that line
#   - if the path is covered by an ancestor "/folder/" entry, expand that
#     ancestor ONE LEVEL at a time (listing its direct children as entries),
#     repeating until the target itself is an explicit entry that can be removed.
#     Sibling folders stay as single "/folder/" entries — not flattened to files.
#
# Only edits .gitignore. Does not run any git commands.
#
# Exit keywords: empty input, q, quit, exit.

cd "$(git rev-parse --show-toplevel)"
[ -f .gitignore ] || { echo "No .gitignore found."; exit 0; }

# --- helpers ---

is_simple_folder_entry() {
    local line="$1"
    [[ -z "$line" ]] && return 1
    [[ "$line" == \#* ]] && return 1
    [[ "$line" == !* ]] && return 1
    [[ "$line" != /* ]] && return 1
    [[ "$line" != */ ]] && return 1
    [[ "$line" == *\** || "$line" == *\?* || "$line" == *\[* ]] && return 1
    return 0
}

# Return 0 (success) if $1 (a folder, without leading/trailing slash) is a
# strict ancestor of $2 (a path without leading slash).
is_strict_ancestor() {
    local ancestor="$1"
    local path="$2"
    [[ "$path" == "$ancestor"/* ]]
}

# Find the deepest folder entry in .gitignore that covers $1 (path w/o leading /).
# Echoes the folder (no leading/trailing slash), or empty string if none.
find_deepest_ancestor_entry() {
    local target="$1"
    local deepest=""
    local deepest_depth=-1
    while IFS= read -r line; do
        if is_simple_folder_entry "$line"; then
            local folder="${line#/}"
            folder="${folder%/}"
            if is_strict_ancestor "$folder" "$target"; then
                # depth = number of slashes in folder + 1 ; longer string = deeper
                local depth=${#folder}
                if [ "$depth" -gt "$deepest_depth" ]; then
                    deepest="$folder"
                    deepest_depth="$depth"
                fi
            fi
        fi
    done < .gitignore
    echo "$deepest"
}

# Remove a specific line from .gitignore (exact match).
remove_exact_line() {
    local needle="$1"
    local tmp
    tmp=$(mktemp)
    while IFS= read -r line; do
        [ "$line" = "$needle" ] && continue
        echo "$line" >> "$tmp"
    done < .gitignore
    mv "$tmp" .gitignore
}

# Expand a folder entry /ancestor/ ONE LEVEL into its direct children.
# Files under it become /ancestor/file ; subfolders become /ancestor/subfolder/.
# The original /ancestor/ line is removed.
# Skip any child that matches $skip_child (pass empty string to skip nothing).
expand_one_level() {
    local ancestor="$1"      # without leading/trailing slash
    local skip_child="$2"    # child path (relative to repo root) to omit, or ""

    if [ ! -d "$ancestor" ]; then
        echo "  Warning: '$ancestor' doesn't exist on disk; cannot expand." >&2
        return 1
    fi

    # Read existing lines into a set for dedup
    declare -A existing=()
    while IFS= read -r l; do
        [ -n "$l" ] && existing["$l"]=1
    done < .gitignore

    # Remove the ancestor line
    remove_exact_line "/$ancestor/"
    unset 'existing[/'"$ancestor"'/]'

    # Append direct children
    local added=0
    local skipped=0
    while IFS= read -r child; do
        [ -z "$child" ] && continue
        child="${child#./}"

        # Determine entry format: file -> /path ; dir -> /path/
        local entry
        if [ -d "$child" ]; then
            entry="/$child/"
        else
            entry="/$child"
        fi

        # Skip the target (and its descendants if target is inside this child)
        if [ -n "$skip_child" ]; then
            # If child == skip_child exactly → drop it.
            # If skip_child is a descendant of child → keep child but mark for further expansion
            # (caller will handle that). Here we only drop exact matches.
            if [ "$child" = "$skip_child" ]; then
                skipped=$((skipped + 1))
                continue
            fi
        fi

        if [ -z "${existing[$entry]:-}" ]; then
            echo "$entry" >> .gitignore
            existing["$entry"]=1
            added=$((added + 1))
        fi
    done < <(find "$ancestor" -mindepth 1 -maxdepth 1 | sed 's|^\./||')

    echo "  Expanded '/$ancestor/' → $added child entries (skipped $skipped)"
    return 0
}

# Main routine: un-ignore a single path.
un_ignore_path() {
    local input="$1"

    # Normalize
    input="${input#./}"
    input="${input#/}"
    input="${input%/}"
    [ -z "$input" ] && { echo "Empty path."; return 1; }

    if [ ! -e "$input" ]; then
        echo "Warning: '$input' does not exist on disk. Please enter an existing path."
        return 1
    fi

    # Determine exact-match entry shapes
    local is_dir=0
    [ -d "$input" ] && is_dir=1
    local exact_file="/$input"
    local exact_dir="/$input/"

    # Case 1: exact match(es) in .gitignore → remove them
    local removed_exact=0
    if grep -Fxq "$exact_file" .gitignore; then
        remove_exact_line "$exact_file"
        echo "Removed exact entry: $exact_file"
        removed_exact=1
    fi
    if [ "$is_dir" -eq 1 ] && grep -Fxq "$exact_dir" .gitignore; then
        remove_exact_line "$exact_dir"
        echo "Removed exact entry: $exact_dir"
        removed_exact=1
    fi

    # Case 2: peel ancestor folder entries, one level at a time, until the
    # target is no longer covered by any ancestor.
    local expanded_any=0
    while :; do
        local ancestor
        ancestor=$(find_deepest_ancestor_entry "$input")
        [ -z "$ancestor" ] && break

        echo "Target '$input' is covered by ancestor '/$ancestor/'. Expanding one level ..."

        # The direct child of $ancestor that leads to $input (or equals $input).
        # e.g. ancestor=a, input=a/b/c.txt  →  direct_child=a/b
        local rest="${input#$ancestor/}"
        local first_segment="${rest%%/*}"
        local direct_child="$ancestor/$first_segment"

        # If direct_child IS the target, omit it from expansion entirely.
        # Otherwise, keep it in the expansion (we'll descend into it next loop).
        local skip=""
        if [ "$direct_child" = "$input" ]; then
            skip="$input"
        fi

        expand_one_level "$ancestor" "$skip" || break
        expanded_any=1
    done

    if [ "$removed_exact" -eq 0 ] && [ "$expanded_any" -eq 0 ]; then
        echo "Note: '$input' was not covered by any entry in .gitignore. Nothing to do."
    else
        echo "Done un-ignoring '$input'."
    fi
}

# --- interactive loop ---

echo "Interactive .gitignore un-ignore tool."
echo "Enter a path to remove from .gitignore. Type q/quit/exit or empty line to finish."
echo ""

while :; do
    read -r -p "Path to un-ignore> " input
    case "$input" in
        ""|q|quit|exit) echo "Bye."; break ;;
    esac
    un_ignore_path "$input"
    echo ""
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/gitignore_consolidate.sh"