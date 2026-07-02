#!/usr/bin/env bash
# Consolidate .gitignore: if a folder's real files are ALL already listed as
# individual entries in .gitignore, replace those entries with a single folder entry.
# Also sort the file alphabetically within each section, preserving comment headers.

cd "$(git rev-parse --show-toplevel)"

[ -f .gitignore ] || { echo "No .gitignore found."; exit 0; }

# Work on a temp copy
TMP=$(mktemp)
cp .gitignore "$TMP"

# Helper: is a line a "simple file entry" we're willing to merge?
is_simple_file_entry() {
    local line="$1"
    [[ -z "$line" ]] && return 1
    [[ "$line" == \#* ]] && return 1
    [[ "$line" == !* ]] && return 1
    [[ "$line" != /* ]] && return 1
    [[ "$line" == */ ]] && return 1
    [[ "$line" == *\** || "$line" == *\?* || "$line" == *\[* ]] && return 1
    return 0
}

# Helper: is a line a "sortable entry" (simple file OR simple folder, no globs/negation)?
# These are the lines we'll sort within a section.
is_sortable_entry() {
    local line="$1"
    [[ -z "$line" ]] && return 1
    [[ "$line" == \#* ]] && return 1
    [[ "$line" == !* ]] && return 1
    [[ "$line" != /* ]] && return 1
    [[ "$line" == *\** || "$line" == *\?* || "$line" == *\[* ]] && return 1
    return 0
}

# --- Consolidation loop (unchanged logic) ---
while :; do
    CHANGED=0
    unset GROUPS
    declare -A GROUPS=()

    # Collect all folder entries (/some/folder/) currently in TMP — these
    # can cover files implicitly, so we treat them as valid coverage too.
    unset FOLDER_IGNORES
    declare -A FOLDER_IGNORES=()
    while IFS= read -r line; do
        if [[ "$line" == /*/ && "$line" != *\** && "$line" != *\?* \
              && "$line" != *\[* && "$line" != !* && "$line" != \#* ]]; then
            folder="${line#/}"
            folder="${folder%/}"
            FOLDER_IGNORES["$folder"]=1
        fi
    done < "$TMP"

    # Helper: is $1 covered by any folder entry in FOLDER_IGNORES?
    # Walks up the path checking each ancestor.
    is_covered_by_folder_entry() {
        local f="$1"
        local d
        d=$(dirname "$f")
        while [ "$d" != "." ] && [ "$d" != "/" ]; do
            if [ -n "${FOLDER_IGNORES[$d]:-}" ]; then
                return 0
            fi
            d=$(dirname "$d")
        done
        return 1
    }

    while IFS= read -r line; do
        if is_simple_file_entry "$line"; then
            rel="${line#/}"
            parent=$(dirname "$rel")
            [ "$parent" = "." ] && continue
            GROUPS["$parent"]+="$rel"$'\n'
        fi
    done < "$TMP"

    for parent in "${!GROUPS[@]}"; do
        [ -d "$parent" ] || continue

        mapfile -t real_files < <(find "$parent" -type f -not -path "*/.git/*" | sed 's|^\./||')
        [ "${#real_files[@]}" -eq 0 ] && continue

        declare -A ignored_set=()
        while IFS= read -r f; do
            [ -z "$f" ] && continue
            ignored_set["$f"]=1
        done <<< "${GROUPS[$parent]}"

        all_covered=1
        any_tracked=0
        for rf in "${real_files[@]}"; do
            # A file is "covered" if either:
            #   (a) it's listed as a single-file entry, OR
            #   (b) some ancestor folder entry covers it
            if [ -z "${ignored_set[$rf]:-}" ] && ! is_covered_by_folder_entry "$rf"; then
                all_covered=0
                break
            fi
            if git ls-files --error-unmatch "$rf" >/dev/null 2>&1; then
                any_tracked=1
            fi
        done

        unset ignored_set

        if [ "$all_covered" -eq 1 ] && [ "$any_tracked" -eq 0 ]; then
            echo "Consolidating '$parent/' — ${#real_files[@]} file entries -> 1 folder entry"

            NEW_TMP=$(mktemp)
            while IFS= read -r line; do
                # Drop single-file entries whose parent matches
                if is_simple_file_entry "$line"; then
                    rel="${line#/}"
                    p=$(dirname "$rel")
                    if [ "$p" = "$parent" ]; then
                        continue
                    fi
                fi
                # Drop redundant child folder entries (e.g. /file_1/file_2/
                # when we're about to write /file_1/)
                if [[ "$line" == /*/ && "$line" != *\** && "$line" != *\?* \
                      && "$line" != *\[* && "$line" != !* && "$line" != \#* ]]; then
                    child_folder="${line#/}"
                    child_folder="${child_folder%/}"
                    if [[ "$child_folder" == "$parent"/* ]]; then
                        continue
                    fi
                fi
                echo "$line"
            done < "$TMP" > "$NEW_TMP"

            FOLDER_ENTRY="/$parent/"
            if ! grep -Fxq "$FOLDER_ENTRY" "$NEW_TMP"; then
                if [ -s "$NEW_TMP" ] && [ "$(tail -c 1 "$NEW_TMP")" != "" ]; then
                    printf "\n" >> "$NEW_TMP"
                fi
                echo "$FOLDER_ENTRY" >> "$NEW_TMP"
            fi

            mv "$NEW_TMP" "$TMP"
            CHANGED=1
        fi
    done

    unset GROUPS
    [ "$CHANGED" -eq 0 ] && break
done

# --- Section-aware sort ---
# Split file into sections at blank lines.
# Within each section:
#   - keep leading comment lines as a "header" in original order
#   - if the rest of the section is entirely sortable entries, sort them (unique)
#   - if the rest contains any glob/negation/unsortable line, leave the whole section untouched
sort_sections() {
    local src="$1"
    local dst="$2"

    # Read full file into an array (preserve blank lines)
    mapfile -t LINES < "$src"

    : > "$dst"

    local i=0
    local n=${#LINES[@]}
    local wrote_any_section=0

    while [ "$i" -lt "$n" ]; do
        # Skip over any blank lines between sections
        while [ "$i" -lt "$n" ] && [ -z "${LINES[$i]}" ]; do
            i=$((i + 1))
        done

        # Collect one section: everything until the next blank line (or EOF)
        local section=()
        while [ "$i" -lt "$n" ] && [ -n "${LINES[$i]}" ]; do
            section+=( "${LINES[$i]}" )
            i=$((i + 1))
        done

        # Nothing collected (trailing blanks only) — done
        [ "${#section[@]}" -eq 0 ] && continue

        # Separate leading comments (header) from body
        local header=()
        local body=()
        local in_body=0
        local sortable=1

        for line in "${section[@]}"; do
            if [ "$in_body" -eq 0 ] && [[ "$line" == \#* ]]; then
                header+=( "$line" )
            else
                in_body=1
                body+=( "$line" )
                if ! is_sortable_entry "$line"; then
                    sortable=0
                fi
            fi
        done

        # Insert exactly one blank line between sections
        if [ "$wrote_any_section" -eq 1 ]; then
            echo "" >> "$dst"
        fi
        wrote_any_section=1

        # Write header as-is
        for h in "${header[@]}"; do
            echo "$h" >> "$dst"
        done

        # Write body: sorted if all sortable, otherwise original order
        if [ "${#body[@]}" -gt 0 ]; then
            if [ "$sortable" -eq 1 ]; then
                # Sort and dedupe, then group by top-level path component.
                # Insert a blank line between groups with different top-level prefixes.
                local prev_prefix=""
                local first_in_body=1
                while IFS= read -r entry; do
                    [ -z "$entry" ] && continue
                    # Extract top-level component: "/foo/bar/baz" -> "foo"
                    local stripped="${entry#/}"
                    local prefix="${stripped%%/*}"

                    if [ "$first_in_body" -eq 0 ] && [ "$prefix" != "$prev_prefix" ]; then
                        echo "" >> "$dst"
                    fi
                    echo "$entry" >> "$dst"
                    prev_prefix="$prefix"
                    first_in_body=0
                done < <(printf "%s\n" "${body[@]}" | sort -u)
            else
                for b in "${body[@]}"; do
                    echo "$b" >> "$dst"
                done
            fi
        fi
    done
}

SORTED_TMP=$(mktemp)
sort_sections "$TMP" "$SORTED_TMP"
mv "$SORTED_TMP" "$TMP"

# --- Write back if different ---
if ! cmp -s "$TMP" .gitignore; then
    cp "$TMP" .gitignore
    echo "Updated .gitignore (consolidated + sorted)."
else
    echo "No changes needed."
fi

rm -f "$TMP"

# --- Final pass: force exactly one blank line between every non-empty line ---
FINAL_TMP=$(mktemp)
first=1
while IFS= read -r line || [ -n "$line" ]; do
    [ -z "$line" ] && continue
    if [ "$first" -eq 0 ]; then
        echo "" >> "$FINAL_TMP"
    fi
    echo "$line" >> "$FINAL_TMP"
    first=0
done < .gitignore
mv "$FINAL_TMP" .gitignore
echo "Forced single blank line between every entry."