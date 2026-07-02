#!/usr/bin/env bash
# in this file, all folders as git repositories will be detected, and see whether there are
# any uncommitted changes, or, any update from remote repo
# it will list each repo with both situation, and then prompt the user to choose what to do next
# specifically, it will ask whether the user will selectively push or pull

# git_search.sh
#
# Scan a base directory for git repositories and show, for each:
#   - whether it has uncommitted local changes (staged / unstaged)
#   - whether there are incoming updates from remote (behind upstream)
#   - whether there are local commits not yet pushed (ahead of upstream)
#
# After displaying, presents a menu for selective batch pull / batch push.

# uncomment on Linux if edited on Windows:
# sed -i 's/\r$//' "$0"

# Resolve this script's own directory (handy for any sibling tool scripts).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Central source of truth for the git_script/ folder shared across all repos.
# When the user asks to check/update script versions, we first `git pull` here,
# then diff every repo's git_script/ against these files.
TEMPLATE_REPO="/c/Users/yi_lu/my_files/script_tool/git_script_template"
TEMPLATE_DIR="$TEMPLATE_REPO/git_script"

# --- 1. Ask for base directory ---
DEFAULT_BASE="$PWD"
read -r -p "Base directory to scan (Enter for $DEFAULT_BASE): " BASE_DIR
BASE_DIR="${BASE_DIR:-$DEFAULT_BASE}"
# Convert any Windows-style backslashes to forward slashes
BASE_DIR="${BASE_DIR//\\//}"

if [ ! -d "$BASE_DIR" ]; then
    echo "Directory not found: $BASE_DIR"
    exit 1
fi

# --- 2. Find all git repos ---
echo "Searching for git repositories under: $BASE_DIR"

mapfile -t RAW_REPOS < <(find "$BASE_DIR" -name ".git" -prune \
                         2>/dev/null | sed 's#/\.git$##' | sort)

# Keep only GitHub repos: at least one remote URL must contain "github.com".
# This excludes purely-local repos (no remote), JetBrains settingsSync,
# vendored .git directories in conda/site-packages, GitLab/Bitbucket, etc.
REPOS=()
SKIPPED=0
for R in "${RAW_REPOS[@]}"; do
    # Get all remote URLs for this repo and check if any point to github.com
    if git -C "$R" remote -v 2>/dev/null | grep -qi "github\.com"; then
        REPOS+=("$R")
    else
        SKIPPED=$((SKIPPED + 1))
    fi
done

if [ "$SKIPPED" -gt 0 ]; then
    echo "Skipped $SKIPPED non-GitHub repo(s) (no github.com remote)."
fi

if [ "${#REPOS[@]}" -eq 0 ]; then
    echo "No GitHub repositories found under $BASE_DIR."
    exit 0
fi

echo "Found ${#REPOS[@]} GitHub repo(s). Inspecting status (fetching from remotes) ..."
echo

# --- 3. Inspect each repo ---
# We collect raw results into parallel arrays so the menu at the end can
# operate on them. Each index i describes the same repo.
STATUS_REPO=()
STATUS_BRANCH=()
STATUS_LABEL=()
STATUS_BEHIND=()
STATUS_AHEAD=()
STATUS_DETAIL=()

# Collect output via a single temp file (subshells can't mutate parent arrays).
SCAN_TMP=$(mktemp)

for REPO in "${REPOS[@]}"; do
    (
        cd "$REPO" 2>/dev/null || { printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
            "ERROR" "$REPO" "-" "0" "0" "cannot cd"; exit 0; }

        BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
        [ -z "$BRANCH" ] && BRANCH="(detached)"

        # Uncommitted local changes — distinguish staged vs unstaged.
        # `git status --porcelain` output: col1=staged flag, col2=unstaged flag
        #   " M" = unstaged modification
        #   "M " = staged modification
        #   "MM" = both
        #   "??" = untracked file (counts as unstaged)
        PORCELAIN=$(git status --porcelain 2>/dev/null)
        HAS_STAGED=0
        HAS_UNSTAGED=0
        if [ -n "$PORCELAIN" ]; then
            while IFS= read -r pline; do
                [ -z "$pline" ] && continue
                c1="${pline:0:1}"
                c2="${pline:1:1}"
                # Staged column: any non-space, non-? means staged change
                [ "$c1" != " " ] && [ "$c1" != "?" ] && HAS_STAGED=1
                # Unstaged column: non-space means working-tree change; ?? is untracked
                [ "$c2" != " " ] && HAS_UNSTAGED=1
            done <<< "$PORCELAIN"
        fi

        # Incoming / outgoing vs. upstream
        BEHIND=0
        AHEAD=0
        DETAIL=""
        if git config --get remote.origin.url >/dev/null 2>&1 \
           && [ "$BRANCH" != "(detached)" ]; then
            git fetch --quiet 2>/dev/null || DETAIL="(fetch failed) "
            UPSTREAM=$(git rev-parse --abbrev-ref --symbolic-full-name "@{u}" 2>/dev/null)
            if [ -n "$UPSTREAM" ]; then
                BEHIND=$(git rev-list --count "HEAD..$UPSTREAM" 2>/dev/null || echo 0)
                AHEAD=$(git rev-list --count "$UPSTREAM..HEAD" 2>/dev/null || echo 0)
            else
                DETAIL="${DETAIL}(no upstream) "
            fi
        else
            DETAIL="(no remote) "
        fi

        # Compose status label — order: STAGED, UNSTAGED, BEHIND, AHEAD
        PARTS=()
        [ "$HAS_STAGED"   = "1" ] && PARTS+=("STAGED")
        [ "$HAS_UNSTAGED" = "1" ] && PARTS+=("UNSTAGED")
        [ "${BEHIND:-0}" -gt 0 ]  && PARTS+=("BEHIND")
        [ "${AHEAD:-0}"  -gt 0 ]  && PARTS+=("AHEAD")
        if [ "${#PARTS[@]}" -eq 0 ]; then
            LABEL="CLEAN"
        else
            LABEL=$(IFS='+'; echo "${PARTS[*]}")
        fi

        [ "${BEHIND:-0}" -gt 0 ] && DETAIL="${DETAIL}behind $BEHIND "
        [ "${AHEAD:-0}"  -gt 0 ] && DETAIL="${DETAIL}ahead $AHEAD "

        printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$LABEL" "$REPO" "$BRANCH" "${BEHIND:-0}" "${AHEAD:-0}" "$DETAIL"
    )
done > "$SCAN_TMP"

# Populate arrays from scan results (no display yet)
while IFS=$'\t' read -r LBL REPO BR BH AH DET; do
    STATUS_LABEL+=("$LBL")
    STATUS_REPO+=("$REPO")
    STATUS_BRANCH+=("$BR")
    STATUS_BEHIND+=("$BH")
    STATUS_AHEAD+=("$AH")
    STATUS_DETAIL+=("$DET")
done < "$SCAN_TMP"
rm -f "$SCAN_TMP"

# Group repos by what the user needs to do next.
# Pull-first ordering (most blocking → least blocking → nothing to do):
#
#   DIRTY          — has BEHIND + local changes (STAGED/UNSTAGED/AHEAD).
#                    Must pull first (stash+pull+pop), then push.
#   BEHIND         — BEHIND only, no local changes. Just pull.
#   READY_TO_PUSH  — has local changes (STAGED/UNSTAGED/AHEAD) but NOT behind.
#                    Push directly.
#   CLEAN          — fully in sync.
GRP_DIRTY=()
GRP_BEHIND=()
GRP_READY=()
GRP_CLEAN=()
for i in "${!STATUS_LABEL[@]}"; do
    lbl="${STATUS_LABEL[$i]}"
    has_local=0
    has_behind=0
    [[ "$lbl" == *STAGED* || "$lbl" == *UNSTAGED* || "$lbl" == *AHEAD* ]] && has_local=1
    [[ "$lbl" == *BEHIND* ]] && has_behind=1

    if   [ "$has_behind" = "1" ] && [ "$has_local" = "1" ]; then GRP_DIRTY+=("$i")
    elif [ "$has_behind" = "1" ];                            then GRP_BEHIND+=("$i")
    elif [ "$has_local"  = "1" ];                            then GRP_READY+=("$i")
    else                                                          GRP_CLEAN+=("$i")
    fi
done

# Pretty-print one group. Args: <group title> <description> <array name> <start-index-var-name>
# Uses a GLOBAL counter so numbers are continuous across groups.
# The start-index variable (passed by name) is incremented by the number of entries printed.
print_group() {
    local title="$1"
    local desc="$2"
    local -n arr="$3"
    local -n counter="$4"

    echo "======== $title (${#arr[@]}) — $desc ========"
    if [ "${#arr[@]}" -eq 0 ]; then
        echo "  (none)"
        echo
        return
    fi
    printf "  %-5s  %-20s  %-40s  %s\n" "#" "STATUS" "REPO" "BRANCH / DETAIL"
    for idx in "${arr[@]}"; do
        printf "  %-5s  %-20s  %-40s  %s  %s\n" \
            "${counter}." \
            "${STATUS_LABEL[$idx]}" \
            "${STATUS_REPO[$idx]}" \
            "${STATUS_BRANCH[$idx]}" \
            "${STATUS_DETAIL[$idx]}"
        # Also record the group-ordered display index: GLOBAL_ORDER[$counter] = $idx
        GLOBAL_ORDER[$counter]="$idx"
        counter=$((counter + 1))
    done
    echo
}

# Global map: display_number (1-based) -> STATUS_* array index
# Populated by print_group calls below.
GLOBAL_ORDER=()

DISPLAY_N=1
print_group "DIRTY"         "local changes AND behind — pull first, then push"       GRP_DIRTY  DISPLAY_N
print_group "BEHIND"        "remote has commits you haven't pulled (clean locally)"  GRP_BEHIND DISPLAY_N
print_group "READY_TO_PUSH" "local changes, not behind — safe to push directly"      GRP_READY  DISPLAY_N
print_group "CLEAN"         "fully in sync"                                          GRP_CLEAN  DISPLAY_N

TOTAL_DISPLAYED=$((DISPLAY_N - 1))

echo "================================================================"
echo "Label meanings (shown in STATUS column):"
echo "  STAGED   — added to index (\`git add\`), not yet committed"
echo "  UNSTAGED — modified in working tree OR untracked new file"
echo "  AHEAD    — you have local commits not yet pushed"
echo "  BEHIND   — remote has commits you haven't pulled"
echo "  CLEAN    — fully in sync"
echo "  (combinations like 'STAGED+UNSTAGED+BEHIND+AHEAD' are possible)"
echo
echo "Groups (by what to do next, pull-first order):"
echo "  DIRTY         — has BEHIND + local changes: pull first, then push"
echo "  BEHIND        — only behind remote: pull to fast-forward"
echo "  READY_TO_PUSH — has local changes, not behind: push directly"
echo "  CLEAN         — nothing to do"
echo
echo "  NOTICE: Always PULL before PUSH."
echo "  Pushing on top of a behind branch will be rejected by the remote,"
echo "  and pulling first avoids unnecessary merge commits."
echo "================================================================"
echo

# --- 4. Helpers for batch operations ---

# Print an indexed candidate list and let the user pick.
# Args: <list-title> <array-name of indices into STATUS_*>
# Returns chosen indices (space-separated) on stdout.
# Parse user selection string against GLOBAL_ORDER (1-based display numbers
# mapped to STATUS_* indices).
#
# Accepted syntax:
#   all
#   single:     1
#   list:       1, 3, 5
#   range:      1-4
#   mixed:      1-4, 6, 8-10
#   (whitespace around numbers and commas is ignored)
#
# Returns chosen STATUS_* indices (space-separated) on stdout.
# Exit codes:
#   0 — success (prints picks to stdout)
#   1 — empty input (user cancelled)
#   2 — invalid syntax
#   3 — out of range
parse_selection() {
    local input="$1"
    local max="$2"

    # Trim leading/trailing whitespace
    input="${input#"${input%%[![:space:]]*}"}"
    input="${input%"${input##*[![:space:]]}"}"

    [ -z "$input" ] && return 1

    if [ "$input" = "all" ]; then
        local result=()
        local n
        for ((n = 1; n <= max; n++)); do
            result+=("${GLOBAL_ORDER[$n]}")
        done
        echo "${result[*]}"
        return 0
    fi

    # First pass: validate syntax on the WHOLE input. Any deviation → reject.
    # Allowed overall pattern: TOKEN (, TOKEN)*
    # TOKEN := number | number-number   (each number is 1+ digits)
    # We tolerate spaces around commas and hyphens.
    local stripped="${input// /}"
    if ! [[ "$stripped" =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]]; then
        return 2
    fi

    # Second pass: expand, checking ranges
    local result=()
    local IFS_BAK="$IFS"
    IFS=','
    for tok in $stripped; do
        IFS="$IFS_BAK"
        if [[ "$tok" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            local lo="${BASH_REMATCH[1]}"
            local hi="${BASH_REMATCH[2]}"
            if [ "$lo" -gt "$hi" ]; then
                return 2    # backwards range like 5-3 is a syntax problem
            fi
            if [ "$lo" -lt 1 ] || [ "$hi" -gt "$max" ]; then
                return 3
            fi
            local n
            for ((n = lo; n <= hi; n++)); do
                result+=("${GLOBAL_ORDER[$n]}")
            done
        else
            local n="$tok"
            if [ "$n" -lt 1 ] || [ "$n" -gt "$max" ]; then
                return 3
            fi
            result+=("${GLOBAL_ORDER[$n]}")
        fi
        IFS=','
    done
    IFS="$IFS_BAK"

    echo "${result[*]}"
    return 0
}

# List repos whose label matches ANY of the given glob patterns before prompting.
# Uses global display numbers so they line up with the grouped output above.
# Usage: list_candidates "header text" "*AHEAD*" ["*STAGED*" "*UNSTAGED*" ...]
list_candidates() {
    local header="$1"
    shift
    local patterns=("$@")

    local found=0
    echo "$header:" >&2
    for n in "${!GLOBAL_ORDER[@]}"; do
        local idx="${GLOBAL_ORDER[$n]}"
        local lbl="${STATUS_LABEL[$idx]}"
        local match=0
        for pat in "${patterns[@]}"; do
            if [[ "$lbl" == $pat ]]; then
                match=1
                break
            fi
        done
        if [ "$match" = "1" ]; then
            printf "  %-5s  %-20s  %-40s  %s  %s\n" \
                "${n}." \
                "${STATUS_LABEL[$idx]}" \
                "${STATUS_REPO[$idx]}" \
                "${STATUS_BRANCH[$idx]}" \
                "${STATUS_DETAIL[$idx]}" >&2
            found=1
        fi
    done
    [ "$found" -eq 0 ] && echo "  (none)" >&2
    echo >&2
}

# Prompt for a selection, parse, and return picks. Handles error reporting.
# Usage: picks=$(prompt_selection "push") || continue
prompt_selection() {
    local action="$1"   # "push" or "pull" — purely for prompt text
    echo "Enter repos to $action (by display number)." >&2
    echo "  Examples:  1         single" >&2
    echo "             1,3,5     list" >&2
    echo "             1-4       range" >&2
    echo "             1-4,6,8-10  mixed" >&2
    echo "             all         every repo" >&2
    echo "  Enter an empty line to cancel." >&2
    read -r -p "> " input

    local picks
    picks=$(parse_selection "$input" "$TOTAL_DISPLAYED")
    local rc=$?

    case "$rc" in
        0) echo "$picks"; return 0 ;;
        1) return 1 ;;   # cancelled
        2) echo "invalid syntax" >&2; return 2 ;;
        3) echo "out of range" >&2; return 3 ;;
    esac
}

# Run 'git pull' on a specific repo index.
# If the repo is DIRTY (STAGED/UNSTAGED), automatically stash + pull + pop.
# On success, mutate STATUS_* arrays so the menu reflects the new state.
do_pull_one() {
    local idx="$1"
    local repo="${STATUS_REPO[$idx]}"
    local branch="${STATUS_BRANCH[$idx]}"
    local label="${STATUS_LABEL[$idx]}"

    echo "--- $repo ($branch) ---"

    # Auto-stash whenever there are local changes — no prompt.
    local use_stash=0
    if [[ "$label" == *STAGED* || "$label" == *UNSTAGED* ]]; then
        use_stash=1
        echo "  (local changes present — will stash, pull, then pop)"
    fi

    local pull_ok=0
    (
        cd "$repo" || exit 1

        if [ "$use_stash" = "1" ]; then
            echo "  Stashing ..."
            if ! git stash push -u -m "git_search.sh autostash" >/dev/null 2>&1; then
                echo "  stash failed; aborting pull for this repo."
                exit 1
            fi
        fi

        echo "  Pulling ..."
        if git pull --ff-only; then
            if [ "$use_stash" = "1" ]; then
                echo "  Popping stash (preserving staged/unstaged state) ..."
                if ! git stash pop --index; then
                    echo "  !! Stash pop had conflicts. Resolve manually in $repo. !!"
                    exit 2
                fi
            fi
            exit 0
        else
            echo "  Fast-forward failed. You may need to merge/rebase manually."
            [ "$use_stash" = "1" ] && git stash pop --index >/dev/null 2>&1
            exit 1
        fi
    )
    local rc=$?
    [ "$rc" -eq 0 ] && pull_ok=1

    # On success, update in-memory status: BEHIND is cleared, DIRTY parts stay
    if [ "$pull_ok" = "1" ]; then
        STATUS_BEHIND[$idx]=0
        local parts=()
        [[ "$label" == *STAGED*   ]] && parts+=("STAGED")
        [[ "$label" == *UNSTAGED* ]] && parts+=("UNSTAGED")
        [ "${STATUS_AHEAD[$idx]:-0}" -gt 0 ] && parts+=("AHEAD")
        if [ "${#parts[@]}" -eq 0 ]; then
            STATUS_LABEL[$idx]="CLEAN"
        else
            STATUS_LABEL[$idx]=$(IFS='+'; echo "${parts[*]}")
        fi
    fi
    echo
}

# Run the target repo's own git_script/git_push.sh on a specific repo index.
# Skip repos that are BEHIND (unsafe to push — must pull first) with a notice.
# Uses the default commit message "update" for every repo.
do_push_one() {
    local idx="$1"
    local repo="${STATUS_REPO[$idx]}"
    local branch="${STATUS_BRANCH[$idx]}"
    local label="${STATUS_LABEL[$idx]}"

    echo "--- $repo ($branch) ---"

    # Re-check BEHIND right now, not from the stale scan snapshot. The repo's
    # remote state may have changed (or our scan's fetch may have failed
    # silently), so we fetch fresh and recount before deciding to push.
    local behind_now=0
    if [ "$branch" != "(detached)" ]; then
        (cd "$repo" && git fetch --quiet 2>/dev/null) || true
        behind_now=$(cd "$repo" && git rev-list --count "HEAD..@{u}" 2>/dev/null || echo 0)
    fi

    if [ "${behind_now:-0}" -gt 0 ]; then
        echo "  Skipped: repo is BEHIND by $behind_now commit(s). Pull first, then retry push."
        # Keep STATUS_BEHIND in sync with reality
        STATUS_BEHIND[$idx]="$behind_now"
        echo
        return 0
    fi

    # Each repo carries its own git_script/git_push.sh — prefer it.
    # It does: git add . → scan for large files → commit → push.
    # If the repo doesn't ship that script, fall back to the same behavior
    # inline (add everything → commit → push) so the net effect is identical.
    local helper="$repo/git_script/git_push.sh"
    local push_ok=0
    if [ -f "$helper" ]; then
        (cd "$repo" && echo "  Running git_script/git_push.sh ..." && bash "$helper" "update") && push_ok=1
    else
        echo "  No repo-local git_script/git_push.sh; doing inline add + commit + push."
        (
            cd "$repo" || exit 1
            git add .
            # If there's nothing new to commit (e.g. only AHEAD, no working-tree
            # changes), skip the commit step but still attempt the push.
            if ! git diff --cached --quiet; then
                git commit -m "update" || exit 1
            fi
            git push origin "$branch"
        ) && push_ok=1
    fi

    if [ "$push_ok" = "1" ]; then
        # Both paths commit+push everything local, so STAGED/UNSTAGED/AHEAD
        # are all cleared. BEHIND (if any) stays.
        STATUS_AHEAD[$idx]=0
        local parts=()
        [ "${STATUS_BEHIND[$idx]:-0}" -gt 0 ] && parts+=("BEHIND")
        if [ "${#parts[@]}" -eq 0 ]; then
            STATUS_LABEL[$idx]="CLEAN"
        else
            STATUS_LABEL[$idx]=$(IFS='+'; echo "${parts[*]}")
        fi
    fi
    echo
}

# Refresh the source-of-truth git_script folder by pulling latest from GitHub.
# Return 0 on success (including when already up-to-date), 1 on failure.
refresh_template() {
    if [ ! -d "$TEMPLATE_DIR" ]; then
        echo "  ERROR: template dir not found: $TEMPLATE_DIR"
        return 1
    fi
    echo "Refreshing template from origin ..."
    if (cd "$TEMPLATE_REPO" && git pull --ff-only --quiet); then
        echo "  Template is up-to-date."
        return 0
    else
        echo "  Warning: couldn't pull template (diverged? offline?). Using local version as-is."
        return 0   # non-fatal: we can still compare against whatever is local
    fi
}

# Check whether a single repo's git_script/ is out-of-date vs. the template.
# Follows the standard shell "truthy" convention where return 0 = yes.
# Returns:
#   0 — YES, out-of-date (missing files or content differs)
#   1 — NO, up-to-date (or this repo IS the template itself)
is_script_outdated() {
    local repo="$1"

    # Don't compare the template repo against itself — treat as not-outdated
    if [ "$(cd "$repo" && pwd)" = "$(cd "$TEMPLATE_REPO" && pwd)" ]; then
        return 1
    fi

    local target="$repo/git_script"

    # Missing folder entirely = outdated
    if [ ! -d "$target" ]; then
        return 0
    fi

    # Compare each .sh file in the template to the corresponding one in the repo
    for f in "$TEMPLATE_DIR"/*.sh; do
        [ -f "$f" ] || continue
        local name
        name=$(basename "$f")
        if ! cmp -s "$f" "$target/$name"; then
            return 0
        fi
    done

    return 1
}

# For every repo, mark whether its git_script/ matches the template.
# Populates global arrays:
#   OUTDATED_REPOS  — STATUS_* indices for repos that need updating
#   UPTODATE_REPOS  — STATUS_* indices for repos already in sync (feel-good list)
# Note: the template repo itself is excluded from BOTH lists — comparing it
# against itself is meaningless.
scan_outdated_scripts() {
    OUTDATED_REPOS=()
    UPTODATE_REPOS=()
    local template_abs
    template_abs="$(cd "$TEMPLATE_REPO" 2>/dev/null && pwd)"
    for i in "${!STATUS_REPO[@]}"; do
        local repo="${STATUS_REPO[$i]}"
        # Skip template repo in both lists
        if [ "$(cd "$repo" 2>/dev/null && pwd)" = "$template_abs" ]; then
            continue
        fi
        if is_script_outdated "$repo"; then
            OUTDATED_REPOS+=("$i")
        else
            UPTODATE_REPOS+=("$i")
        fi
    done
}

# Print the list of repos whose git_script/ matches the template.
# Pure informational — no numbering needed, no selection, just a feel-good
# confirmation that these are already in sync.
list_uptodate() {
    echo "Repos already in sync with the template (${#UPTODATE_REPOS[@]}):" >&2
    if [ "${#UPTODATE_REPOS[@]}" -eq 0 ]; then
        echo "  (none)" >&2
        echo >&2
        return
    fi
    for idx in "${UPTODATE_REPOS[@]}"; do
        printf "  ✓  %s\n" "${STATUS_REPO[$idx]}" >&2
    done
    echo >&2
}

# Print the list of repos whose git_script/ is out-of-date.
# Uses its own per-call numbering (1..N over OUTDATED_REPOS) because the
# main GLOBAL_ORDER display numbers aren't meaningful for this list.
list_outdated() {
    OUTDATED_DISPLAY=()
    echo "Repos whose git_script/ differs from the template:" >&2
    if [ "${#OUTDATED_REPOS[@]}" -eq 0 ]; then
        echo "  (none — all repos already up-to-date)" >&2
        echo >&2
        return
    fi
    local n=1
    for idx in "${OUTDATED_REPOS[@]}"; do
        local target="${STATUS_REPO[$idx]}/git_script"
        local reason="modified"
        [ ! -d "$target" ] && reason="MISSING"
        printf "  %-5s  %-10s  %s\n" "${n}." "$reason" "${STATUS_REPO[$idx]}" >&2
        OUTDATED_DISPLAY[$n]="$idx"
        n=$((n + 1))
    done
    echo >&2
}

# Parse a selection against OUTDATED_DISPLAY (its own 1..N numbering, not
# GLOBAL_ORDER). Same syntax as parse_selection but for the outdated list.
parse_outdated_selection() {
    local input="$1"
    local max="${#OUTDATED_DISPLAY[@]}"
    # Work around that OUTDATED_DISPLAY is 1-indexed (length counts from 1)
    max=$((max))
    # Actually count indices: the 1..N range
    local real_max=0
    for k in "${!OUTDATED_DISPLAY[@]}"; do
        [ "$k" -gt "$real_max" ] && real_max="$k"
    done
    max="$real_max"

    input="${input#"${input%%[![:space:]]*}"}"
    input="${input%"${input##*[![:space:]]}"}"
    [ -z "$input" ] && return 1

    if [ "$input" = "all" ]; then
        local result=()
        local n
        for ((n = 1; n <= max; n++)); do
            result+=("${OUTDATED_DISPLAY[$n]}")
        done
        echo "${result[*]}"
        return 0
    fi

    local stripped="${input// /}"
    if ! [[ "$stripped" =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]]; then
        return 2
    fi

    local result=()
    local IFS_BAK="$IFS"
    IFS=','
    for tok in $stripped; do
        IFS="$IFS_BAK"
        if [[ "$tok" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            local lo="${BASH_REMATCH[1]}"
            local hi="${BASH_REMATCH[2]}"
            [ "$lo" -gt "$hi" ] && return 2
            [ "$lo" -lt 1 ] || [ "$hi" -gt "$max" ] && return 3
            local n
            for ((n = lo; n <= hi; n++)); do
                result+=("${OUTDATED_DISPLAY[$n]}")
            done
        else
            local n="$tok"
            [ "$n" -lt 1 ] || [ "$n" -gt "$max" ] && return 3
            result+=("${OUTDATED_DISPLAY[$n]}")
        fi
        IFS=','
    done
    IFS="$IFS_BAK"
    echo "${result[*]}"
    return 0
}

# Run git_update_script.sh in a specific repo; fallback to inline cp if
# the update script isn't present. After running, verify that the files
# actually match the template — the helper may itself be outdated or broken.
do_update_scripts_one() {
    local idx="$1"
    local repo="${STATUS_REPO[$idx]}"

    echo "--- $repo ---"

    local updater="$repo/git_script/git_update_script.sh"
    local target="$repo/git_script"

    if [ -f "$updater" ]; then
        echo "  Running git_script/git_update_script.sh ..."
        (cd "$repo" && bash "$updater")
    else
        echo "  No git_update_script.sh in this repo; copying template files inline."
        mkdir -p "$target"
        cp "$TEMPLATE_DIR"/*.sh "$target/"
        echo "  Copied $(ls "$TEMPLATE_DIR"/*.sh | wc -l) file(s) from template."
    fi

    # Post-update verification: did the helper actually bring things into sync?
    # If files still differ, the helper is probably buggy or outdated itself.
    if is_script_outdated "$repo"; then
        echo "  !! Warning: files still differ from template after update."
        echo "  !! The helper script in this repo may be outdated/broken."
        echo "  !! Consider inspecting $updater or deleting it to trigger the inline fallback."
    fi

    # After update, the repo's git_script/ files have changed on disk, so its
    # status likely shifted to UNSTAGED. Refresh the stored label.
    local new_porcelain
    new_porcelain=$(cd "$repo" && git status --porcelain 2>/dev/null)
    if [ -n "$new_porcelain" ]; then
        local old_label="${STATUS_LABEL[$idx]}"
        if [[ "$old_label" != *UNSTAGED* ]]; then
            if [ "$old_label" = "CLEAN" ]; then
                STATUS_LABEL[$idx]="UNSTAGED"
            else
                STATUS_LABEL[$idx]="${old_label}+UNSTAGED"
            fi
        fi
    fi
    echo
}

# --- 5. Main menu ---
while :; do
    # Count repos matching each action's criteria for the menu hint.
    # Push = has anything local not yet on remote (AHEAD / STAGED / UNSTAGED).
    # Pull = has incoming remote commits (BEHIND).
    n_pushable=0
    n_behind=0
    for i in "${!STATUS_LABEL[@]}"; do
        lbl="${STATUS_LABEL[$i]}"
        if [[ "$lbl" == *AHEAD* || "$lbl" == *STAGED* || "$lbl" == *UNSTAGED* ]]; then
            n_pushable=$((n_pushable + 1))
        fi
        [[ "$lbl" == *BEHIND* ]] && n_behind=$((n_behind + 1))
    done

    echo "What next?"
    echo "  1) Push part or all of the repos  (${n_pushable} with local changes)"
    echo "  2) Pull part or all of the repos  (${n_behind} BEHIND)"
    echo "  3) Sync all: pull every BEHIND, then push every repo with local changes"
    echo "  4) Check/update git_script across repos"
    echo "  5) Quit"
    read -r -p "Choose (1/2/3/4/5): " CHOICE

    case "$CHOICE" in
        1)
            list_candidates "Repos you can push (anything not yet on remote)" \
                "*AHEAD*" "*STAGED*" "*UNSTAGED*"
            picks=$(prompt_selection "push") || { echo; continue; }
            for i in $picks; do
                do_push_one "$i"
            done
            ;;
        2)
            list_candidates "Repos you can pull (BEHIND)" "*BEHIND*"
            picks=$(prompt_selection "pull") || { echo; continue; }
            for i in $picks; do
                do_pull_one "$i"
            done
            ;;
        3)
            # Sync: pull everything that's BEHIND, then push everything that
            # still has local changes. Diverged repos (BEHIND+AHEAD) can't be
            # auto-pulled with --ff-only and are explicitly skipped so we don't
            # leave behind stash orphans.
            echo "=== Sync: phase 1 — pulling BEHIND repos ==="
            for i in "${!STATUS_LABEL[@]}"; do
                lbl="${STATUS_LABEL[$i]}"
                if [[ "$lbl" == *BEHIND* ]]; then
                    # Skip diverged cases — needs manual merge/rebase
                    if [ "${STATUS_AHEAD[$i]:-0}" -gt 0 ]; then
                        echo "--- ${STATUS_REPO[$i]} (${STATUS_BRANCH[$i]}) ---"
                        echo "  Skipped: diverged (behind ${STATUS_BEHIND[$i]}, ahead ${STATUS_AHEAD[$i]})."
                        echo "  Needs manual 'git pull --rebase' or 'git pull --no-rebase'."
                        echo
                        continue
                    fi
                    do_pull_one "$i"
                fi
            done

            echo "=== Sync: phase 2 — pushing repos with local changes ==="
            for i in "${!STATUS_LABEL[@]}"; do
                lbl="${STATUS_LABEL[$i]}"
                if [[ "$lbl" == *AHEAD* || "$lbl" == *STAGED* || "$lbl" == *UNSTAGED* ]]; then
                    do_push_one "$i"
                fi
            done
            echo "=== Sync complete ==="
            ;;
        4)
            # Check whether each repo's git_script/ matches the template.
            # After updating the outdated ones, they'll typically become UNSTAGED
            # (because git_script/*.sh files changed on disk) — the user can then
            # use option 1 or 3 to commit and push those updates.
            refresh_template || { echo; continue; }
            scan_outdated_scripts
            list_uptodate
            list_outdated
            if [ "${#OUTDATED_REPOS[@]}" -eq 0 ]; then
                continue
            fi

            echo "Enter repos to update (by number above), 'all', or empty to cancel." >&2
            read -r -p "> " input
            picks_raw=$(parse_outdated_selection "$input")
            rc=$?
            case "$rc" in
                0) ;;
                1) echo "Cancelled."; continue ;;
                2) echo "Invalid syntax."; continue ;;
                3) echo "Out of range."; continue ;;
            esac

            for i in $picks_raw; do
                do_update_scripts_one "$i"
            done
            echo "Script update complete. Use option 1 or 3 to push the changes."
            ;;
        5|q|Q|"")
            echo "Bye."
            break
            ;;
        *)
            echo "Invalid choice."
            ;;
    esac
    echo
done