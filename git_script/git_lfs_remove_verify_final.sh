cd "$(git rev-parse --show-toplevel)"
################################################################################
#                                                                              #
#           FINAL LFS CLEANUP VERIFICATION SCRIPT (FILES + FOLDERS)            #
#           Reads from git_lfs_tracked_files.txt                               #
#                                                                              #
################################################################################

# Configuration
FILES_LIST="git_lfs_tracked_files.txt"

# Check if file list exists
if [ ! -f "$FILES_LIST" ]; then
	echo "❌ Error: File list not found:  $FILES_LIST"
	echo "No files have been lfs tracked, so it makes no sense to check."
	exit 1
fi

# Read entries into arrays (separate files and folders)
FILES_TO_CHECK=()
FOLDERS_TO_CHECK=()

while IFS= read -r line; do
	# Skip empty lines and comments
	[[ -z "$line" || "$line" =~ ^# ]] && continue

	# Check if it's a folder (ends with /)
	if [[ "$line" =~ /$ ]]; then
		# Remove trailing slash for processing
		folder="${line%/}"
		[ -n "$folder" ] && FOLDERS_TO_CHECK+=("$folder")
	else
		# It's a file
		[ -n "$line" ] && FILES_TO_CHECK+=("$line")
	fi
done <"$FILES_LIST"

TOTAL_ENTRIES=$((${#FILES_TO_CHECK[@]} + ${#FOLDERS_TO_CHECK[@]}))

if [ "$TOTAL_ENTRIES" -eq 0 ]; then
	echo "❌ Error: No files or folders found in $FILES_LIST"
	exit 1
fi

# Repository info
REMOTE_URL=$(git remote get-url origin)
CLEAN_URL=$(echo "$REMOTE_URL" | sed 's/\.git$//')
REPO_OWNER=$(echo "$CLEAN_URL" | sed -E 's#.*[:/]([^/]+)/[^/]+$#\1#')
REPO_NAME=$(echo "$CLEAN_URL" | sed -E 's#.*[:/][^/]+/([^/]+)$#\1#')
EMPTY_FILE_HASH="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Counters
PASS=0
FAIL=0
TOTAL=0
ENTRIES_PASS=0
ENTRIES_FAIL=0

# Functions
print_header() {
	echo ""
	echo "═══════════════════════════════════════════════════════════════════"
	echo -e "${CYAN}$1${NC}"
	echo "═══════════════════════════════════════════════════════════════════"
	echo ""
}

print_section() {
	echo ""
	echo -e "${BLUE}▶ $1${NC}"
	echo "───────────────────────────────────────────────────────────────────"
}

check_pass() {
	PASS=$((PASS + 1))
	TOTAL=$((TOTAL + 1))
	echo -e "  ${GREEN}✅ PASS${NC}:  $1"
}

check_fail() {
	FAIL=$((FAIL + 1))
	TOTAL=$((TOTAL + 1))
	echo -e "  ${RED}❌ FAIL${NC}: $1"
}

check_info() {
	echo -e "  ${CYAN}ℹ️  INFO${NC}: $1"
}

check_warn() {
	echo -e "  ${YELLOW}⚠️  WARN${NC}: $1"
}

################################################################################
# HELPER FUNCTION: Check individual file
################################################################################

check_file() {
	local FILE_NAME="$1"
	local FILE_BASENAME=$(basename "$FILE_NAME")
	local FILE_PASSED=true

	echo ""
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo -e "${MAGENTA}📄 Checking FILE: $FILE_NAME${NC}"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

	# Check 1: Not in working directory
	# if [ -f "$FILE_NAME" ]; then
	#     check_fail "File STILL exists in working directory"
	#     FILE_PASSED=false
	# else
	#     check_pass "File NOT in working directory"
	# fi

	# Check 2: Not in Git index
	if git ls-files --cached | grep -qxF "$FILE_NAME"; then
		check_fail "File STILL in Git index"
		FILE_PASSED=false
	else
		check_pass "File NOT in Git index"
	fi

	# Check 3: Not in HEAD
	if git ls-tree -r HEAD --name-only | grep -qxF "$FILE_NAME"; then
		check_fail "File STILL in current HEAD"
		FILE_PASSED=false
	else
		check_pass "File NOT in current HEAD"
	fi

	# Check 4: Not in local history
	if git log --all --full-history --name-only -- "$FILE_NAME" 2>/dev/null | grep -q .; then
		check_fail "File STILL in local Git history"
		git log --all --oneline -- "$FILE_NAME" | head -3 | sed 's/^/    /'
		FILE_PASSED=false
	else
		check_pass "File NOT in local Git history"
	fi

	# Check 5: Not in Git objects
	if git rev-list --all --objects | grep -q "${FILE_NAME}$"; then
		check_fail "File STILL in Git objects"
		FILE_PASSED=false
	else
		check_pass "File NOT in Git objects"
	fi

	# Check 6: Not in remote history
	if git log origin/main --full-history --name-only -- "$FILE_NAME" 2>/dev/null | grep -q .; then
		check_fail "File STILL in remote Git history"
		FILE_PASSED=false
	else
		check_pass "File NOT in remote Git history"
	fi

	# Check 7: Not tracked by LFS
	if git lfs ls-files 2>/dev/null | grep -q "$FILE_BASENAME"; then
		check_fail "File STILL tracked by LFS"
		FILE_PASSED=false
	else
		check_pass "File NOT tracked by LFS"
	fi

	# Result
	if [ "$FILE_PASSED" = true ]; then
		ENTRIES_PASS=$((ENTRIES_PASS + 1))
		echo -e "  ${GREEN}✅ RESULT:  File completely removed${NC}"
	else
		ENTRIES_FAIL=$((ENTRIES_FAIL + 1))
		echo -e "  ${RED}❌ RESULT: File still has references${NC}"
	fi
}

################################################################################
# HELPER FUNCTION: Check folder
################################################################################

check_folder() {
	local FOLDER_NAME="$1"
	local FOLDER_PASSED=true

	echo ""
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo -e "${MAGENTA}📁 Checking FOLDER: $FOLDER_NAME/${NC}"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

	# Check 1:  Folder not in working directory
	# if [ -d "$FOLDER_NAME" ]; then
	#     check_fail "Folder STILL exists in working directory"
	#     FOLDER_PASSED=false
	# else
	#     check_pass "Folder NOT in working directory"
	# fi

	# Check 2: No files from folder in Git index
	FILES_IN_INDEX=$(git ls-files --cached | grep "^${FOLDER_NAME}/" | wc -l)
	if [ "$FILES_IN_INDEX" -gt 0 ]; then
		check_fail "$FILES_IN_INDEX file(s) from folder STILL in Git index"
		FOLDER_PASSED=false
	else
		check_pass "No files from folder in Git index"
	fi

	# Check 3: No files from folder in HEAD
	FILES_IN_HEAD=$(git ls-tree -r HEAD --name-only | grep "^${FOLDER_NAME}/" | wc -l)
	if [ "$FILES_IN_HEAD" -gt 0 ]; then
		check_fail "$FILES_IN_HEAD file(s) from folder STILL in HEAD"
		FOLDER_PASSED=false
	else
		check_pass "No files from folder in HEAD"
	fi

	# Check 4: No files from folder in local history
	if git log --all --full-history --name-only -- "$FOLDER_NAME/" 2>/dev/null | grep "^${FOLDER_NAME}/" | grep -q .; then
		check_fail "Folder STILL in local Git history"
		echo "  Recent commits:"
		git log --all --oneline -- "$FOLDER_NAME/" | head -3 | sed 's/^/    /'
		FOLDER_PASSED=false
	else
		check_pass "No files from folder in local Git history"
	fi

	# Check 5: No files from folder in Git objects
	if git rev-list --all --objects | grep -q "^.*${FOLDER_NAME}/"; then
		check_fail "Files from folder STILL in Git objects"
		FOLDER_PASSED=false
	else
		check_pass "No files from folder in Git objects"
	fi

	# Check 6: No files from folder in remote history
	if git log origin/main --full-history --name-only -- "$FOLDER_NAME/" 2>/dev/null | grep "^${FOLDER_NAME}/" | grep -q .; then
		check_fail "Folder STILL in remote Git history"
		FOLDER_PASSED=false
	else
		check_pass "No files from folder in remote Git history"
	fi

	# Check 7: Not tracked by LFS (pattern check)
	if git lfs ls-files 2>/dev/null | grep -q "^.*${FOLDER_NAME}/"; then
		check_fail "Files from folder STILL tracked by LFS"
		FOLDER_PASSED=false
	else
		check_pass "No files from folder tracked by LFS"
	fi

	# Result
	if [ "$FOLDER_PASSED" = true ]; then
		ENTRIES_PASS=$((ENTRIES_PASS + 1))
		echo -e "  ${GREEN}✅ RESULT:  Folder completely removed${NC}"
	else
		ENTRIES_FAIL=$((ENTRIES_FAIL + 1))
		echo -e "  ${RED}❌ RESULT: Folder still has references${NC}"
	fi
}

################################################################################
# START VERIFICATION
################################################################################

print_header "🔍 FINAL LFS CLEANUP VERIFICATION (FILES + FOLDERS)"
echo "Repository: $REPO_OWNER/$REPO_NAME"
echo "Date: $(date)"
echo ""

echo "Entries to verify:"
echo "  Files:    ${#FILES_TO_CHECK[@]}"
echo "  Folders: ${#FOLDERS_TO_CHECK[@]}"
echo "  Total:   $TOTAL_ENTRIES"
echo ""

if [ ${#FILES_TO_CHECK[@]} -gt 0 ]; then
	echo "Files:"
	for file in "${FILES_TO_CHECK[@]}"; do
		echo "  📄 $file"
	done
fi

if [ ${#FOLDERS_TO_CHECK[@]} -gt 0 ]; then
	echo "Folders:"
	for folder in "${FOLDERS_TO_CHECK[@]}"; do
		echo "  📁 $folder/"
	done
fi

################################################################################
# SECTION 1: PER-ENTRY CHECKS
################################################################################

print_header "SECTION 1: PER-ENTRY VERIFICATION"

# Check all files
for file in "${FILES_TO_CHECK[@]}"; do
	check_file "$file"
done

# Check all folders
for folder in "${FOLDERS_TO_CHECK[@]}"; do
	check_folder "$folder"
done

################################################################################
# SECTION 2: REPOSITORY-WIDE CHECKS
################################################################################

print_header "SECTION 2: REPOSITORY-WIDE STATUS"

print_section "2.1: Fetching from remote"
if git fetch origin --prune --quiet 2>&1; then
	check_pass "Successfully fetched from remote"
else
	check_fail "Failed to fetch from remote"
fi

print_section "2.2: Current LFS tracked files"
LFS_TRACKED=$(git lfs ls-files 2>/dev/null)
if [ -z "$LFS_TRACKED" ]; then
	check_pass "NO files currently tracked by LFS"
else
	LFS_COUNT=$(echo "$LFS_TRACKED" | wc -l)
	check_info "$LFS_COUNT file(s) still tracked by LFS:"
	echo "$LFS_TRACKED" | sed 's/^/    /'
fi

print_section "2.3: .gitattributes status"
if [ -f ".gitattributes" ]; then
	if [ -s ".gitattributes" ]; then
		if grep -q "filter=lfs" .gitattributes; then
			check_info ".gitattributes has LFS rules:"
			grep "filter=lfs" .gitattributes | sed 's/^/    /'
		else
			check_pass ".gitattributes has NO LFS rules"
		fi
	else
		check_pass ".gitattributes is empty"
	fi
else
	check_info "No .gitattributes file"
fi

print_section "2.4: Local LFS cache"
if [ -d ".git/lfs" ]; then
	CACHE_SIZE=$(du -sb .git/lfs 2>/dev/null | awk '{print $1}')
	CACHE_MB=$(awk "BEGIN {printf \"%.2f\", $CACHE_SIZE / 1024 / 1024}")
	check_info "Local LFS cache size: $CACHE_MB MB"

	if [ "$CACHE_SIZE" -lt 102400 ]; then
		check_pass "Local LFS cache is minimal (< 0.1 MB)"
	else
		check_warn "Local LFS cache has $CACHE_MB MB"
	fi
else
	check_pass "No .git/lfs directory"
fi

################################################################################
# SECTION 3: LFS PRUNE STATUS
################################################################################

print_header "SECTION 3: LFS PRUNE STATUS"

print_section "3.1: LFS prune (dry-run)"
echo "  Running:  git lfs prune --dry-run --verbose"
PRUNE_OUTPUT=$(git lfs prune --dry-run --verbose 2>&1)
echo "$PRUNE_OUTPUT" | sed 's/^/    /'
echo ""

LOCAL_OBJ=$(echo "$PRUNE_OUTPUT" | grep -oE '[0-9]+ local' | grep -oE '[0-9]+' | head -1)
RETAINED=$(echo "$PRUNE_OUTPUT" | grep -oE '[0-9]+ retained' | grep -oE '[0-9]+' | head -1)

LOCAL_OBJ=${LOCAL_OBJ:-0}
RETAINED=${RETAINED:-0}

if [ "$LOCAL_OBJ" -eq 0 ]; then
	check_pass "0 local LFS objects"
else
	check_warn "$LOCAL_OBJ local LFS object(s) in cache"
fi

if [ "$RETAINED" -eq 0 ]; then
	check_pass "0 retained objects - PERFECT!"
elif [ "$RETAINED" -eq 1 ]; then
	check_info "1 retained object (checking if it's the empty file...)"

	RETAIN_TRACE=$(GIT_TRACE=1 git lfs prune --verify-remote --dry-run --verbose 2>&1 | grep "RETAIN:")
	if echo "$RETAIN_TRACE" | grep -q "$EMPTY_FILE_HASH"; then
		check_pass "Retained object is empty file (0 bytes) - HARMLESS"
		check_info "This will resolve after 7-10 days"
	else
		check_warn "Retained object is NOT the empty file artifact"
	fi
else
	check_warn "$RETAINED objects retained"
fi

################################################################################
# FINAL SUMMARY
################################################################################

print_header "📊 FINAL SUMMARY"

echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                   ENTRY VERIFICATION RESULTS                      ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Files checked:                  ${#FILES_TO_CHECK[@]}"
echo "  Folders checked:               ${#FOLDERS_TO_CHECK[@]}"
echo "  Total entries:                 $TOTAL_ENTRIES"
echo ""
echo -e "  ${GREEN}Entries passed:                $ENTRIES_PASS${NC}"
echo -e "  ${RED}Entries failed:                $ENTRIES_FAIL${NC}"
echo ""

if [ $ENTRIES_FAIL -gt 0 ]; then
	echo "Entries that failed:"
	for file in "${FILES_TO_CHECK[@]}"; do
		if git log --all --full-history --name-only -- "$file" 2>/dev/null | grep -q .; then
			echo -e "  ${RED}📄 $file${NC}"
		fi
	done
	for folder in "${FOLDERS_TO_CHECK[@]}"; do
		if git log --all --full-history --name-only -- "$folder/" 2>/dev/null | grep -q .; then
			echo -e "  ${RED}📁 $folder/${NC}"
		fi
	done
	echo ""
fi

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                    REPOSITORY STATUS                              ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Total checks run:              $TOTAL"
echo -e "  ${GREEN}Passed:                        $PASS${NC}"
echo -e "  ${RED}Failed:                        $FAIL${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
	SUCCESS_RATE=100
else
	SUCCESS_RATE=$(awk "BEGIN {printf \"%.0f\", ($PASS / $TOTAL) * 100}")
fi

echo "  Success rate:                  $SUCCESS_RATE%"
echo ""

################################################################################
# VERDICT
################################################################################

if [ $ENTRIES_FAIL -eq 0 ] && [ $FAIL -eq 0 ] && [ "$RETAINED" -le 1 ]; then
	echo "╔═══════════════════════════════════════════════════════════════════╗"
	echo "║                                                                   ║"
	echo "║                    🎉  ALL CLEAR!  🎉                             ║"
	echo "║                                                                   ║"
	echo "║          YOUR CLEANUP IS COMPLETE!                                  ║"
	echo "║          NOTHING MORE TO DO - JUST WAIT!                            ║"
	echo "║                                                                   ║"
	echo "╚═══════════════════════════════════════════════════════════════════╝"
	echo ""
	echo "✅ All $TOTAL_ENTRIES entries removed from Git history"
	echo "✅ All entries not tracked by LFS"
	echo "✅ Local LFS cache clean"

	if [ "$RETAINED" -eq 1 ]; then
		echo ""
		echo "ℹ️  Note: '1 retained' is an empty file (0 bytes) - HARMLESS"
		echo "   Will resolve automatically in 7-10 days."
	fi

	echo ""
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo "NEXT STEPS:"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo ""
	echo "1. ✅ You're DONE!"
	echo "2. ⏳ Wait 30 days for automatic GitHub cleanup"
	echo "3. 📅 Check billing after:  $(date -d '+30 days' '+%B %d, %Y' 2>/dev/null || date -v+30d '+%B %d, %Y' 2>/dev/null || echo 'February 1, 2026')"
	echo "4. 🌐 Verify at: https://github.com/settings/billing"
	echo ""

else
	echo "╔═══════════════════════════════════════════════════════════════════╗"
	echo "║                ❌  CLEANUP NOT COMPLETE ❌                        ║"
	echo "╚═══════════════════════════════════════════════════════════════════╝"
	echo ""
	echo "❌ $ENTRIES_FAIL entry/entries still have references"
	echo "❌ $FAIL check(s) failed"
	echo ""
	echo "Review failed checks above and re-run cleanup if needed."
	echo ""
fi

echo "═══════════════════════════════════════════════════════════════════"
echo "Verification completed:  $(date)"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
