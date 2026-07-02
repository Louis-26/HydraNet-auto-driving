cd "$(git rev-parse --show-toplevel)"
################################################################################
#                                                                              #
#        VERIFY ALL LFS FILES ARE ORPHANED (NOT REFERENCED)                    #
#        ENHANCED VERSION - WITH DETAILED SUMMARY                              #
#                                                                              #
################################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Known empty file hash
EMPTY_FILE_HASH="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

# Counters for summary
ORPHANED_COUNT=0
REFERENCED_COUNT=0
EMPTY_FILE_COUNT=0
LARGE_FILE_COUNT=0

print_header() {
	echo ""
	echo "═══════════════════════════════════════════════════════════════════"
	echo -e "${CYAN}${BOLD}$1${NC}"
	echo "═══════════════════════════════════════════════════════════════════"
	echo ""
}

print_section() {
	echo ""
	echo -e "${BLUE}▶ $1${NC}"
	echo "───────────────────────────────────────────────────────────────────"
}

print_pass() {
	echo -e "  ${GREEN}✅ $1${NC}"
}

print_fail() {
	echo -e "  ${RED}❌ $1${NC}"
}

print_warn() {
	echo -e "  ${YELLOW}⚠️  $1${NC}"
}

print_info() {
	echo -e "  ${CYAN}ℹ️  $1${NC}"
}

################################################################################
# MAIN VERIFICATION
################################################################################

clear
print_header "🔍 ORPHANED LFS FILES VERIFICATION (WITH DETAILED SUMMARY)"
echo "Repository: $(git config --get remote.origin.url 2>/dev/null | sed 's/.*github.com[:/]\(.*\)\.git/\1/' || echo 'Unknown')"
echo "Date: $(date)"
echo ""

################################################################################
# STEP 1: DISCOVER ALL LFS OBJECTS THAT EVER EXISTED
################################################################################

print_header "STEP 1: DISCOVERING ALL LFS OBJECTS IN REPOSITORY HISTORY"

print_section "1.1: Finding all LFS pointer files in entire Git history"

# Create temp file for results
TEMP_LFS_LIST=$(mktemp)
TEMP_CURRENT_LIST=$(mktemp)

echo "  Scanning all commits for LFS pointers..."
echo "  (This may take a moment for large repositories)"
echo ""

# Search for LFS pointers in all commits
git log --all --full-history -p --no-ext-diff --no-textconv -S "version https://git-lfs.github.com/spec/v1" \
	--format="%H" -- 2>/dev/null | sort -u >"$TEMP_LFS_LIST"

COMMITS_WITH_LFS=$(cat "$TEMP_LFS_LIST" | wc -l)

if [ "$COMMITS_WITH_LFS" -eq 0 ]; then
	print_pass "No LFS pointers found in any commit"
	print_info "Either:  (1) LFS was never used, or (2) All pointers removed by BFG"
else
	print_info "Found $COMMITS_WITH_LFS commit(s) that had/have LFS pointers"
fi

print_section "1.2: Extracting LFS object IDs from history"

# Find all unique LFS OIDs in entire history
LFS_OIDS_FILE=$(mktemp)

# Method 1: Check all commits for LFS pointer content
git log --all --full-history -p --no-ext-diff --no-textconv -S "oid sha256:" --format="" |
	grep "^+.*oid sha256:" |
	sed 's/.*oid sha256://g' | sed 's/[^a-f0-9]//g' |
	grep -E '^[a-f0-9]{64}$' |
	sort -u >"$LFS_OIDS_FILE"

# Method 2: Also check current LFS tracked files
git lfs ls-files --all --long 2>/dev/null | awk '{print $1}' | grep -E '^[a-f0-9]+$' >>"$LFS_OIDS_FILE"

# Remove duplicates
sort -u "$LFS_OIDS_FILE" -o "$LFS_OIDS_FILE"

TOTAL_LFS_OIDS=$(cat "$LFS_OIDS_FILE" | wc -l)

if [ "$TOTAL_LFS_OIDS" -eq 0 ]; then
	print_pass "No LFS objects found in Git history"
	print_info "Repository appears to have no LFS files in Git commits"
else
	print_info "Found $TOTAL_LFS_OIDS unique LFS object(s) in Git history"
	echo ""
	echo "  LFS Object IDs (first 12 characters):"
	cat "$LFS_OIDS_FILE" | cut -c1-12 | sed 's/^/    /'
fi

echo ""

################################################################################
# STEP 2: CHECK EACH LFS OBJECT FOR REFERENCES
################################################################################

print_header "STEP 2: CHECKING IF LFS OBJECTS ARE ORPHANED"

if [ "$TOTAL_LFS_OIDS" -eq 0 ]; then
	print_info "No LFS objects to check in Git history"
else
	echo "Checking each LFS object for references..."
	echo ""

	while IFS= read -r OID; do
		if [ -z "$OID" ]; then
			continue
		fi

		SHORT_OID=$(echo "$OID" | cut -c1-12)
		echo "─────────────────────────────────────────────────────────────────"
		echo -e "${MAGENTA}Object:    $SHORT_OID... ${NC}"

		# Check if it's the empty file
		IS_EMPTY_FILE=false
		if [ "$OID" = "$EMPTY_FILE_HASH" ]; then
			echo -e "${CYAN}  (This is the empty file - 0 bytes)${NC}"
			IS_EMPTY_FILE=true
		fi

		echo "─────────────────────────────────────────────────────────────────"

		IS_ORPHANED=true

		# Check 1: In current working directory
		echo -n "  Checking working directory...  "
		if git lfs ls-files 2>/dev/null | grep -q "$SHORT_OID"; then
			echo -e "${RED}FOUND${NC}"
			print_fail "Referenced in working directory"
			IS_ORPHANED=false
		else
			echo -e "${GREEN}not found${NC}"
		fi

		# Check 2: In staging area
		echo -n "  Checking staging area... "
		if git lfs ls-files --cached 2>/dev/null | grep -q "$SHORT_OID"; then
			echo -e "${RED}FOUND${NC}"
			print_fail "Referenced in staging area"
			IS_ORPHANED=false
		else
			echo -e "${GREEN}not found${NC}"
		fi

		# Check 3: In current HEAD
		echo -n "  Checking HEAD commit... "
		if git show HEAD 2>/dev/null | grep -q "$OID"; then
			echo -e "${RED}FOUND${NC}"
			print_fail "Referenced by HEAD"
			IS_ORPHANED=false
		else
			echo -e "${GREEN}not found${NC}"
		fi

		# Check 4: In any local branch
		echo -n "  Checking local branches... "
		FOUND_IN_BRANCH=false
		for branch in $(git branch --format='%(refname:short)'); do
			if git show "$branch" 2>/dev/null | grep -q "$OID"; then
				echo -e "${RED}FOUND in $branch${NC}"
				print_fail "Referenced by branch:   $branch"
				IS_ORPHANED=false
				FOUND_IN_BRANCH=true
				break
			fi
		done
		if [ "$FOUND_IN_BRANCH" = false ]; then
			echo -e "${GREEN}not found${NC}"
		fi

		# Check 5: In any remote branch
		echo -n "  Checking remote branches... "
		FOUND_IN_REMOTE=false
		for branch in $(git branch -r --format='%(refname: short)'); do
			if git show "$branch" 2>/dev/null | grep -q "$OID"; then
				echo -e "${RED}FOUND in $branch${NC}"
				print_fail "Referenced by remote branch:  $branch"
				IS_ORPHANED=false
				FOUND_IN_REMOTE=true
				break
			fi
		done
		if [ "$FOUND_IN_REMOTE" = false ]; then
			echo -e "${GREEN}not found${NC}"
		fi

		# Check 6: In any tag
		echo -n "  Checking tags... "
		TAGS=$(git tag)
		if [ -n "$TAGS" ]; then
			FOUND_IN_TAG=false
			for tag in $TAGS; do
				if git show "$tag" 2>/dev/null | grep -q "$OID"; then
					echo -e "${RED}FOUND in $tag${NC}"
					print_fail "Referenced by tag: $tag"
					IS_ORPHANED=false
					FOUND_IN_TAG=true
					break
				fi
			done
			if [ "$FOUND_IN_TAG" = false ]; then
				echo -e "${GREEN}not found${NC}"
			fi
		else
			echo -e "${CYAN}no tags${NC}"
		fi

		# Check 7: In recent commits (last 30 days)
		echo -n "  Checking recent commits (30 days)... "
		RECENT_COMMITS=$(git log --all --since="30 days ago" --format="%H")
		FOUND_IN_RECENT=false
		for commit in $RECENT_COMMITS; do
			if git show "$commit" 2>/dev/null | grep -q "$OID"; then
				echo -e "${RED}FOUND${NC}"
				print_fail "Referenced by recent commit: $(echo $commit | cut -c1-8)"
				IS_ORPHANED=false
				FOUND_IN_RECENT=true
				break
			fi
		done
		if [ "$FOUND_IN_RECENT" = false ]; then
			echo -e "${GREEN}not found${NC}"
		fi

		# Check 8: In any commit in entire history
		echo -n "  Checking entire Git history... "
		if git log --all --full-history -p --no-ext-diff -S "$OID" --format="%H" 2>/dev/null | grep -q .; then
			echo -e "${YELLOW}FOUND (in old history)${NC}"

			# Get the commits
			COMMITS_WITH_OID=$(git log --all --full-history -p --no-ext-diff -S "$OID" --format="%h %s" 2>/dev/null | head -3)
			echo ""
			echo "    Commits that referenced this object:"
			echo "$COMMITS_WITH_OID" | sed 's/^/      /'

			# Check if these commits are reachable from current branches
			echo -n "    Checking if reachable from current branches... "
			REACHABLE=false
			for commit in $(git log --all --full-history -p --no-ext-diff -S "$OID" --format="%H" 2>/dev/null | head -1); do
				if git branch --contains "$commit" 2>/dev/null | grep -q .; then
					echo -e "${RED}YES${NC}"
					print_fail "Old commit still reachable from a branch"
					IS_ORPHANED=false
					REACHABLE=true
					break
				fi
			done
			if [ "$REACHABLE" = false ]; then
				echo -e "${GREEN}NO${NC}"
				print_pass "Old commits not reachable (orphaned in history)"
			fi
		else
			echo -e "${GREEN}not found${NC}"
		fi

		# Check 9: In local LFS cache
		echo -n "  Checking local LFS cache... "
		if [ -f ". git/lfs/objects/${OID:0:2}/${OID:2:2}/$OID" ]; then
			echo -e "${YELLOW}EXISTS${NC}"
			FILE_SIZE=$(du -h ".git/lfs/objects/${OID:0:2}/${OID:2:2}/$OID" 2>/dev/null | cut -f1)
			print_warn "File exists in local cache ($FILE_SIZE)"
			echo "      This is OK - will be pruned after verification"
		else
			echo -e "${GREEN}not cached${NC}"
		fi

		# Final verdict for this object
		echo ""
		if [ "$IS_ORPHANED" = true ]; then
			ORPHANED_COUNT=$((ORPHANED_COUNT + 1))

			if [ "$IS_EMPTY_FILE" = true ]; then
				EMPTY_FILE_COUNT=$((EMPTY_FILE_COUNT + 1))
				echo -e "  ${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
				echo -e "  ${GREEN}║  ✅  ORPHANED - Empty file (0 bytes)                  ║${NC}"
				echo -e "  ${GREEN}║      This is harmless and will be deleted             ║${NC}"
				echo -e "  ${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
			else
				LARGE_FILE_COUNT=$((LARGE_FILE_COUNT + 1))
				echo -e "  ${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
				echo -e "  ${GREEN}║  ✅  ORPHANED - Will be deleted after 30 days         ║${NC}"
				echo -e "  ${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
			fi
		else
			REFERENCED_COUNT=$((REFERENCED_COUNT + 1))
			echo -e "  ${RED}╔════════════════════════════════════════════════════════╗${NC}"
			echo -e "  ${RED}║  ❌  STILL REFERENCED - NOT orphaned                   ║${NC}"
			echo -e "  ${RED}╚════════════════════════════════════════════════════════╝${NC}"
		fi

		echo ""

	done <"$LFS_OIDS_FILE"
fi

################################################################################
# STEP 3: CHECK WHAT GIT LFS SEES (ENHANCED)
################################################################################

print_header "STEP 3: WHAT GIT LFS PRUNE SEES"

print_section "3.1: Current LFS tracked files"
LFS_CURRENT=$(git lfs ls-files --all 2>/dev/null)
CURRENTLY_TRACKED_COUNT=$(echo "$LFS_CURRENT" | grep -c . || echo 0)

if [ -z "$LFS_CURRENT" ] || [ "$CURRENTLY_TRACKED_COUNT" -eq 0 ]; then
	print_pass "No files currently tracked by LFS"
else
	print_warn "$CURRENTLY_TRACKED_COUNT file(s) currently tracked by LFS:"
	echo "$LFS_CURRENT" | sed 's/^/    /'
fi

print_section "3.2: LFS prune status (local)"
echo "  Running:   git lfs prune --dry-run --verbose"
echo ""
PRUNE_LOCAL=$(git lfs prune --dry-run --verbose 2>&1)
echo "$PRUNE_LOCAL" | sed 's/^/    /'
echo ""

LOCAL_OBJ=$(echo "$PRUNE_LOCAL" | grep -oP '\d+(?= local)' | head -1)
if [ "$LOCAL_OBJ" = "0" ]; then
	print_pass "0 local LFS objects in cache"
else
	print_info "$LOCAL_OBJ local LFS object(s) in cache"
fi

print_section "3.3: LFS prune with remote verification (ENHANCED)"
echo "  Running: git lfs prune --verify-remote --dry-run --verbose"
echo "  Extracting retained object details..."
echo ""

# Create temp file for full trace
TRACE_FILE=$(mktemp)

# Run with full trace
GIT_TRACE=1 git lfs prune --verify-remote --dry-run --verbose 2>&1 | tee "$TRACE_FILE" | grep -v "^[0-9:]" | sed 's/^/    /'

echo ""

# Extract retention info
RETAINED=$(grep -oP '\d+(?= retained)' "$TRACE_FILE" | head -1)
RETAINED_EMPTY=0
RETAINED_LARGE=0

# Extract RETAIN lines
RETAIN_LINES=$(grep "RETAIN:" "$TRACE_FILE")

if [ "$RETAINED" = "0" ]; then
	print_pass "0 retained objects - PERFECT!"

elif [ "$RETAINED" = "1" ]; then
	print_info "1 retained object detected on GitHub's LFS server"
	echo ""

	# Try to extract the OID
	if [ -n "$RETAIN_LINES" ]; then
		echo "  Retained object details:"
		echo "$RETAIN_LINES" | sed 's/^/    /'
		echo ""

		# Extract the OID from RETAIN line
		RETAINED_OID=$(echo "$RETAIN_LINES" | grep -oP 'RETAIN: \K[a-f0-9]{64}' | head -1)

		if [ -n "$RETAINED_OID" ]; then
			echo "  Retained Object ID:"
			echo "    $(echo $RETAINED_OID | cut -c1-12)...  (showing first 12 chars)"
			echo "    Full:  $RETAINED_OID"
			echo ""

			# Check if it's the empty file
			if [ "$RETAINED_OID" = "$EMPTY_FILE_HASH" ]; then
				RETAINED_EMPTY=1
				echo -e "  ${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
				echo -e "  ${GREEN}║                                                           ║${NC}"
				echo -e "  ${GREEN}║  ✅  THIS IS THE EMPTY FILE (0 BYTES)                     ║${NC}"
				echo -e "  ${GREEN}║                                                           ║${NC}"
				echo -e "  ${GREEN}║  Status:    Harmless artifact from cleanup                 ║${NC}"
				echo -e "  ${GREEN}║  Size:     0 bytes (no storage used)                      ║${NC}"
				echo -e "  ${GREEN}║  Impact:   None                                           ║${NC}"
				echo -e "  ${GREEN}║  Action:   None needed - will auto-expire in 7-10 days    ║${NC}"
				echo -e "  ${GREEN}║                                                           ║${NC}"
				echo -e "  ${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"

				# Verify the hash
				echo ""
				echo "  Verification:"
				COMPUTED_HASH=$(echo -n "" | sha256sum | awk '{print $1}')
				echo "    Computed empty file hash: $COMPUTED_HASH"
				echo "    Retained object hash:      $RETAINED_OID"
				if [ "$COMPUTED_HASH" = "$RETAINED_OID" ]; then
					echo -e "    ${GREEN}✅ MATCH - Confirmed empty file${NC}"
				else
					echo -e "    ${RED}❌ NO MATCH${NC}"
				fi
			else
				RETAINED_LARGE=1
				echo -e "  ${YELLOW}╔═══════════════════════════════════════════════════════════╗${NC}"
				echo -e "  ${YELLOW}║                                                           ║${NC}"
				echo -e "  ${YELLOW}║  ⚠️  THIS IS NOT THE EMPTY FILE                           ║${NC}"
				echo -e "  ${YELLOW}║                                                           ║${NC}"
				echo -e "  ${YELLOW}║  This is likely your large file or another LFS object     ║${NC}"
				echo -e "  ${YELLOW}║  Status:  Orphaned on GitHub's LFS server                 ║${NC}"
				echo -e "  ${YELLOW}║  Action:  Will be deleted after 30 days                   ║${NC}"
				echo -e "  ${YELLOW}║                                                           ║${NC}"
				echo -e "  ${YELLOW}╚═══════════════════════════════════════════════════════════╝${NC}"
			fi
		else
			print_warn "Could not extract OID from trace"
			print_info "The retained object is on GitHub's LFS server"
		fi
	else
		print_info "Retention determined by remote (GitHub's LFS server)"
		print_info "The object is not in your local Git history"
	fi

else
	print_warn "$RETAINED objects retained on GitHub's LFS server"

	if [ -n "$RETAIN_LINES" ]; then
		echo ""
		echo "  Retained objects:"
		while IFS= read -r line; do
			RETAINED_OID=$(echo "$line" | grep -oP 'RETAIN:  \K[a-f0-9]{64}')
			if [ "$RETAINED_OID" = "$EMPTY_FILE_HASH" ]; then
				RETAINED_EMPTY=$((RETAINED_EMPTY + 1))
				echo "    $(echo $RETAINED_OID | cut -c1-12)... (empty file, 0 bytes)"
			else
				RETAINED_LARGE=$((RETAINED_LARGE + 1))
				echo "    $(echo $RETAINED_OID | cut -c1-12)... (large file)"
			fi
		done <<<"$RETAIN_LINES"
	fi
fi

################################################################################
# STEP 4: COMPREHENSIVE SUMMARY
################################################################################

print_header "📊 COMPREHENSIVE SUMMARY"

echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                   OBJECT COUNT SUMMARY                            ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# Git Repository Objects
echo "┌─────────────────────────────────────────────────────────────────┐"
echo "│ 📁 GIT REPOSITORY (Your Local + Remote Git Commits)            │"
echo "└─────────────────────────────────────────────────────────────────┘"
echo ""
echo "  Total LFS objects found in Git history:     $TOTAL_LFS_OIDS"

if [ "$TOTAL_LFS_OIDS" -gt 0 ]; then
	echo ""
	echo -e "  ${GREEN}├─ Orphaned objects:                        $ORPHANED_COUNT${NC}"
	if [ "$EMPTY_FILE_COUNT" -gt 0 ]; then
		echo -e "  ${GREEN}│  └─ Empty files (0 bytes):                $EMPTY_FILE_COUNT${NC}"
	fi
	if [ "$LARGE_FILE_COUNT" -gt 0 ]; then
		echo -e "  ${GREEN}│  └─ Large files:                           $LARGE_FILE_COUNT${NC}"
	fi
	echo ""

	if [ "$REFERENCED_COUNT" -gt 0 ]; then
		echo -e "  ${RED}└─ Still referenced (NOT orphaned):        $REFERENCED_COUNT${NC}"
	fi
else
	echo "  (No LFS objects in Git commits - completely clean)"
fi

echo ""
echo "┌─────────────────────────────────────────────────────────────────┐"
echo "│ ☁️  GITHUB LFS SERVER (Separate Storage)                       │"
echo "└─────────────────────────────────────────────────────────────────┘"
echo ""
echo "  Objects retained on GitHub's LFS server:   ${RETAINED:-0}"

if [ "${RETAINED:-0}" -gt 0 ]; then
	if [ "$RETAINED_EMPTY" -gt 0 ]; then
		echo -e "  ${GREEN}├─ Empty file artifacts (0 bytes):         $RETAINED_EMPTY${NC}"
	fi
	if [ "$RETAINED_LARGE" -gt 0 ]; then
		echo -e "  ${YELLOW}└─ Large files (orphaned):                 $RETAINED_LARGE${NC}"
	fi
fi

echo ""
echo "┌─────────────────────────────────────────────────────────────────┐"
echo "│ 💾 LOCAL STATUS                                                 │"
echo "└─────────────────────────────────────────────────────────────────┘"
echo ""
echo "  Files currently tracked by LFS:            $CURRENTLY_TRACKED_COUNT"
echo "  Objects in local LFS cache:                ${LOCAL_OBJ:-0}"

echo ""
echo "═══════════════════════════════════════════════════════════════════"

################################################################################
# VERDICT
################################################################################

print_header "🎯 FINAL VERDICT"

TOTAL_ORPHANED=$((ORPHANED_COUNT + RETAINED_EMPTY + RETAINED_LARGE))
TOTAL_OBJECTS=$((TOTAL_LFS_OIDS + RETAINED))

if [ "$TOTAL_LFS_OIDS" -eq 0 ] && [ "${RETAINED:-0}" -eq 0 ]; then
	echo ""
	echo "╔═══════════════════════════════════════════════════════════════╗"
	echo "║                                                               ║"
	echo "║         🎉🎉🎉  PERFECT!   COMPLETELY CLEAN!   🎉🎉🎉              ║"
	echo "║                                                               ║"
	echo "╚═══════════════════════════════════════════════════════════════╝"
	echo ""
	print_pass "No LFS objects in Git repository"
	print_pass "No objects retained on GitHub"
	print_pass "Cleanup 100% complete!"

elif [ "$TOTAL_LFS_OIDS" -eq 0 ] && [ "${RETAINED:-0}" -eq 1 ] && [ "$RETAINED_EMPTY" -eq 1 ]; then
	echo ""
	echo "╔═══════════════════════════════════════════════════════════════╗"
	echo "║                                                               ║"
	echo "║         🎉  ALL CLEAR!  (WITH EMPTY FILE ARTIFACT) 🎉          ║"
	echo "║                                                               ║"
	echo "╚═══════════════════════════════════════════════════════════════╝"
	echo ""
	print_pass "Git repository:  0 LFS objects (completely clean)"
	print_info "GitHub LFS server: 1 empty file artifact (0 bytes, harmless)"
	print_pass "All large files successfully removed!"
	echo ""
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo "Summary:"
	echo "  • Your cleanup is COMPLETE ✅"
	echo "  • Empty file artifact will auto-expire in 7-10 days"
	echo "  • Takes 0 bytes of storage"
	echo "  • No action needed from you"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

elif [ "$TOTAL_LFS_OIDS" -eq 0 ] && [ "${RETAINED:-0}" -gt 0 ]; then
	echo ""
	echo "╔═══════════════════════════════════════════════════════════════╗"
	echo "║                                                               ║"
	echo "║         ✅  CLEANUP COMPLETE - OBJECTS ON GITHUB  ✅           ║"
	echo "║                                                               ║"
	echo "╚═══════════════════════════════════════════════════════════════╝"
	echo ""
	print_pass "Git repository: 0 LFS objects (completely clean)"
	print_info "GitHub LFS server: ${RETAINED} object(s) orphaned"

	if [ "$RETAINED_EMPTY" -gt 0 ]; then
		echo ""
		echo "  └─ $RETAINED_EMPTY empty file(s) - 0 bytes, harmless"
	fi
	if [ "$RETAINED_LARGE" -gt 0 ]; then
		echo "  └─ $RETAINED_LARGE large file(s) - will be freed after 30 days"
	fi

	echo ""
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo "NEXT STEPS:"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo ""
	echo "1. ✅ You're DONE - all work complete"
	echo "2. ⏳ Wait for automatic GitHub cleanup:"
	if [ "$RETAINED_EMPTY" -gt 0 ]; then
		echo "     • 7-10 days:   Empty file(s) expire"
	fi
	if [ "$RETAINED_LARGE" -gt 0 ]; then
		echo "     • 30 days:    Large file(s) deleted, storage freed"
	fi
	echo "3. 📅 Check after:  $(date -d '+30 days' '+%B %d, %Y' 2>/dev/null || date -v+30d '+%B %d, %Y' 2>/dev/null || echo 'January 30, 2026')"
	echo "4. 🌐 Verify at:    https://github.com/settings/billing"
	echo ""
	echo "OR:  Contact GitHub Support for immediate cleanup"
	echo "    https://support.github.com/contact"
	echo ""

elif [ "$ORPHANED_COUNT" -eq "$TOTAL_LFS_OIDS" ] && [ "$TOTAL_LFS_OIDS" -gt 0 ]; then
	echo ""
	echo "╔═══════════════════════════════════════════════════════════════╗"
	echo "║                                                               ║"
	echo "║         🎉  ALL LFS FILES ARE ORPHANED! 🎉                    ║"
	echo "║                                                               ║"
	echo "╚═══════════════════════════════════════════════════════════════╝"
	echo ""
	print_pass "ALL $ORPHANED_COUNT LFS object(s) in Git are orphaned"
	print_pass "NO references found in current repository"

	if [ "$EMPTY_FILE_COUNT" -gt 0 ]; then
		echo "  ├─ $EMPTY_FILE_COUNT empty file(s) - 0 bytes"
	fi
	if [ "$LARGE_FILE_COUNT" -gt 0 ]; then
		echo "  └─ $LARGE_FILE_COUNT large file(s)"
	fi

	echo ""
	print_pass "GitHub will automatically delete these after 30 days"
	echo ""
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo "Wait 30 days for automatic cleanup"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

elif [ "$REFERENCED_COUNT" -gt 0 ]; then
	echo ""
	echo "╔═══════════════════════════════════════════════════════════════╗"
	echo "║                                                               ║"
	echo "║         ❌  SOME FILES STILL REFERENCED ❌                    ║"
	echo "║                                                               ║"
	echo "╚═══════════════════════════════════════════════════════════════╝"
	echo ""
	print_fail "$REFERENCED_COUNT LFS object(s) are STILL REFERENCED"

	if [ "$ORPHANED_COUNT" -gt 0 ]; then
		print_pass "$ORPHANED_COUNT LFS object(s) are orphaned"
	fi

	echo ""
	echo "Review the detailed checks above to see where files are referenced."
	echo ""
	echo "Actions:"
	echo "  • Check failed checks above for reference locations"
	echo "  • Re-run BFG Repo-Cleaner if needed"
	echo "  • Ensure all branches are cleaned"
	echo "  • Check and clean tags if necessary"
	echo ""
fi

echo "═══════════════════════════════════════════════════════════════════"
echo "Verification completed:  $(date)"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Cleanup
rm -f "$TEMP_LFS_LIST" "$TEMP_CURRENT_LIST" "$LFS_OIDS_FILE" "$TRACE_FILE"
