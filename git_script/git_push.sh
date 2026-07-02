cd "$(git rev-parse --show-toplevel)"

# uncomment if it is in linux, and need to convert dos to unix
# sed -i 's/\r$//' git_script/git_push. sh

# Stage all changes including deletions
git add .
# First, check for large files and warn before staging anything
# First, check for large files and warn before staging anything
LARGE_FILES=$(git ls-files --cached --others --exclude-standard | while read -r file; do
    if [ -f "$file" ] && [ "$(stat -c%s "$file" 2>/dev/null)" -gt 104857600 ]; then
        echo "$file"
    fi
done)

if [ -n "$LARGE_FILES" ]; then
	echo "WARNING: The following files are 100MB or larger and will NOT be committed:"
	echo "$LARGE_FILES"
	echo "$LARGE_FILES" | while read -r file; do
		# Remove leading . / if present for git commands
		clean_file="${file#./}"
		git restore --staged "$clean_file" 2>/dev/null || true
	done
	echo "Large files have been unstaged.  Use git_lfs_push. sh for large files."
fi

# Commit changes with a custom message if provided, otherwise use a default message
if [ -n "$1" ]; then
	git commit -m "$1"
else
	git commit -m "update"
fi

branch_name=$(git rev-parse --abbrev-ref HEAD)
git push origin "$branch_name"
