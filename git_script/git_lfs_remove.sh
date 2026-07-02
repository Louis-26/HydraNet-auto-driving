cd "$(git rev-parse --show-toplevel)"

# use git lfs to remove large files or folders already pushed to remote repository, for main branch

# Uncomment if you need to convert DOS to Unix line endings
# sed -i 's/\r$//' git_script/git_lfs_remove.sh

# notice:
# 1. execute it in the same level where the .git folder is located, i.e., git_script/git_lfs_remove.sh
# construct mirror clone of the repository to completely remove LFS files from history
# 2. you need to push all branches and tags

# Function to update .gitattributes when files in a folder are partially or fully removed
update_folder_in_gitattributes() {
	local folder="$1"

	# List of all files in the folder that are still tracked by LFS
	remaining_files=($(git lfs ls-files | grep -P "^.*$folder/.*$" | awk '{print $3}'))

	# Check if there are still files in the folder being tracked by LFS
	if [ "${#remaining_files[@]}" -eq 0 ]; then
		# No files from the folder are tracked, remove the folder entry from .gitattributes
		echo "Removing folder from .gitattributes: $folder"
		sed -i "/$folder\/\*\*/d" .gitattributes
	else
		# Some files in the folder are still tracked, update .gitattributes to list them explicitly
		echo "Updating folder in .gitattributes: $folder"

		# Remove the old folder pattern (folder/**)
		sed -i "/$folder\/\*\*/d" .gitattributes

		# Add each of the remaining files explicitly
		for file in "${remaining_files[@]}"; do
			# Add each file to .gitattributes
			echo "$file filter=lfs diff=lfs merge=lfs -text" >>.gitattributes
		done
	fi
}

# Find the root directory containing .git
find_git_root() {
	local dir="$1"
	while [ "$dir" != "/" ]; do
		if [ -d "$dir/.git" ]; then
			echo "$dir"
			return 0
		fi
		dir=$(dirname "$dir")
	done
	return 1
}

# Ensure we start from the directory where .git is located
CURRENT_DIR=$(find_git_root "$(pwd)")
if [ -z "$CURRENT_DIR" ]; then
	echo "❌ Error: Not in a Git repository!"
	exit 1
fi

cd "$CURRENT_DIR"

echo "Select the type of files to untrack and remove:"
echo "1. Type 'all' to untrack and remove all LFS files."
echo "2. Type 'ext' to untrack and remove files with certain extensions (e.g., '.mp4')."
echo "3. Type 'files' to untrack and remove specific files or folders (e.g., 'file.mp4' or 'folder/')."
read -p "Enter your choice (all/ext/files): " INPUT_TYPE

FILES_TO_REMOVE=()

if [ "$INPUT_TYPE" == "all" ]; then
	echo ""
	echo "Collecting all LFS files..."

	while IFS= read -r file; do
		if [ -n "$file" ]; then
			FILES_TO_REMOVE+=("$file")
		fi
	done < <(git lfs ls-files | awk '{print $3}')

elif [ "$INPUT_TYPE" == "ext" ]; then
	read -p "Enter file extension (e.g., *.mp4): " EXTENSION

	echo ""
	echo "Collecting files with extension: $EXTENSION"

	# Normalize extension (remove leading *)
	EXT_PATTERN="${EXTENSION#\*}"

	while IFS= read -r file; do
		if [[ "$file" == *"$EXT_PATTERN" ]]; then
			FILES_TO_REMOVE+=("$file")
		fi
	done < <(git lfs ls-files | awk '{print $3}')

elif [ "$INPUT_TYPE" == "files" ]; then
	echo ""
	echo "Enter files or folders to remove (press Enter on empty line to finish):"
	echo ""

	while true; do
		read -p "File/folder name: " FILE_NAME

		if [ "$FILE_NAME" == "" ]; then
			break
		fi

		if [ -d "$FILE_NAME" ]; then
			# Directory - add all files in it
			echo "Removing all files in folder: $FILE_NAME"
			while IFS= read -r file; do
				if [ -f "$file" ]; then
					FILES_TO_REMOVE+=("$file")
				fi
			done < <(find "$FILE_NAME" -type f)

		elif [ -f "$FILE_NAME" ]; then
			# Single file
			echo " Removing file: $FILE_NAME"
			FILES_TO_REMOVE+=("$FILE_NAME")

		else
			echo "⚠️'$FILE_NAME' not found, skipping"
		fi
	done

	if [ ${#FILES_TO_REMOVE[@]} -eq 0 ]; then
		echo "❌ No files selected!"
		exit 1
	fi

	echo ""
	echo "✅ Collected ${#FILES_TO_REMOVE[@]} file(s)"

else
	echo "❌ Invalid choice:  $INPUT_TYPE"
	echo "Please run again and choose:  all, ext, or files"
	exit 1
fi

FOLDERS_CHANGED=()

# Extract unique folders from FILES_TO_REMOVE
for file in "${FILES_TO_REMOVE[@]}"; do
	folder=$(dirname "$file")
	if [[ ! " ${FOLDERS_CHANGED[@]} " =~ " ${folder} " ]]; then
		FOLDERS_CHANGED+=("$folder")
	fi
done

echo "$FILES_TO_REMOVE"
# STEP 1: UNTRACK FROM LFS

for file in "${FILES_TO_REMOVE[@]}"; do
	echo "  Untracking:  $file"
	git lfs untrack "$file" 2>/dev/null || true
	git rm --cached "$file" 2>/dev/null || true
done

git add .gitattributes
git commit -m "Untrack and remove LFS files:  ${FILES_TO_REMOVE[*]}"
git push origin main

echo "✅ Step 1 complete"

# STEP 2: REMOVE FROM GIT HISTORY

for file in "${FILES_TO_REMOVE[@]}"; do
	basename_file=$(basename "$file")
	java -jar git_script/bfg.jar --delete-files "$basename_file" --no-blob-protection .
done

echo "  Cleaning repository..."
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# STEP 3:  COMMIT GITATTRIBUTES & PUSH EVERYTHING

git push --force --all && git push --force --tags

git fetch origin --force --prune
git reset --hard origin/main

for branch in $(git branch | sed 's/^[* ]*//'); do
	if [ "$branch" != "main" ]; then
		git branch -D "$branch" 2>/dev/null || true
	fi
done

for branch in $(git branch -r | grep -v 'HEAD' | sed 's|origin/||'); do
	if [ "$branch" != "main" ]; then
		git checkout -B "$branch" "origin/$branch" 2>/dev/null || true
	fi
done

git checkout main

git lfs prune --verify-remote --force

rm -rf ..bfg-report 2>/dev/null || true

## record removed lfs files, create "git_lfs_removed_files_new.txt", and append it to "git_lfs_removed_files.txt"
touch "git_lfs_removed_files_new.txt"

for file in "${FILES_TO_REMOVE[@]}"; do
	if [ ! -s "git_lfs_removed_files_new.txt" ]; then
		echo "$file" >>git_lfs_removed_files_new.txt
	else
		echo -e "\n$file" >>git_lfs_removed_files_new.txt
	fi

done
# post-process: if "git_lfs_removed_files.txt" exists,
# check: can I combine those files into the pre-existing folder,
# if so, remove those files, replace with the folder only("add /")
if [ -s "git_lfs_removed_files_new.txt" ]; then

	# Create temporary file
	TEMP_FILE="git_lfs_removed_files_new.tmp"
	cp "git_lfs_removed_files_new.txt" "$TEMP_FILE"

	# Get all unique directories from tracked files
	DIRECTORIES=$(grep -v '/$' "$TEMP_FILE" | xargs -I {} dirname {} | sort -u)

	# For each directory, check if we should consolidate
	for dir in $DIRECTORIES; do
		# Skip current directory
		if [ "$dir" = "." ]; then
			continue
		fi

		# Check if folder entry already exists
		if grep -qxF "$dir/" "$TEMP_FILE"; then
			echo "  ℹ️  Folder already tracked: $dir/"
			continue
		fi

		# Get all files in this directory (from filesystem)
		if [ ! -d "$dir" ]; then
			continue
		fi

		ALL_FILES_IN_DIR=$(find "$dir" -type f -not -path "*/.git/*" 2>/dev/null)

		if [ -z "$ALL_FILES_IN_DIR" ]; then
			continue
		fi

		# Check if ALL files in directory are tracked
		ALL_TRACKED=true
		while IFS= read -r file_in_dir; do
			if [ -n "$file_in_dir" ]; then
				# Check if this file is in tracking list
				if ! grep -qxF "$file_in_dir" "$TEMP_FILE"; then
					ALL_TRACKED=false
					break
				fi
			fi
		done <<<"$ALL_FILES_IN_DIR"

		# If all files are tracked, consolidate to folder
		if [ "$ALL_TRACKED" = true ]; then
			echo "  ✅ Consolidating:  All files in '$dir/' are tracked"

			# Remove individual file entries for this directory
			grep -v "^$dir/" "$TEMP_FILE" >"${TEMP_FILE}.new"
			mv "${TEMP_FILE}.new" "$TEMP_FILE"

			# Add folder entry
			echo "" >>"$TEMP_FILE"
			echo "$dir/" >>"$TEMP_FILE"
		fi
	done

	# Clean up:  remove empty lines and sort
	sed -i '/^[[:space:]]*$/d' "$TEMP_FILE" 2>/dev/null ||
		sed -i '' '/^[[:space:]]*$/d' "$TEMP_FILE" 2>/dev/null || true

	sort -u "$TEMP_FILE" -o "$TEMP_FILE"

	# Replace original file
	mv "$TEMP_FILE" "git_lfs_removed_files_new.txt"

	echo "  📝 Consolidation complete"
fi

## add files from "git_lfs_removed_files_new.txt" to .gitignore

if [ -s "git_lfs_removed_files_new.txt" ]; then
	echo "Adding files to .gitignore..."
	if ! grep -qxF "# lfs files" ".gitignore"; then
		if [ -s ".gitignore" ]; then
			perl -0pi -e "s/\\R*\\z//g" ".gitignore"
			echo -e "\n\n# lfs files" >>.gitignore
		else
			echo "# lfs files" >>.gitignore
		fi
	else
		perl -0pi -e 's/\R*\z//g' ".gitignore"
		printf '\n# %s\n' "$(printf '=%.0s' {1..50})" >>".gitignore"
	fi

	# Loop through each line in the removed files list
	while IFS= read -r file; do
		# Check if the file is not already in .gitignore
		if ! grep -qxF "$file" ".gitignore"; then
			# Check if the last line in .gitignore is '# lfs files'
			LAST_NON_EMPTY=$(grep -v '^[[:space:]]*$' .gitignore | tail -n 1)

			if [ "$LAST_NON_EMPTY" = "# lfs files" ] || [[ "$LAST_NON_EMPTY" =~ ^#\ =+ ]]; then
				echo "$file" >>.gitignore
			else
				echo -e "\n$file" >>.gitignore
			fi
			echo "  Added to .gitignore: $file"
		else
			echo "  Already in .gitignore: $file"
		fi
	done <"git_lfs_removed_files_new.txt"

	echo "✅ Files added to .gitignore"
else
	echo "❌ No files to add to .gitignore, 'git_lfs_removed_files.txt' is empty."
fi

# move content from "git_lfs_removed_files_new.txt" to "git_lfs_removed_files.txt", and "="*50
if [ ! -s "git_lfs_removed_files.txt" ]; then
	touch "git_lfs_removed_files.txt"
fi

# Add separator
if [ -s "git_lfs_removed_files.txt" ]; then
	perl -0pi -e 's/\R*\z//g' "git_lfs_removed_files.txt"
	printf '\n%*s\n' 50 '' | tr ' ' '=' >>"git_lfs_removed_files.txt"
fi

# Append new content
cat "git_lfs_removed_files_new.txt" >>"git_lfs_removed_files.txt"

# remove temporary file
rm -f "git_lfs_removed_files_new.txt"

for folder in "${FOLDERS_CHANGED[@]}"; do
	update_folder_in_gitattributes "$folder"
done

git add ".gitignore" ".gitattributes" "git_lfs_removed_files.txt"
git commit -m "update"
git push origin main
