cd "$(git rev-parse --show-toplevel)"
# use git lfs to push large files or folders to remote repository

# this script only pushes those large files, but doesn't push other normal files.

# Uncomment if you need to convert DOS to Unix line endings
# sed -i 's/\r$//' git_script/git_lfs_push.sh

# find the current branch
CURRENT_BRANCH=$(git branch --show-current)

# record newly tracked files in "git_lfs_tracked_files_new.txt"
touch "git_lfs_tracked_files_new.txt"

echo "Select the type of files to track:"
echo "1. Type 'all' to trackall LFS files."
echo "2. Type 'ext' to track  files with certain extensions (e.g., '.mp4')."
echo "3. Type 'files' to track specific files or folders (e.g., 'file.mp4' or 'folder/')."
read -p "Enter your choice (all/ext/files): " INPUT_TYPE

# if [ -s ".gitattributes" ]; then
#     perl -0pi -e 's/\R*\z//g' ".gitattributes"
# 	printf '\n# %s\n' "$(printf '=%.0s' {1..50})" >>".gitattributes"
# fi
# Function to scan files and automatically track those over 100MB
scan_and_push_large_files() {
	echo "Scanning all files to identify those over 100MB..."

	# Find all files over 100MB and add them to git-lfs tracking

	(find . -type f -size +104857600c -not -path "./.git/*") | while read -r file; do
		# if the file is in .gitignore, skip it
		if git check-ignore -q "$file"; then
			echo "Skipping ignored file: $file"
			continue
		fi
		# Remove the leading './' from the file path if present
		file="${file#./}"
		echo "Tracking large file: $file"
		git lfs track "$file"
		git add "$file"
		if [ ! -s "git_lfs_tracked_files_new.txt" ]; then
			echo "$file" >>"git_lfs_tracked_files_new.txt"
		else
			echo -e "\n$file" >>"git_lfs_tracked_files_new.txt"
		fi

	done

	# remove if empty
	if [ ! -s "git_lfs_tracked_files_new.txt" ]; then
		rm -f "git_lfs_tracked_files_new.txt"

	fi

}

# Initialize Git LFS
git lfs install

# Process based on user input
if [ "$INPUT_TYPE" == "all" ]; then
	scan_and_push_large_files
elif [ "$INPUT_TYPE" == "ext" ]; then
	read -p "Enter the file extension to track and push (e.g., '.mp4'): " FILE_EXT

	(find . -type f -name "*$FILE_EXT" -size +104857600c -not -path "./.git/*") | while read -r file; do
		# if the file is in .gitignore, skip it
		if git check-ignore -q "$file"; then
			echo "Skipping ignored file: $file"
			continue
		fi
		file="${file#./}"
		echo "Tracking file: $file"
		git lfs track "$file"
		git add "$file"
		if [ ! -s "git_lfs_tracked_files_new.txt" ]; then
			echo "$file" >>"git_lfs_tracked_files_new.txt"
		else
			echo -e "\n$file" >>"git_lfs_tracked_files_new.txt"
		fi
	done
elif [ "$INPUT_TYPE" == "files" ]; then
	# Manual file/folder tracking and pushing
	while true; do
		# Ask for the file/folder path
		read -p "Enter the path of the file or folder to track and push (or directly enter to finish): " TARGET_PATH

		# Exit condition
		if [ "$TARGET_PATH" == "" ]; then
			echo "Finished processing all files/folders."
			break
		fi

		# Check if the path exists
		if [ ! -e "$TARGET_PATH" ]; then
			echo "Error: '$TARGET_PATH' does not exist."
			continue
		fi

		# Determine if it's a file or folder
		if [ -f "$TARGET_PATH" ]; then
			# if the file is in .gitignore, skip it
			if git check-ignore -q "$TARGET_PATH"; then
				echo "Skipping ignored file: $file"
				continue
			fi
			# if file size < 100MB, stop and continue
			FILE_SIZE_BYTES=$(stat -c%s "$TARGET_PATH" 2>/dev/null || stat -f%z "$TARGET_PATH" 2>/dev/null)
			if [ "$FILE_SIZE_BYTES" -lt 104857600 ]; then
				echo "Skipping '$TARGET_PATH': File size is less than 100MB."
				continue
			fi
			echo "Tracking a single file: $TARGET_PATH"
			git lfs track "$TARGET_PATH"
			if [ ! -s "git_lfs_tracked_files_new.txt" ]; then
				echo "$TARGET_PATH" >>"git_lfs_tracked_files_new.txt"
			else
				echo -e "\n$TARGET_PATH" >>"git_lfs_tracked_files_new.txt"
			fi
		elif [ -d "$TARGET_PATH" ]; then
			# if the folder is in .gitignore, skip it
			if git check-ignore -q "${TARGET_PATH}/"; then
				echo "Skipping ignored file: $file"
				continue
			fi
			echo "Tracking all files in folder: $TARGET_PATH"
			SKIP_FOLDER=false

			# Iterate through all files inside the folder
			find "$TARGET_PATH" -type f | while read -r file; do
				FILE_SIZE_BYTES=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)

				# If any file inside the folder is smaller than 100MB, skip the folder entirely
				if [ "$FILE_SIZE_BYTES" -lt 104857600 ]; then
					echo "Skipping folder '$TARGET_PATH' because it contains small file: $file"
					SKIP_FOLDER=true
					break # Exit the loop early since the folder is to be skipped
				fi
			done
			git lfs track "$TARGET_PATH/**"
			if [ ! -s "git_lfs_tracked_files_new.txt" ]; then
				echo "$TARGET_PATH/" >>"git_lfs_tracked_files_new.txt"
			else
				echo -e "\n$TARGET_PATH/" >>"git_lfs_tracked_files_new.txt"
			fi
		else
			echo "Error: '$TARGET_PATH' is neither a file nor a folder."
			continue
		fi

		# Stage .gitattributes and the selected file/folder
		git add "$TARGET_PATH"
	done

	# Commit and push to the corresponding branch
	# if the tracked files list is empty, remove the file
	if [ ! -s "git_lfs_tracked_files_new.txt" ]; then
		rm -f "git_lfs_tracked_files_new.txt"
	fi

else
	echo "Invalid choice! No action taken."
	exit 0
fi

# post-process: if "git_lfs_tracked_files_new.txt" exists, check: can I combine those files into the pre-existing folder,
# if so, remove those files, replace with the folder only("add /")
if [ -s "git_lfs_tracked_files_new.txt" ]; then

	# Create temporary file
	TEMP_FILE="git_lfs_tracked_files_new.tmp"
	cp "git_lfs_tracked_files_new.txt" "$TEMP_FILE"

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

		ALL_FILES_IN_DIR=$(find "$dir" -type f -not -path "*/.git/*" 2>/dev/null | sed 's|^\./||')

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
	mv "$TEMP_FILE" "git_lfs_tracked_files_new.txt"
	echo "  📝 Consolidation complete"

fi

# then move the "git_lfs_tracked_files_new.txt" to "git_lfs_tracked_files.txt", with "="*50 separator
if [ ! -e "git_lfs_tracked_files.txt" ]; then
	touch "git_lfs_tracked_files.txt"
fi

# Add separator
if [ -s "git_lfs_tracked_files.txt" ]; then
	perl -0pi -e 's/\R*\z//g' "git_lfs_tracked_files.txt"
	printf '\n%*s\n' 50 '' | tr ' ' '=' >>"git_lfs_tracked_files.txt"
fi
# Append new content
cat "git_lfs_tracked_files_new.txt" >>"git_lfs_tracked_files.txt"

# remove temporary file
rm -f "git_lfs_tracked_files_new.txt"

# combine .gitattributes entries if needed
if [ -s ".gitattributes" ]; then
	# Check for folders with all files listed
	while IFS= read -r line; do
		if [[ "$line" == */* ]]; then
			folder="${line%/*}"
			if [ -d "$folder" ]; then
				all_files_tracked=true
				for file in "$folder"/*; do
					if ! grep -q "$file" ".gitattributes"; then
						all_files_tracked=false
						break
					fi
				done

				if [ "$all_files_tracked" = true ]; then
					sed -i "/^$folder\//d" ".gitattributes"
					echo "$folder/** filter=lfs diff=lfs merge=lfs -text" >>".gitattributes"
				fi
			fi
		fi
	done <".gitattributes"

	# Replace the original .gitattributes with the cleaned-up version
	echo "  📝 Consolidated entries in .gitattributes, added folder patterns where applicable."
fi

# final push
git add ".gitattributes" "git_lfs_tracked_files.txt"
git commit -m "update"
git push origin "$CURRENT_BRANCH"
