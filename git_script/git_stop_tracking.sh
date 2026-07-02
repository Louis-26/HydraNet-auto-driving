cd "$(git rev-parse --show-toplevel)"
touch .gitignore
# uncomment if it is in linux, and need to convert dos to unix
# sed -i 's/\r$//' git_script/git_stop_tracking.sh
# this script is used to stop tracking files or folders
while true; do
	read -r -p "Enter the name of file/folder you want to stop tracking, or directly enter to exit: " FILE_NAME

	if [ "$FILE_NAME" == "" ]; then
		break
	fi

	if [ -s .gitignore ]; then
        prefix=$'\n'
    else
        prefix=""
    fi

	# directory
	if [ -d "$FILE_NAME" ]; then
        git rm -r --cached "$FILE_NAME"

        CLEAN_NAME=$(echo "$FILE_NAME" | sed 's/^\///;s/\/$//')
        if ! grep -Fxq "/$CLEAN_NAME/" .gitignore; then

            printf "%s/%s/" "$prefix" "$CLEAN_NAME" >> .gitignore
            echo "Added directory /$CLEAN_NAME/ to .gitignore"
        fi

    # file
    elif [ -f "$FILE_NAME" ]; then
        git rm --cached "$FILE_NAME"
        CLEAN_NAME=$(echo "$FILE_NAME" | sed 's/^\///')
        if ! grep -Fxq "/$CLEAN_NAME" .gitignore; then
            printf "%s/%s" "$prefix" "$CLEAN_NAME" >> .gitignore
            echo "Added file /$CLEAN_NAME to .gitignore"
        fi

	else
		echo "'$FILE_NAME' not found as a file or folder, skipping"
	fi
done
bash gitignore_consolidate.sh
