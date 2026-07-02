cd "$(git rev-parse --show-toplevel)"
# verify that we are in a git repo
if ! git remote get-url origin >/dev/null 2>&1; then
	echo "ERROR: This script must be run inside a git repository with an origin remote"
	exit 1
fi

OWNER=$(git remote get-url origin |
	sed -E 's#(https://github.com/|git@github.com:)([^/]+)/.*#\2#')

OLD=$(git remote get-url origin |
	sed -E 's#(https://github.com/|git@github.com:)[^/]+/([^/]+)#\2#' | sed 's/\.git$//')

if [ "$INPUT_TYPE" == "copy" ]; then

	read -p "Enter the github username for the new repo(directly enter if it is in the same account): " NEW_OWNER
	# if you want to keep it in the same account
	if [ -z "$NEW_OWNER" ]; then
		read -p "Enter the NEW repo name: " NEW
		if [ $NEW == "" ] || [ $NEW == "$OLD" ]; then
			echo "❌ You must enter a nonempty different repo name."
			exit 1
		fi

		# if you want to change to a different account
	else
		read -p "Enter the NEW repo name(directly enter if you want to keep the original one): " NEW
		NEW=${NEW:-$OLD}
		read -sp "Enter the GitHub Personal Access Token for $NEW_OWNER: " TARGET_TOKEN
	fi

	NEW_OWNER=${NEW_OWNER:-$OWNER}

	echo "Detecting visibility for $OWNER/$OLD..."

	VISIBILITY=$(gh repo view "$OWNER/$OLD" --json visibility -q .visibility)

	case "$VISIBILITY" in
	PUBLIC) CREATE_FLAG="--public" ;;
	PRIVATE) CREATE_FLAG="--private" ;;
	INTERNAL) CREATE_FLAG="--internal" ;;
	*)
		echo "ERROR: Unknown visibility: $VISIBILITY"
		exit 1
		;;
	esac

	echo "Migrating $OWNER/$OLD → $NEW_OWNER/$NEW ($VISIBILITY)"

	if [ ! -z "$TARGET_TOKEN" ]; then
		export GITHUB_TOKEN="$TARGET_TOKEN"
	fi

	gh repo create "$NEW_OWNER/$NEW" $CREATE_FLAG --confirm

	git clone --mirror "https://github.com/$OWNER/$OLD.git"
	cd "$OLD.git"

	git config http.postBuffer 524288000
	git config core.compression 0

	if [ ! -z "$TARGET_TOKEN" ]; then
		PUSH_URL="https://$NEW_OWNER:$TARGET_TOKEN@github.com/$NEW_OWNER/$NEW.git"
	else
		PUSH_URL="https://github.com/$NEW_OWNER/$NEW.git"
	fi
	git push --mirror "$PUSH_URL"

	cd ..
	rm -rf "$OLD.git"

else
	echo "❌ Invalid choice:  $INPUT_TYPE"
	echo "Please run again and choose:  copy or migrate"
	exit 1
fi
