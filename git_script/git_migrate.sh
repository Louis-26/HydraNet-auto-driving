cd "$(git rev-parse --show-toplevel)"
# run this script to copy one repo to create a new repo with same content, different name
# alternatively, transfer the repo from a lfs repo to a non-lfs repo, make sure the current version has already removed all lfs content
echo "Select the type:"
echo "1. Type 'copy' to copy one repo to create a new repo with same content, different name."
echo "2. Type 'migrate' to transfer the repo from a lfs repo to a non-lfs repo."
read -p "Enter your choice: " INPUT_TYPE

# verify that we are in a git repo
if ! git remote get-url origin >/dev/null 2>&1; then
	echo "ERROR: This script must be run inside a git repository with an origin remote"
	exit 1
fi

OWNER=$(git remote get-url origin |
	sed -E 's#(https://github.com/|git@github.com:)([^/]+)/.*#\2#')

OLD=$(git remote get-url origin |
	sed -E 's#(https://github.com/|git@github.com:)[^/]+/([^/]+)#\2#' | sed 's/\.git$//')

if [ "$INPUT_TYPE" == "migrate" ]; then

	TEMP="${OLD}_temp"

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

	# Step 1: create temp repo
	gh repo create "$OWNER/$TEMP" $CREATE_FLAG --confirm

	# Step 2: mirror push
	git clone --mirror "https://github.com/$OWNER/$OLD.git"
	cd "$OLD.git"
	git config http.postBuffer 524288000
	git config core.compression 0

	git push --mirror "https://github.com/$OWNER/$TEMP.git"

	cd ..
	rm -rf "$OLD.git"

	# Step 3: delete old repo
	gh repo delete "$OWNER/$OLD" --confirm

	# Step 4: rename temp repo back to original name
	gh repo rename "$OLD" --repo "$OWNER/$TEMP" --yes

else
	echo "❌ Invalid choice:  $INPUT_TYPE"
	echo "Please run again and choose:  copy or migrate"
	exit 1
fi
