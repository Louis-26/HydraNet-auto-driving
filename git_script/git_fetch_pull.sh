cd "$(git rev-parse --show-toplevel)"
# download from remote repository to local directory
# pull from remote repository to keep local directory updated

CURRENT_BRANCH=$(git branch --show-current)

git fetch

git checkout "$CURRENT_BRANCH"

git pull origin "$CURRENT_BRANCH"

