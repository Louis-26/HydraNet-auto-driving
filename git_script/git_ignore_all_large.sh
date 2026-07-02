#!/usr/bin/env bash
# ignore all files larger than 100MB and append them into .gitignore
cd "$(git rev-parse --show-toplevel)"

DEFAULT_SIZE=104857600



touch .gitignore

# Ensure .gitignore ends with a newline before appending
if [ -s .gitignore ] && [ "$(tail -c 1 .gitignore)" != "" ]; then
    printf "\n" >> .gitignore
fi

echo "Searching for files larger than $DEFAULT_SIZE bytes (respecting .gitignore) ..."

ADDED_COUNT=0
# Only look at files Git cares about: tracked + untracked-but-not-ignored
while IFS= read -r -d '' FILE; do
    [ -f "$FILE" ] || continue

    SIZE=$(stat -c%s "$FILE" 2>/dev/null) || continue
    [ "$SIZE" -gt "$DEFAULT_SIZE" ] || continue

    ENTRY="/$FILE"

    if grep -Fxq "$ENTRY" .gitignore; then
        echo "Already ignored: $ENTRY"
        continue
    fi

    echo "$ENTRY" >> .gitignore
    echo "Added to .gitignore: $ENTRY ($SIZE bytes)"

    git rm --cached --quiet "$FILE" 2>/dev/null || true

    ADDED_COUNT=$((ADDED_COUNT + 1))
done < <(git ls-files --cached --others --exclude-standard -z)

# bash gitignore_consolidate.sh

echo "Done. Added $ADDED_COUNT new entries to .gitignore."