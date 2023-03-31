SORT_DIR="$1"
if [ -z "$SORT_DIR" ]; then
    echo "Usage: run linter {directory e.g. .}"
    exit 1
fi

echo "Formatting code with isort..."
pipenv run isort $SORT_DIR 

echo "Formatting code with black..." 
pipenv run black $SORT_DIR 

echo "Removing unused imports with autoflake..." 
pipenv run autoflake --remove-unused-variables --remove-all-unused-imports --recursive --exclude '**/__init__.py' --in-place $SORT_DIR 

echo "Done formatting code!"
