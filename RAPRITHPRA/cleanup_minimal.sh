#!/bin/bash
# Simpler bash script to clean up the most space-consuming unnecessary files
# Run this in the root directory of your project

echo "Cleaning up unnecessary files to save space..."

# Remove directories that take up the most space and are not needed for production
dirs_to_remove=(
    "sunita willams/sunita williams/AI-CHATBOT/__pycache__"
    "sunita willams/sunita williams/AI-CHATBOT/speech_cache"
    "sunita willams/sunita williams/AI-CHATBOT/tests"
    "sunita willams/sunita williams/AI-CHATBOT/temp"
    "temp"
    "temp_audio"
)

# Remove directories
for dir in "${dirs_to_remove[@]}"; do
    if [ -d "$dir" ]; then
        echo "Removing directory: $dir"
        rm -rf "$dir"
    fi
done

# Remove specific files
echo "Removing server.py.bak"
rm -f "sunita willams/sunita williams/AI-CHATBOT/server.py.bak"

echo "Removing create_test_user.py"
rm -f "sunita willams/sunita williams/AI-CHATBOT/create_test_user.py"

# Remove files with wildcard patterns
echo "Removing test_*.py files"
find "sunita willams/sunita williams/AI-CHATBOT" -name "test_*.py" -type f -delete

echo "Removing conversation history files"
find "sunita willams/sunita williams/AI-CHATBOT" -name "conversation_history_*.json" -type f -delete

# Remove all Python bytecode files
echo "Removing Python bytecode files (.pyc)"
find . -name "*.pyc" -type f -delete
find . -name "__pycache__" -type d -exec rm -rf {} \; 2>/dev/null || true

echo "Cleaning completed. Your project is now smaller and ready for hosting." 