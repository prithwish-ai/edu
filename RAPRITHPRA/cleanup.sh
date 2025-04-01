#!/bin/bash
# Bash script to clean up unnecessary files before hosting
# Run this in the root directory of your project

# Create a function to safely remove items
remove_safely() {
    if [ -e "$1" ]; then
        echo "Removing: $1"
        rm -rf "$1"
    else
        echo "Path not found: $1"
    fi
}

# Create a backup directory
timestamp=$(date +"%Y%m%d-%H%M%S")
backup_dir="backups/backup-$timestamp"
mkdir -p "$backup_dir"
echo "Created backup directory: $backup_dir"

# 1. Remove development and test files
development_files=(
    "sunita willams/sunita williams/AI-CHATBOT/__pycache__"
    "sunita willams/sunita williams/AI-CHATBOT/tests"
    "sunita willams/sunita williams/AI-CHATBOT/examples"
    "sunita willams/sunita williams/AI-CHATBOT/docs"
    "sunita willams/sunita williams/AI-CHATBOT/dist"
    "sunita willams/sunita williams/AI-CHATBOT/setup.py"
    "sunita willams/sunita williams/AI-CHATBOT/install.sh"
    "sunita willams/sunita williams/AI-CHATBOT/install.bat"
    "sunita willams/sunita williams/AI-CHATBOT/create_test_user.py"
    "sunita willams/sunita williams/AI-CHATBOT/STRUCTURE.md"
    "sunita willams/sunita williams/AI-CHATBOT/.git"
    "eduspark-tts/.git"
    ".git"
)

for file in "${development_files[@]}"; do
    remove_safely "$file"
done

# Handle wildcard patterns
find "sunita willams/sunita williams/AI-CHATBOT" -name "test_*.py" -type f -exec rm -f {} \;

# 2. Remove temporary files and caches
temp_files=(
    "sunita willams/sunita williams/AI-CHATBOT/temp"
    "sunita willams/sunita williams/AI-CHATBOT/speech_cache"
    "sunita willams/sunita williams/AI-CHATBOT/cache"
    "sunita willams/sunita williams/AI-CHATBOT/logs"
    "eduspark-tts/temp_audio"
    "eduspark-tts/temp"
    "eduspark-tts/logs"
    "temp"
    "temp_audio"
)

for file in "${temp_files[@]}"; do
    remove_safely "$file"
done

# 3. Remove backup and unnecessary data files
remove_safely "sunita willams/sunita williams/AI-CHATBOT/server.py.bak"

# Handle wildcard patterns for conversation history files
find "sunita willams/sunita williams/AI-CHATBOT" -name "conversation_history_*.json" -type f -exec rm -f {} \;

# 4. Find and remove all .pyc files (Python bytecode)
find . -name "*.pyc" -type f -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

echo -e "\nCleaning completed. Your project is now ready for hosting with minimal size."
echo "Note: If you need to restore any files, check the backup directory: $backup_dir" 