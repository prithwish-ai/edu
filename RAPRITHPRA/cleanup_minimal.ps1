# Simpler PowerShell script to clean up the most space-consuming unnecessary files
# Run this in the root directory of your project

# These are the directories that take up the most space and are not needed for production
$dirsToRemove = @(
    "sunita willams/sunita williams/AI-CHATBOT/__pycache__",
    "sunita willams/sunita williams/AI-CHATBOT/speech_cache",
    "sunita willams/sunita williams/AI-CHATBOT/tests",
    "sunita willams/sunita williams/AI-CHATBOT/temp",
    "temp",
    "temp_audio"
)

# These are individual files that are not needed for production
$filesToRemove = @(
    "sunita willams/sunita williams/AI-CHATBOT/server.py.bak",
    "sunita willams/sunita williams/AI-CHATBOT/test_*.py",
    "sunita willams/sunita williams/AI-CHATBOT/conversation_history_*.json",
    "sunita willams/sunita williams/AI-CHATBOT/create_test_user.py"
)

# Remove directories
foreach ($dir in $dirsToRemove) {
    if (Test-Path $dir) {
        Write-Host "Removing directory: $dir"
        Remove-Item -Path $dir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Remove individual files
foreach ($filePattern in $filesToRemove) {
    if ($filePattern -like "*`**") {
        # It's a wildcard pattern
        $basePath = Split-Path -Path $filePattern
        $pattern = Split-Path -Path $filePattern -Leaf
        
        Get-ChildItem -Path $basePath -Filter $pattern -ErrorAction SilentlyContinue | ForEach-Object {
            Write-Host "Removing file: $($_.FullName)"
            Remove-Item -Path $_.FullName -Force -ErrorAction SilentlyContinue
        }
    } else {
        # It's a specific file
        if (Test-Path $filePattern) {
            Write-Host "Removing file: $filePattern"
            Remove-Item -Path $filePattern -Force -ErrorAction SilentlyContinue
        }
    }
}

Write-Host "Cleaning completed. Your project is now smaller and ready for hosting." 