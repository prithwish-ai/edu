# PowerShell script to clean up unnecessary files before hosting
# Run this in the root directory of your project

# Create a function to safely remove items
function Remove-Safely {
    param (
        [string]$Path
    )
    
    if (Test-Path $Path) {
        Write-Host "Removing: $Path"
        if ((Get-Item $Path).PSIsContainer) {
            Remove-Item $Path -Recurse -Force
        } else {
            Remove-Item $Path -Force
        }
    } else {
        Write-Host "Path not found: $Path"
    }
}

# Create a backup directory
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backupDir = "backups/backup-$timestamp"
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
Write-Host "Created backup directory: $backupDir"

# 1. Remove development and test files
$developmentFiles = @(
    "sunita willams/sunita williams/AI-CHATBOT/__pycache__",
    "sunita willams/sunita williams/AI-CHATBOT/tests",
    "sunita willams/sunita williams/AI-CHATBOT/examples",
    "sunita willams/sunita williams/AI-CHATBOT/docs",
    "sunita willams/sunita williams/AI-CHATBOT/dist",
    "sunita willams/sunita williams/AI-CHATBOT/test_*.py",
    "sunita willams/sunita williams/AI-CHATBOT/setup.py",
    "sunita willams/sunita williams/AI-CHATBOT/install.sh",
    "sunita willams/sunita williams/AI-CHATBOT/install.bat",
    "sunita willams/sunita williams/AI-CHATBOT/create_test_user.py",
    "sunita willams/sunita williams/AI-CHATBOT/STRUCTURE.md",
    "sunita willams/sunita williams/AI-CHATBOT/.git",
    "eduspark-tts/.git",
    ".git"
)

foreach ($file in $developmentFiles) {
    # If it's a wildcard pattern, handle differently
    if ($file -match "\*") {
        $dir = Split-Path -Path $file
        $pattern = Split-Path -Path $file -Leaf
        $items = Get-ChildItem -Path $dir -Filter $pattern
        foreach ($item in $items) {
            Remove-Safely -Path $item.FullName
        }
    } else {
        Remove-Safely -Path $file
    }
}

# 2. Remove temporary files and caches
$tempFiles = @(
    "sunita willams/sunita williams/AI-CHATBOT/temp",
    "sunita willams/sunita williams/AI-CHATBOT/speech_cache",
    "sunita willams/sunita williams/AI-CHATBOT/cache",
    "sunita willams/sunita williams/AI-CHATBOT/logs",
    "eduspark-tts/temp_audio",
    "eduspark-tts/temp",
    "eduspark-tts/logs",
    "temp",
    "temp_audio"
)

foreach ($file in $tempFiles) {
    Remove-Safely -Path $file
}

# 3. Remove backup and unnecessary data files
$dataFiles = @(
    "sunita willams/sunita williams/AI-CHATBOT/server.py.bak",
    "sunita willams/sunita williams/AI-CHATBOT/conversation_history_*.json"
)

foreach ($file in $dataFiles) {
    # If it's a wildcard pattern, handle differently
    if ($file -match "\*") {
        $dir = Split-Path -Path $file
        $pattern = Split-Path -Path $file -Leaf
        $items = Get-ChildItem -Path $dir -Filter $pattern
        foreach ($item in $items) {
            Remove-Safely -Path $item.FullName
        }
    } else {
        Remove-Safely -Path $file
    }
}

Write-Host "`nCleaning completed. Your project is now ready for hosting with minimal size."
Write-Host "Note: If you need to restore any files, check the backup directory: $backupDir" 