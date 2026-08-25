# Dot-source this script so the variables remain available in the current shell:
#   . .\scripts\activate_env.ps1

$repoRoot = Split-Path -Parent $PSScriptRoot
$activateScript = Join-Path $repoRoot ".venv\Scripts\Activate.ps1"
$envFile = Join-Path $repoRoot ".env"

if (-not (Test-Path -LiteralPath $activateScript)) {
    throw "Virtual environment not found. Run: py -3.11 -m venv .venv"
}

if (-not (Test-Path -LiteralPath $envFile)) {
    throw ".env not found. Copy .env.example to .env and add your API tokens."
}

. $activateScript

foreach ($rawLine in Get-Content -LiteralPath $envFile -Encoding UTF8) {
    $line = $rawLine.Trim()
    if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
        continue
    }

    $parts = $line.Split("=", 2)
    $key = $parts[0].Trim()
    $value = $parts[1].Trim().Trim('"').Trim("'")

    if ($key -match '^[A-Za-z_][A-Za-z0-9_]*$' -and $value) {
        Set-Item -Path "Env:$key" -Value $value
    }
}

# Always use UTF-8 even before a Kaggle token has been configured.
$env:PYTHONUTF8 = "1"

$geminiState = if ($env:GEMINI_API_KEY) { "configured" } else { "missing" }
$kaggleState = if ($env:KAGGLE_API_TOKEN) { "configured" } else { "missing" }
$nvidiaState = if ($env:NVIDIA_API_KEY) { "configured" } else { "missing" }

Write-Host "Environment ready: $repoRoot"
Write-Host "Python: $((Get-Command python).Source)"
Write-Host "Gemini token: $geminiState | NVIDIA token: $nvidiaState | Kaggle token: $kaggleState"
