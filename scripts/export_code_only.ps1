param(
    [string]$Destination = ""
)

$ErrorActionPreference = "Stop"
$Root = (git rev-parse --show-toplevel).Trim()
if (-not $Root) { throw "Run this script inside the repository." }

if (-not $Destination) {
    $Destination = Join-Path (Split-Path $Root -Parent) "maxent-code-only"
}
elseif (-not [System.IO.Path]::IsPathRooted($Destination)) {
    $Destination = Join-Path (Get-Location) $Destination
}
$Destination = [System.IO.Path]::GetFullPath($Destination)
$Archive = Join-Path ([System.IO.Path]::GetTempPath()) ("maxent-code-only-" + [guid]::NewGuid() + ".zip")

Push-Location $Root
try {
    python scripts/validate_repository.py
    if (Test-Path $Destination) { Remove-Item $Destination -Recurse -Force }
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null

    git archive --format=zip --output=$Archive HEAD
    Expand-Archive -Path $Archive -DestinationPath $Destination -Force

    python "$Destination/scripts/validate_repository.py" `
        --root $Destination `
        --all-files
}
finally {
    if (Test-Path $Archive) { Remove-Item $Archive -Force }
    Pop-Location
}

Write-Host ""
Write-Host "Created a history-free code export at:"
Write-Host "  $Destination"
Write-Host ""
Write-Host "Publish this directory as a NEW repository. Do not make the source"
Write-Host "repository public because deleted data remain in its Git history."
Write-Host "Initialize the export with:"
Write-Host "  cd `"$Destination`""
Write-Host "  git init -b main"
Write-Host "  git add ."
Write-Host "  git commit -m `"Initial public code release`""
