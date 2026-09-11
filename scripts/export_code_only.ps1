param(
    [string]$Destination = ""
)

$ErrorActionPreference = "Stop"
$Root = (git rev-parse --show-toplevel).Trim()
if (-not $Root) { throw "Run this script inside the repository." }

if (-not $Destination) {
    $Destination = Join-Path (Split-Path $Root -Parent) "wildboar-france-code-only"
}
elseif (-not [System.IO.Path]::IsPathRooted($Destination)) {
    $Destination = Join-Path (Get-Location) $Destination
}
$Destination = [System.IO.Path]::GetFullPath($Destination)
$Archive = Join-Path ([System.IO.Path]::GetTempPath()) ("wildboar-france-code-only-" + [guid]::NewGuid() + ".zip")

Push-Location $Root
try {
    python scripts/validate_repository.py
    if ($LASTEXITCODE -ne 0) { throw "Repository hygiene validation failed." }
    if (Test-Path $Destination) { throw "Destination already exists; choose a new directory: $Destination" }
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null

    git archive --format=zip --output=$Archive HEAD
    if ($LASTEXITCODE -ne 0) { throw "Git archive failed." }
    Expand-Archive -Path $Archive -DestinationPath $Destination

    python "$Destination/scripts/validate_repository.py" `
        --root $Destination `
        --all-files
    if ($LASTEXITCODE -ne 0) { throw "Export hygiene validation failed." }
}
finally {
    if (Test-Path $Archive) { Remove-Item $Archive -Force }
    Pop-Location
}

Write-Host ""
Write-Host "Created a history-free code export at:"
Write-Host "  $Destination"
Write-Host ""
Write-Host "The export contains this commit without Git history."
Write-Host "To publish it separately, use a new empty repository."
Write-Host "Initialize the export with:"
Write-Host "  cd `"$Destination`""
Write-Host "  git init -b main"
Write-Host "  git add ."
Write-Host "  git commit -m `"Initial public code release`""
