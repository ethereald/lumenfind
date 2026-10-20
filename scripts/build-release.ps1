$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$distRoot = Join-Path $repoRoot 'dist\LumenFind-windows-x64'
$resolvedDistRoot = [System.IO.Path]::GetFullPath($distRoot)
$resolvedRepoRoot = [System.IO.Path]::GetFullPath($repoRoot)
if (-not $resolvedDistRoot.StartsWith($resolvedRepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Refusing to write outside the repository.'
}

Push-Location $repoRoot
try {
    cargo build --release --locked
    New-Item -ItemType Directory -Force -Path $distRoot | Out-Null
    Copy-Item -LiteralPath (Join-Path $repoRoot 'target\release\lumenfind.exe') -Destination $distRoot -Force
    Copy-Item -LiteralPath (Join-Path $repoRoot 'target\release\lumenfind-gui.exe') -Destination $distRoot -Force
    Copy-Item -LiteralPath (Join-Path $repoRoot 'target\release\lumenfind-inspector.exe') -Destination $distRoot -Force
    Copy-Item -LiteralPath (Join-Path $repoRoot 'README.md') -Destination $distRoot -Force
    Copy-Item -LiteralPath (Join-Path $repoRoot 'LICENSE') -Destination $distRoot -Force
    Copy-Item -LiteralPath (Join-Path $repoRoot 'scripts\install-service.ps1') -Destination $distRoot -Force
    Copy-Item -LiteralPath (Join-Path $repoRoot 'scripts\uninstall-service.ps1') -Destination $distRoot -Force
    Write-Host "Built $distRoot"
}
finally {
    Pop-Location
}
