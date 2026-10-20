$ErrorActionPreference = 'Stop'
$exe = Join-Path $PSScriptRoot 'lumenfind.exe'
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "lumenfind.exe was not found beside this script."
}
Start-Process -FilePath $exe -ArgumentList 'uninstall-service' -Verb RunAs -WindowStyle Hidden -Wait

