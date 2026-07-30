# Windows shim for the Makefile targets - GNU make is not typically installed on Windows.
# The Makefile stays canonical; this file must run the same commands.
#
#   ./tasks.ps1 install | check | doctor | clean | kaggle-bundle

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('install', 'check', 'doctor', 'clean', 'kaggle-bundle')]
    [string]$Target
)

$ErrorActionPreference = 'Stop'

# Use the project venv without requiring the caller to activate it first, so `check` works
# immediately after `install`. A shell that already activated .venv is unaffected.
$venvScripts = Join-Path $PSScriptRoot '.venv\Scripts'
if (Test-Path $venvScripts) {
    $env:PATH = "$venvScripts;$env:PATH"
}

function Invoke-Step([string]$Label, [scriptblock]$Step) {
    Write-Host "==> $Label" -ForegroundColor Cyan
    & $Step
    if ($LASTEXITCODE -ne 0) {
        Write-Host "$Label failed (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

switch ($Target) {
    'install' {
        Invoke-Step 'uv venv .venv' { uv venv .venv }
        Invoke-Step 'uv pip install -e ".[dev]"' { uv pip install -e ".[dev]" }
    }
    'check' {
        Invoke-Step 'ruff check' { ruff check . }
        Invoke-Step 'ruff format --check' { ruff format --check . }
        Invoke-Step 'pytest' { pytest }
    }
    'doctor' {
        Invoke-Step 'mcpr doctor' { mcpr doctor }
    }
    'clean' {
        foreach ($d in '.pytest_cache', '.ruff_cache', 'dist', 'build') {
            if (Test-Path $d) { Remove-Item -Recurse -Force $d }
        }
        Get-ChildItem -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force
        Write-Host 'clean: done' -ForegroundColor Green
    }
    'kaggle-bundle' {
        [Console]::Error.WriteLine('kaggle-bundle: implemented in F6')
        exit 1
    }
}

Write-Host "${Target}: ok" -ForegroundColor Green
