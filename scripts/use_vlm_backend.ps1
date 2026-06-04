param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("lmstudio", "llama", "status", "stop-all")]
    [string]$Backend,

    [switch]$AllowOtherBackend,
    [int]$LmStudioPort = 1234,
    [int]$LlamaPort = 18080,
    [string]$LlamaDir = "C:\llama"
)

$ErrorActionPreference = "Stop"

function Get-VlmProcesses {
    $items = @()
    try {
        $items = @(Get-CimInstance Win32_Process -Filter "Name='llama-server.exe' OR Name='LM Studio.exe' OR Name='lms.exe'")
    } catch {
        $items = @()
    }

    foreach ($item in $items) {
        $name = [string]$item.Name
        $path = [string]$item.ExecutablePath
        $cmd = [string]$item.CommandLine
        $kind = "other"

        if ($name -eq "LM Studio.exe" -or $name -eq "lms.exe") {
            $kind = "lmstudio"
        } elseif ($name -eq "llama-server.exe") {
            if ($path -like "$LlamaDir*") {
                $kind = "standalone-llama"
            } elseif ($path -like "$env:USERPROFILE\.lmstudio\extensions\backends\*") {
                $kind = "lmstudio-worker"
            } elseif ($cmd -match "--image-max-tokens") {
                $kind = "standalone-llama"
            } else {
                $kind = "other-llama"
            }
        }

        [pscustomobject]@{
            Kind = $kind
            Name = $name
            ProcessId = [int]$item.ProcessId
            Path = $path
            CommandLine = $cmd
        }
    }
}

function Stop-VlmProcesses {
    param([string[]]$Kinds)

    $targets = @(Get-VlmProcesses | Where-Object { $Kinds -contains $_.Kind })
    foreach ($target in $targets) {
        Write-Host "Stopping $($target.Kind): $($target.Name) pid=$($target.ProcessId)"
        Stop-Process -Id $target.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

function Show-VlmStatus {
    $processes = @(Get-VlmProcesses)
    if ($processes.Count -eq 0) {
        Write-Host "No LM Studio / llama-server process found."
    } else {
        $processes |
            Select-Object Kind, Name, ProcessId, Path |
            Format-Table -AutoSize
    }

    foreach ($entry in @(
        @{ Name = "LM Studio"; Url = "http://127.0.0.1:$LmStudioPort/v1/models" },
        @{ Name = "standalone llama-server"; Url = "http://127.0.0.1:$LlamaPort/v1/models" }
    )) {
        try {
            $data = Invoke-RestMethod -Uri $entry.Url -TimeoutSec 3
            $models = @($data.data | ForEach-Object { $_.id }) -join ", "
            Write-Host "$($entry.Name): OK $($entry.Url) [$models]"
        } catch {
            Write-Host "$($entry.Name): not reachable $($entry.Url)"
        }
    }
}

function Start-LmStudioServer {
    $lms = Get-Command lms -ErrorAction SilentlyContinue
    if ($lms) {
        Write-Host "Starting LM Studio server on port $LmStudioPort..."
        & $lms.Source server start --port $LmStudioPort
        return
    }

    $app = "C:\Program Files\LM Studio\LM Studio.exe"
    if (Test-Path -LiteralPath $app) {
        Write-Host "Starting LM Studio app. Enable Local Server on port $LmStudioPort if it is not already active."
        Start-Process -FilePath $app -WindowStyle Hidden
        return
    }

    throw "LM Studio CLI/app bulunamadi. LM Studio'yu elle acin veya PATH'e lms ekleyin."
}

function Start-StandaloneLlama {
    $script = Join-Path $PSScriptRoot "start_llama_server_gemma4.ps1"
    if (-not (Test-Path -LiteralPath $script)) {
        throw "Launcher bulunamadi: $script"
    }
    Write-Host "Starting standalone llama-server on port $LlamaPort..."
    & $script -LlamaDir $LlamaDir -Port $LlamaPort
}

switch ($Backend) {
    "status" {
        Show-VlmStatus
    }
    "stop-all" {
        Stop-VlmProcesses -Kinds @("lmstudio", "lmstudio-worker", "standalone-llama")
        Show-VlmStatus
    }
    "lmstudio" {
        if (-not $AllowOtherBackend) {
            Stop-VlmProcesses -Kinds @("standalone-llama")
        }
        Start-LmStudioServer
        Show-VlmStatus
    }
    "llama" {
        if (-not $AllowOtherBackend) {
            Stop-VlmProcesses -Kinds @("lmstudio", "lmstudio-worker")
        }
        Start-StandaloneLlama
    }
}
