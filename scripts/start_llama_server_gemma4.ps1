param(
    [string]$LlamaDir = "C:\llama",
    [string]$ModelPath = "$env:USERPROFILE\.lmstudio\models\llmfan46\gemma-4-26B-A4B-it-uncensored-heretic-GGUF\gemma-4-26B-A4B-it-uncensored-heretic-Q3_K_M.gguf",
    [string]$MmprojPath = "$env:USERPROFILE\.lmstudio\models\llmfan46\gemma-4-26B-A4B-it-uncensored-heretic-GGUF\gemma-4-26B-A4B-it-mmproj-BF16.gguf",
    [int]$GpuLayers = 24,
    [int]$CtxSize = 8192,
    [int]$ImageTokens = 1120,
    [int]$BatchSize = 2048,
    [int]$UBatchSize = 2048,
    [int]$Parallel = 1,
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8080,
    [string]$Alias = "gemma-4-26b-a4b-it-uncensored-heretic",
    [switch]$NoFlashAttn
)

$ErrorActionPreference = "Stop"

$serverExe = Join-Path $LlamaDir "llama-server.exe"
if (-not (Test-Path -LiteralPath $serverExe)) {
    throw "llama-server.exe bulunamadi: $serverExe"
}
if (-not (Test-Path -LiteralPath $ModelPath)) {
    throw "Model GGUF bulunamadi: $ModelPath"
}

$helpText = (& $serverExe --help 2>&1 | Out-String)
if ($helpText -notmatch "--image-max-tokens" -or $helpText -notmatch "--image-min-tokens") {
    throw "Bu llama-server surumu --image-min-tokens/--image-max-tokens desteklemiyor. Daha yeni llama.cpp paketi kurun."
}

$serverArgs = @(
    "-m", $ModelPath,
    "--ctx-size", $CtxSize,
    "-ngl", $GpuLayers,
    "--image-min-tokens", $ImageTokens,
    "--image-max-tokens", $ImageTokens,
    "--batch-size", $BatchSize,
    "--ubatch-size", $UBatchSize,
    "--parallel", $Parallel,
    "--alias", $Alias,
    "--host", $HostAddress,
    "--port", $Port
)

if ($MmprojPath -and (Test-Path -LiteralPath $MmprojPath)) {
    $serverArgs = @("-m", $ModelPath, "--mmproj", $MmprojPath) + $serverArgs[2..($serverArgs.Count - 1)]
} elseif ($MmprojPath) {
    Write-Warning "mmproj bulunamadi, --mmproj olmadan baslatiliyor: $MmprojPath"
}

if (-not $NoFlashAttn) {
    $serverArgs += @("--flash-attn", "on")
}

Write-Host "Starting llama-server:"
Write-Host "  $serverExe"
Write-Host "OpenAI base URL: http://$HostAddress`:$Port/v1"
Write-Host "Image tokens: min=$ImageTokens max=$ImageTokens"
Write-Host "Stop with Ctrl+C."

& $serverExe @serverArgs
