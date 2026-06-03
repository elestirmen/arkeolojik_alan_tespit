@echo off
setlocal
chcp 65001 >nul

REM ============================================================
REM  Gemma 4 26B-A4B icin llama-server baslatici (.bat)
REM  Goruntu token butcesi 1120 (kucuk-nesne tespiti) ile baslatir.
REM  OpenAI uyumlu endpoint: http://127.0.0.1:8080/v1
REM
REM  Kullanim:
REM    start_llama_server_gemma4.bat            -> 1120 token, port 8080
REM    start_llama_server_gemma4.bat 560        -> ImageTokens=560
REM    start_llama_server_gemma4.bat 1120 8080  -> ImageTokens + Port
REM
REM  NOT: Once LM Studio modelini unload edin / LM Studio'yu kapatin;
REM       16GB VRAM ikisine birden yetmez.
REM ============================================================

REM ---- Ayarlanabilir degiskenler ----
set "LLAMA_DIR=C:\llama"
set "MODEL=%USERPROFILE%\.lmstudio\models\llmfan46\gemma-4-26B-A4B-it-uncensored-heretic-GGUF\gemma-4-26B-A4B-it-uncensored-heretic-Q3_K_M.gguf"
set "MMPROJ=%USERPROFILE%\.lmstudio\models\llmfan46\gemma-4-26B-A4B-it-uncensored-heretic-GGUF\gemma-4-26B-A4B-it-mmproj-BF16.gguf"
REM GPU_LAYERS=24 bu modelde optimal: MoE (3.8B aktif) oldugu icin daha
REM yuksek -ngl hizi artirmaz ama VRAM'i tuketir. 16GB'de 24 + parallel 1
REM ~1.7GB bos birakir (guvenli). VRAM bolca varsa artirabilirsiniz.
set "GPU_LAYERS=24"
set "CTX_SIZE=8192"
set "BATCH=2048"
set "UBATCH=2048"
REM PARALLEL=1: pipeline tek tek istek gonderir; 4 (varsayilan) bosa KV VRAM yer.
set "PARALLEL=1"
set "HOST=127.0.0.1"
set "ALIAS=gemma-4-26b-a4b-it-uncensored-heretic"

REM ---- Opsiyonel arg override (1=ImageTokens, 2=Port) ----
set "IMAGE_TOKENS=%~1"
if "%IMAGE_TOKENS%"=="" set "IMAGE_TOKENS=1120"
set "PORT=%~2"
if "%PORT%"=="" set "PORT=8080"

set "SERVER=%LLAMA_DIR%\llama-server.exe"

REM ---- On kontroller ----
if not exist "%SERVER%" (
    echo HATA: llama-server.exe bulunamadi: %SERVER%
    echo LLAMA_DIR degiskenini bu dosyada duzenleyin.
    pause
    exit /b 1
)
if not exist "%MODEL%" (
    echo HATA: Model GGUF bulunamadi: %MODEL%
    pause
    exit /b 1
)

REM ---- llama-server bu bayraklari destekliyor mu? ----
"%SERVER%" --help 2>&1 | find "--image-max-tokens" >nul
if errorlevel 1 (
    echo HATA: Bu llama-server surumu --image-min/max-tokens desteklemiyor.
    echo Daha yeni bir llama.cpp paketi kurun.
    pause
    exit /b 1
)

echo ============================================================
echo  llama-server baslatiliyor
echo  Model        : %MODEL%
echo  Image tokens : %IMAGE_TOKENS% (min=max)
echo  Batch/UBatch : %BATCH% / %UBATCH%
echo  GPU layers   : %GPU_LAYERS%
echo  OpenAI URL   : http://%HOST%:%PORT%/v1
echo  Durdurmak    : Ctrl+C
echo ============================================================

if exist "%MMPROJ%" (
    "%SERVER%" -m "%MODEL%" --mmproj "%MMPROJ%" --ctx-size %CTX_SIZE% -ngl %GPU_LAYERS% --image-min-tokens %IMAGE_TOKENS% --image-max-tokens %IMAGE_TOKENS% --batch-size %BATCH% --ubatch-size %UBATCH% --flash-attn on --parallel %PARALLEL% --alias "%ALIAS%" --host %HOST% --port %PORT%
) else (
    echo UYARI: mmproj bulunamadi, vision olmadan baslatiliyor: %MMPROJ%
    "%SERVER%" -m "%MODEL%" --ctx-size %CTX_SIZE% -ngl %GPU_LAYERS% --image-min-tokens %IMAGE_TOKENS% --image-max-tokens %IMAGE_TOKENS% --batch-size %BATCH% --ubatch-size %UBATCH% --flash-attn on --parallel %PARALLEL% --alias "%ALIAS%" --host %HOST% --port %PORT%
)

echo.
echo llama-server kapandi.
pause
endlocal
