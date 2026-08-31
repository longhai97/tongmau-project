@echo off
REM Tong Mau - AI backend: tu dong cai dat (lan dau) va khoi dong server.
REM Chi can double-click file nay. Lan sau chay lai se nhanh vi khong cai lai.
setlocal enabledelayedexpansion
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [Loi] Chua cai Python.
    echo Vui long cai tai https://www.python.org/downloads/
    echo Nho tick "Add python.exe to PATH" khi cai, roi chay lai file nay.
    pause
    exit /b 1
)

if not exist venv (
    echo Dang tao moi truong Python rieng cho project...
    python -m venv venv
)

if not exist venv\Lib\site-packages\torch (
    echo.
    echo ==================================================================
    echo  Lan dau chay: dang cai cac thu vien can thiet ^(torch, fastapi...^)
    echo  Co the mat vai phut tuy toc do mang, chi can cai 1 lan.
    echo ==================================================================
    echo.

    where nvidia-smi >nul 2>nul
    if errorlevel 1 (
        echo Khong tim thay GPU NVIDIA - se cai ban CPU ^(chay duoc nhung cham hon nhieu^).
        set TORCH_ARGS=torch torchvision
    ) else (
        echo Da tim thay GPU NVIDIA - se cai ban CUDA de dung GPU tang toc.
        set TORCH_ARGS=torch torchvision --index-url https://download.pytorch.org/whl/cu121
    )

    REM --timeout/--retries: file torch rat nang (~2-3GB), mang cham/chap
    REM chon se bi ReadTimeoutError neu dung mac dinh cua pip (thuong gap).
    venv\Scripts\python.exe -m pip install !TORCH_ARGS! --timeout 180 --retries 5
    if errorlevel 1 (
        echo.
        echo [Loi] Cai torch that bai ^(xem loi chi tiet o tren^).
        echo Nguyen nhan thuong gap: mang cham/chap chon khi tai file torch ^(~2-3GB^),
        echo firewall/antivirus chan, hoac mat ket noi giua chung.
        echo Thu lai: chay lai file nay ^(da tang timeout + tu dong thu lai 5 lan^).
        echo Neu van loi, thu doi sang wifi/mang khac roi chay lai.
        pause
        exit /b 1
    )

    venv\Scripts\python.exe -m pip install -r requirements.txt --timeout 180 --retries 5
    if errorlevel 1 (
        echo.
        echo [Loi] Cai thu vien that bai ^(xem loi chi tiet o tren^). Kiem tra ket noi mang roi chay lai file nay.
        pause
        exit /b 1
    )

    echo.
    echo Cai dat xong.
)

echo.
echo Dang khoi dong server tai http://localhost:8000 ...
echo Mo file frontend\index.html bang trinh duyet, chuyen sang tab "AI nang cao" de su dung.
echo ^(Dong cua so nay hoac nhan Ctrl+C de tat server^)
echo.
venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
pause
