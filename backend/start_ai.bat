@echo off
REM Tong Mau - AI backend: tu dong cai dat (lan dau) va khoi dong server.
REM Chi can double-click file nay. Lan sau chay lai se nhanh vi khong cai lai.
setlocal enabledelayedexpansion
cd /d "%~dp0"

call :progress 0 "Bat dau kiem tra moi truong..."

REM "where python" khong du tin cay: Windows co san mot shortcut gia
REM (App Execution Alias) tro toi Microsoft Store, "where" van tim thay no
REM du chua cai Python that. Kiem tra bang cach doc that "python --version".
set PYCHECK=%TEMP%\tongmau_pycheck.txt
python --version >"%PYCHECK%" 2>&1
findstr /b /c:"Python " "%PYCHECK%" >nul
if errorlevel 1 (
    echo Python chua duoc cai that su tren may nay ^(chi thay shortcut cua Microsoft Store^).
    where winget >nul 2>nul
    if errorlevel 1 (
        echo [Loi] May chua co winget nen khong tu cai duoc. Vui long tu tai va cai Python 3.12
        echo ^(ban tuong thich voi project nay^) tai link sau:
        echo   https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe
        echo Nho tick "Add python.exe to PATH" khi cai ^(o man hinh dau tien^), roi chay lai file nay.
        echo ^(Trang tai chinh thuc, neu muon chon ban khac: https://www.python.org/downloads/^)
        del "%PYCHECK%" >nul 2>nul
        pause
        exit /b 1
    ) else (
        echo Dang tu dong cai Python 3.12 qua winget, vui long doi...
        winget install --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
        del "%PYCHECK%" >nul 2>nul
        echo.
        echo Da cai xong Python. Dong cua so nay va chay lai file start_ai.bat mot lan nua
        echo ^(can mo cua so moi de nhan duoc PATH vua cap nhat^).
        pause
        exit /b 0
    )
)
del "%PYCHECK%" >nul 2>nul
call :progress 15 "Da xac nhan Python that su tren may."

if exist venv (
    venv\Scripts\python.exe --version >nul 2>nul
    if errorlevel 1 (
        echo Moi truong ao ^(venv^) cu bi hong ^(thuong do Python goc da bi go hoac cai lai o vi tri khac^).
        echo Dang xoa venv cu va tao lai bang Python hien tai tren may...
        rmdir /s /q venv
    )
)
if not exist venv (
    echo Dang tao moi truong Python rieng cho project...
    python -m venv venv
)
call :progress 25 "Moi truong Python rieng (venv) da san sang."

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

    call :progress 30 "Dang cai torch/torchvision - buoc nang nhat, co the mat vai phut..."
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
    call :progress 65 "Da cai xong torch/torchvision."

    call :progress 70 "Dang cai cac thu vien con lai: fastapi, uvicorn, Pillow, numpy..."
    venv\Scripts\python.exe -m pip install -r requirements.txt --timeout 180 --retries 5
    if errorlevel 1 (
        echo.
        echo [Loi] Cai thu vien that bai ^(xem loi chi tiet o tren^). Kiem tra ket noi mang roi chay lai file nay.
        pause
        exit /b 1
    )
    call :progress 90 "Cai dat xong."

    echo.
    echo Cai dat xong.
)

call :progress 100 "San sang!"
echo.
echo Dang khoi dong server tai http://localhost:8000 ...
echo Mo file frontend\index.html bang trinh duyet, chuyen sang tab "AI nang cao" de su dung.
echo ^(Dong cua so nay hoac nhan Ctrl+C de tat server^)
echo.
venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
pause
exit /b 0

:progress
REM Ve thanh tien do dang van ban. Tham so: %1 = phan tram (0-100), %2 = thong diep.
setlocal enabledelayedexpansion
set /a _pct=%~1
if !_pct! lss 0 set _pct=0
if !_pct! gtr 100 set _pct=100
set /a _filled=_pct/5
set "_bar="
for /l %%i in (1,1,20) do (
    if %%i leq !_filled! (set "_bar=!_bar!#") else (set "_bar=!_bar!-")
)
echo [!_bar!] !_pct!%%  %~2
endlocal
goto :eof
