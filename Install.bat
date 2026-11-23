@echo off
setlocal enabledelayedexpansion
pushd "%~dp0"

:: 获取 HDC 设备列表
set "devices="
set /a count=0
for /f "tokens=*" %%i in ('hdc list targets') do (
    set "line=%%i"
    if not "!line!"=="" (
        if not "!line!"=="[Empty]" (
            set /a count+=1
            set "devices[!count!]=!line!"
        )
    )
)

if %count%==0 (
    echo 未检测到任何 HDC 设备，请连接设备后重试。
    pause
    exit /b 1
)

if %count%==1 (
    set "target=!devices[1]!"
    echo 检测到唯一设备: !target!，直接安装...
    call :install_to_device "!target!"
    goto :end
)

:: 多个设备，显示列表
echo 检测到多个 HDC 设备:
for /l %%i in (1,1,%count%) do (
    echo   %%i. !devices[%%i]!
)
echo.
set "selection="
set /p "selection=请输入要安装的设备序号（可多选，用空格或逗号分隔，例如: 1 3）: "

:: 解析用户选择
set "selected_devices="
for %%s in (%selection%) do (
    set "num=%%s"
    set "isnum=1"
    for /f "delims=0123456789" %%a in ("!num!") do set "isnum=0"
    if "!isnum!"=="1" (
        if !num! geq 1 if !num! leq %count% (
            set "selected_devices=!selected_devices! !devices[%%s]!"
        ) else (
            echo 警告: 序号 %%s 超出范围，已忽略。
        )
    ) else (
        echo 警告: 输入 "%%s" 不是有效数字，已忽略。
    )
)

if "!selected_devices!"=="" (
    echo 未选择任何有效设备，退出。
    pause
    exit /b 1
)

:: 对每个选中的设备执行安装
for %%d in (!selected_devices!) do (
    echo.
    echo ===== 开始安装到设备: %%d =====
    call :install_to_device "%%d"
)

goto :end

:install_to_device
set "target=%~1"
echo 正在操作设备: %target%
hdc -t %target% shell "bm uninstall -n com.ohos.smartsubtitle"
hdc -t %target% install SmartSubtitle.hap
hdc -t %target% shell "power-shell wakeup"
hdc -t %target% shell "uinput -T -m 300 800 280 200"
hdc -t %target% shell "aa start -a EntryAbility -b com.ohos.smartsubtitle"
exit /b

:end
popd
endlocal
pause
