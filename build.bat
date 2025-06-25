@echo off
REM AiCraft Build Script for Windows
REM Supports both CPU-only and CUDA builds

setlocal EnableDelayedExpansion

echo ╔════════════════════════════════════════════════════════════════╗
echo ║                     AiCraft Build System                      ║
echo ║            Ultra-Optimized Deep Learning Backend              ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

REM Configuration
set BUILD_TYPE=Release
set BUILD_DIR=build
set INSTALL_DIR=install
set ENABLE_CUDA=AUTO
set ENABLE_TESTS=OFF
set ENABLE_BENCHMARKS=OFF
set VERBOSE=OFF

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :done_parsing
if /I "%~1"=="--debug" set BUILD_TYPE=Debug
if /I "%~1"=="--release" set BUILD_TYPE=Release
if /I "%~1"=="--cuda" set ENABLE_CUDA=ON
if /I "%~1"=="--no-cuda" set ENABLE_CUDA=OFF
if /I "%~1"=="--tests" set ENABLE_TESTS=ON
if /I "%~1"=="--benchmarks" set ENABLE_BENCHMARKS=ON
if /I "%~1"=="--verbose" set VERBOSE=ON
if /I "%~1"=="--clean" goto :clean_build
if /I "%~1"=="--help" goto :show_help
shift
goto :parse_args
:done_parsing

echo [AiCraft] Build Configuration:
echo   Build Type: %BUILD_TYPE%
echo   CUDA Support: %ENABLE_CUDA%
echo   Tests: %ENABLE_TESTS%
echo   Benchmarks: %ENABLE_BENCHMARKS%
echo   Verbose: %VERBOSE%
echo.

REM Check for required tools
echo [AiCraft] Checking build requirements...

REM Check for CMake
cmake --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] CMake not found! Please install CMake and add it to PATH.
    echo Download from: https://cmake.org/download/
    exit /b 1
)
echo   ✓ CMake found

REM Check for Visual Studio Build Tools
where cl >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Visual Studio C++ compiler not found!
    echo Please install Visual Studio Build Tools or run from Developer Command Prompt.
    exit /b 1
)
echo   ✓ Visual Studio C++ compiler found

REM Check for CUDA (if requested)
if /I "%ENABLE_CUDA%"=="ON" (
    nvcc --version >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] CUDA compiler not found. Falling back to CPU-only build.
        set ENABLE_CUDA=OFF
    ) else (
        echo   ✓ CUDA compiler found
    )
) else if /I "%ENABLE_CUDA%"=="AUTO" (
    nvcc --version >nul 2>&1
    if errorlevel 1 (
        echo   ⚠ CUDA not detected - CPU-only build
        set ENABLE_CUDA=OFF
    ) else (
        echo   ✓ CUDA detected - GPU acceleration enabled
        set ENABLE_CUDA=ON
    )
)

echo.

REM Create build directory
if not exist "%BUILD_DIR%" (
    echo [AiCraft] Creating build directory...
    mkdir "%BUILD_DIR%"
)

REM Configure with CMake
echo [AiCraft] Configuring build...
cd "%BUILD_DIR%"

set CMAKE_ARGS=-DCMAKE_BUILD_TYPE=%BUILD_TYPE%
set CMAKE_ARGS=%CMAKE_ARGS% -DCMAKE_INSTALL_PREFIX=../%INSTALL_DIR%
set CMAKE_ARGS=%CMAKE_ARGS% -DBUILD_TESTS=%ENABLE_TESTS%
set CMAKE_ARGS=%CMAKE_ARGS% -DBUILD_BENCHMARKS=%ENABLE_BENCHMARKS%

if /I "%ENABLE_CUDA%"=="OFF" (
    set CMAKE_ARGS=%CMAKE_ARGS% -DCUDA_FOUND=FALSE
)

if /I "%VERBOSE%"=="ON" (
    echo [AiCraft] CMake command: cmake %CMAKE_ARGS% ..
)

cmake %CMAKE_ARGS% ..
if errorlevel 1 (
    echo [ERROR] CMake configuration failed!
    cd ..
    exit /b 1
)

echo   ✓ Configuration completed

REM Build the project
echo [AiCraft] Building AiCraft...

set BUILD_ARGS=--config %BUILD_TYPE%
if /I "%VERBOSE%"=="ON" (
    set BUILD_ARGS=%BUILD_ARGS% --verbose
)

cmake --build . %BUILD_ARGS%
if errorlevel 1 (
    echo [ERROR] Build failed!
    cd ..
    exit /b 1
)

echo   ✓ Build completed successfully

REM Install (optional)
echo [AiCraft] Installing AiCraft...
cmake --install . --config %BUILD_TYPE%
if errorlevel 1 (
    echo [WARNING] Installation failed, but build was successful
) else (
    echo   ✓ Installation completed
)

cd ..

REM Run tests (if enabled)
if /I "%ENABLE_TESTS%"=="ON" (
    echo [AiCraft] Running tests...
    cd "%BUILD_DIR%"
    ctest --config %BUILD_TYPE% --verbose
    if errorlevel 1 (
        echo [WARNING] Some tests failed
    ) else (
        echo   ✓ All tests passed
    )
    cd ..
)

REM Show build results
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║                       Build Completed!                        ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
echo [AiCraft] Build artifacts:
if exist "%BUILD_DIR%\bin\%BUILD_TYPE%\aicraft_demo.exe" (
    echo   ✓ Demo executable: %BUILD_DIR%\bin\%BUILD_TYPE%\aicraft_demo.exe
) else if exist "%BUILD_DIR%\bin\aicraft_demo.exe" (
    echo   ✓ Demo executable: %BUILD_DIR%\bin\aicraft_demo.exe
) else if exist "%BUILD_DIR%\%BUILD_TYPE%\aicraft_demo.exe" (
    echo   ✓ Demo executable: %BUILD_DIR%\%BUILD_TYPE%\aicraft_demo.exe
)

if exist "%BUILD_DIR%\%BUILD_TYPE%\aicraft.lib" (
    echo   ✓ Static library: %BUILD_DIR%\%BUILD_TYPE%\aicraft.lib
) else if exist "%BUILD_DIR%\aicraft.lib" (
    echo   ✓ Static library: %BUILD_DIR%\aicraft.lib
)

echo.
echo [AiCraft] To run the demo:
if exist "%BUILD_DIR%\bin\%BUILD_TYPE%\aicraft_demo.exe" (
    echo   %BUILD_DIR%\bin\%BUILD_TYPE%\aicraft_demo.exe
) else if exist "%BUILD_DIR%\bin\aicraft_demo.exe" (
    echo   %BUILD_DIR%\bin\aicraft_demo.exe
) else if exist "%BUILD_DIR%\%BUILD_TYPE%\aicraft_demo.exe" (
    echo   %BUILD_DIR%\%BUILD_TYPE%\aicraft_demo.exe
)

echo.
echo [AiCraft] Build system information:
echo   CPU Cores: %NUMBER_OF_PROCESSORS%
echo   Build Type: %BUILD_TYPE%
if /I "%ENABLE_CUDA%"=="ON" (
    echo   GPU Acceleration: Enabled
) else (
    echo   GPU Acceleration: Disabled
)
echo   Installation Directory: %INSTALL_DIR%

goto :end

:clean_build
echo [AiCraft] Cleaning build artifacts...
if exist "%BUILD_DIR%" (
    rmdir /s /q "%BUILD_DIR%"
    echo   ✓ Build directory cleaned
)
if exist "%INSTALL_DIR%" (
    rmdir /s /q "%INSTALL_DIR%"
    echo   ✓ Install directory cleaned
)
echo [AiCraft] Clean completed
goto :end

:show_help
echo AiCraft Build Script Help
echo.
echo Usage: build.bat [options]
echo.
echo Options:
echo   --debug       Build in Debug mode (default: Release)
echo   --release     Build in Release mode
echo   --cuda        Force enable CUDA support
echo   --no-cuda     Force disable CUDA support
echo   --tests       Enable building tests
echo   --benchmarks  Enable building benchmarks
echo   --verbose     Enable verbose output
echo   --clean       Clean build artifacts
echo   --help        Show this help message
echo.
echo Examples:
echo   build.bat                    # Default Release build with auto CUDA detection
echo   build.bat --debug --tests   # Debug build with tests
echo   build.bat --cuda --verbose  # Force CUDA build with verbose output
echo   build.bat --clean           # Clean all build artifacts
goto :end

:end
endlocal
echo.
pause
