# CMake toolchain for aarch64 OpenHarmony targets, using the clang toolchain
# bundled with the OpenHarmony SDK (API 18).
#
# Usage:
#   export OHOS_LLVM=~/tools/openharmony/Sdk/18/native/llvm/bin
#   cmake -DCMAKE_TOOLCHAIN_FILE=$(pwd)/toolchain.cmake ..
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

if(NOT DEFINED ENV{OHOS_LLVM})
    message(FATAL_ERROR "export OHOS_LLVM=~/tools/openharmony/Sdk/18/native/llvm/bin first")
endif()

set(OHOS_LLVM $ENV{OHOS_LLVM})
# 使用 SDK 自带的 aarch64-ohos 目标封装器: 自动设置 --target 并定位
# OHOS sysroot(裸 clang 默认目标是宿主机 x86, 会链到错误的运行库)
set(CMAKE_C_COMPILER   ${OHOS_LLVM}/aarch64-unknown-linux-ohos-clang)
set(CMAKE_CXX_COMPILER ${OHOS_LLVM}/aarch64-unknown-linux-ohos-clang++)
set(CMAKE_AR           ${OHOS_LLVM}/llvm-ar)
set(CMAKE_RANLIB       ${OHOS_LLVM}/llvm-ranlib)
set(CMAKE_STRIP        ${OHOS_LLVM}/llvm-strip)

# OHOS aarch64 sysroot(SDK native 目录下)
if(EXISTS "${OHOS_LLVM}/../../sysroot")
    set(CMAKE_SYSROOT "${OHOS_LLVM}/../../sysroot")
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
