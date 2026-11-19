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
set(CMAKE_C_COMPILER   ${OHOS_LLVM}/clang)
set(CMAKE_CXX_COMPILER ${OHOS_LLVM}/clang++)
set(CMAKE_AR           ${OHOS_LLVM}/llvm-ar)
set(CMAKE_RANLIB       ${OHOS_LLVM}/llvm-ranlib)
set(CMAKE_STRIP        ${OHOS_LLVM}/llvm-strip)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
