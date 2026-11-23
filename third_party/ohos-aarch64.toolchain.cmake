# CMake toolchain: cross-compile sherpa-onnx for OpenHarmony aarch64
# using the clang bundled with the OpenHarmony SDK (API 18).
#
# Honors OHOS_LLVM (path to the llvm bin directory) and optional
# OHOS_SYSROOT (defaults to <native>/sysroot).
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

if(NOT DEFINED ENV{OHOS_LLVM})
    message(FATAL_ERROR "export OHOS_LLVM=~/tools/openharmony/Sdk/18/native/llvm/bin first")
endif()

set(OHOS_LLVM $ENV{OHOS_LLVM})
# Use the SDK target wrappers: they set the aarch64-linux-ohos triple and
# know the OHOS sysroot layout (crt objects, libc, libunwind).
set(CMAKE_C_COMPILER   ${OHOS_LLVM}/aarch64-unknown-linux-ohos-clang)
set(CMAKE_CXX_COMPILER ${OHOS_LLVM}/aarch64-unknown-linux-ohos-clang++)
set(CMAKE_AR           ${OHOS_LLVM}/llvm-ar)
set(CMAKE_RANLIB       ${OHOS_LLVM}/llvm-ranlib)
set(CMAKE_STRIP        ${OHOS_LLVM}/llvm-strip)

# OpenHarmony aarch64 sysroot (ships with the SDK native toolchain).
if(DEFINED ENV{OHOS_SYSROOT})
    set(OHOS_SYSROOT $ENV{OHOS_SYSROOT})
else()
    set(OHOS_SYSROOT ${OHOS_LLVM}/../../sysroot)
endif()
if(EXISTS ${OHOS_SYSROOT})
    set(CMAKE_SYSROOT ${OHOS_SYSROOT})
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# C++ runtime (libc++) lives in <llvm>/lib/aarch64-linux-ohos; append it at
# the END of the link line (CMAKE_*_STANDARD_LIBRARIES) so the static
# onnxruntime archive can resolve libc++ symbols.
set(OHOS_CXX_LIB_DIR ${OHOS_LLVM}/../lib/aarch64-linux-ohos)
set(CMAKE_CXX_STANDARD_LIBRARIES "-L${OHOS_CXX_LIB_DIR} -lc++ ${CMAKE_CXX_STANDARD_LIBRARIES}")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-L${OHOS_CXX_LIB_DIR} -lc++")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-L${OHOS_CXX_LIB_DIR} -lc++")
