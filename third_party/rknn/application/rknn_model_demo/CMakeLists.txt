cmake_minimum_required(VERSION 3.6)

project(rknn_model_demo)

# 构建时通过 -DCMAKE_TOOLCHAIN_FILE 指定 OpenHarmony clang 工具链
set(RKNN_INCLUDE ${CMAKE_SOURCE_DIR}/../../include)
set(RKNN_RT_LIB ${CMAKE_SOURCE_DIR}/../../lib/arm64-v8a/librknnrt.so)

if(NOT EXISTS ${RKNN_RT_LIB})
    message(FATAL_ERROR "librknnrt.so not found at ${RKNN_RT_LIB}")
endif()
include_directories(${RKNN_INCLUDE})

add_executable(rknn_model_demo
    src/main.cc
)

target_link_libraries(rknn_model_demo
    ${RKNN_RT_LIB}
)

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)
install(TARGETS rknn_model_demo DESTINATION ./)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
