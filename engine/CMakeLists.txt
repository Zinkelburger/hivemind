cmake_minimum_required(VERSION 3.16)

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

project(hivemind)

# Defaults to tensorrt so existing build commands are unaffected; the
# onnxruntime backend is opt-in and needs no CUDA.
set(HIVEMIND_BACKEND "tensorrt" CACHE STRING
    "Inference backend: tensorrt (NVIDIA, FP16) or onnxruntime (portable, FP32)")
set_property(CACHE HIVEMIND_BACKEND PROPERTY STRINGS onnxruntime tensorrt)
option(HIVEMIND_FAST_BUILD "Favor faster local builds over maximum runtime optimization" ON)
option(HIVEMIND_USE_UNITY_BUILD "Enable unity builds for project targets" ON)
option(HIVEMIND_USE_CCACHE "Use ccache when available" ON)
option(HIVEMIND_NATIVE_ARCH "Optimize for the build machine's CPU" ON)

# `-march=native` is the right default for someone building the engine for the
# machine they are sitting at, and exactly wrong for anything redistributable:
# it bakes in whatever the build host happened to support, so the binary dies
# with SIGILL on an older CPU. Redistributable builds set an explicit baseline
# here instead, which wins over HIVEMIND_NATIVE_ARCH when both are given.
set(HIVEMIND_ARCH "" CACHE STRING
    "Explicit architecture baseline for redistributable builds (e.g. x86-64-v2); empty means follow HIVEMIND_NATIVE_ARCH")

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(HIVEMIND_USE_CCACHE)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif()
endif()

# Fast local builds trade some runtime optimization for much shorter compile/link times.
if(MSVC)
    # Kept building-but-untested: the tree uses __builtin_ctzll / __builtin_bswap64,
    # which MSVC does not provide. Build Windows binaries with mingw-w64 or
    # clang-cl, both of which take the GNU flags below.
    set(HIVEMIND_COMMON_FLAGS "/W3 /DNDEBUG /DNNUE_EMBEDDING_OFF")
    if(HIVEMIND_FAST_BUILD)
        string(APPEND HIVEMIND_COMMON_FLAGS " /O2")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)
    else()
        string(APPEND HIVEMIND_COMMON_FLAGS " /O2 /GL")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
else()
    set(HIVEMIND_COMMON_FLAGS "-Wall -DNDEBUG -DNNUE_EMBEDDING_OFF")

    # Which architecture is actually being produced. Cross-compiling to x86_64
    # on an Apple Silicon host leaves CMAKE_SYSTEM_PROCESSOR at arm64, so on
    # Apple the answer has to come from CMAKE_OSX_ARCHITECTURES when it is set.
    set(HIVEMIND_TARGET_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
    if(APPLE AND CMAKE_OSX_ARCHITECTURES)
        list(GET CMAKE_OSX_ARCHITECTURES 0 HIVEMIND_TARGET_ARCH)
    endif()

    # aarch64 compilers have no -march; they spell the same idea -mcpu.
    if(HIVEMIND_TARGET_ARCH MATCHES "^(arm64|aarch64)$")
        set(HIVEMIND_ARCH_FLAG "-mcpu")
    else()
        set(HIVEMIND_ARCH_FLAG "-march")
    endif()

    if(HIVEMIND_ARCH)
        string(APPEND HIVEMIND_COMMON_FLAGS " ${HIVEMIND_ARCH_FLAG}=${HIVEMIND_ARCH}")
    elseif(HIVEMIND_NATIVE_ARCH)
        string(APPEND HIVEMIND_COMMON_FLAGS " ${HIVEMIND_ARCH_FLAG}=native")
    endif()

    if(HIVEMIND_FAST_BUILD)
        string(APPEND HIVEMIND_COMMON_FLAGS " -O2")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)
    else()
        string(APPEND HIVEMIND_COMMON_FLAGS " -Ofast -flto")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIVEMIND_COMMON_FLAGS}")

# A redistributable binary cannot rely on the target having the libstdc++ the
# build machine linked against; folding both runtimes in leaves glibc as the
# only version floor that matters. GNU-flavoured toolchains only -- Apple's
# clang has no static libc++ to link.
option(HIVEMIND_STATIC_CXX_RUNTIME "Link libstdc++/libgcc statically" OFF)
if(HIVEMIND_STATIC_CXX_RUNTIME)
    if(APPLE OR MSVC)
        message(WARNING "HIVEMIND_STATIC_CXX_RUNTIME is a no-op on this toolchain")
    else()
        string(APPEND CMAKE_EXE_LINKER_FLAGS " -static-libstdc++ -static-libgcc")
        if(WIN32)
            # mingw-w64 threads pull in libwinpthread otherwise.
            string(APPEND CMAKE_EXE_LINKER_FLAGS " -static")
        endif()
    endif()
endif()

# Add source files
set(LIB_SOURCES
    src/common/globals.cc
    src/environment/board.cc
    src/environment/planes.cc
    src/environment/zobrist.cc
    # src/nn/engine.cc or engine_ort.cc is appended below per backend
    src/nn/onnx_utils.cc
    src/search/agent.cc
    src/search/node.cc
    src/search/searchthread.cc
    src/interface/uci.cc
    src/tools/benchmark.cc
    src/tools/selfplay.cc
    src/tools/tournament.cc
)

if(HIVEMIND_BACKEND STREQUAL "tensorrt")
    list(APPEND LIB_SOURCES src/nn/engine.cc)

    # For finding FindTensorRT.cmake
    set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

    if(NOT TensorRT_DIR)
        if(DEFINED ENV{TensorRT_DIR})
            set(TensorRT_DIR $ENV{TensorRT_DIR})
        else()
            set(TensorRT_DIR /usr/src/tensorrt/)
        endif()
    endif()
    if(NOT CUDA_TOOLKIT_ROOT_DIR)
        set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
    endif()

    find_package(TensorRT REQUIRED)
    find_package(CUDA REQUIRED)

    set(CMAKE_BUILD_RPATH "${TensorRT_DIR}/lib;${CUDA_TOOLKIT_ROOT_DIR}/lib64;${CUDA_TOOLKIT_ROOT_DIR}/lib")
    set(CMAKE_INSTALL_RPATH "${TensorRT_DIR}/lib;${CUDA_TOOLKIT_ROOT_DIR}/lib64;${CUDA_TOOLKIT_ROOT_DIR}/lib")
elseif(HIVEMIND_BACKEND STREQUAL "onnxruntime")
    list(APPEND LIB_SOURCES src/nn/engine_ort.cc)

    # Prefer a vendored runtime (third_party/onnxruntime), fall back to the system.
    if(NOT ONNXRuntime_ROOT)
        if(DEFINED ENV{ONNXRuntime_ROOT})
            set(ONNXRuntime_ROOT $ENV{ONNXRuntime_ROOT})
        elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime")
            set(ONNXRuntime_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime")
        endif()
    endif()

    find_path(ONNXRuntime_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS ${ONNXRuntime_ROOT}/include
        PATH_SUFFIXES onnxruntime onnxruntime/core/session)
    find_library(ONNXRuntime_LIBRARY NAMES onnxruntime
        HINTS ${ONNXRuntime_ROOT}/lib)

    if(NOT ONNXRuntime_INCLUDE_DIR OR NOT ONNXRuntime_LIBRARY)
        message(FATAL_ERROR
            "ONNX Runtime not found. Install onnxruntime-devel, or download a "
            "release into third_party/onnxruntime, or pass -DONNXRuntime_ROOT=<dir>.")
    endif()
    message(STATUS "ONNX Runtime: ${ONNXRuntime_LIBRARY}")

    get_filename_component(ONNXRuntime_LIBDIR "${ONNXRuntime_LIBRARY}" DIRECTORY)

    # The shipped layout is the engine and the runtime side by side in one
    # directory, so the loader has to be told to look next to the binary. Both
    # RPATHs get it: CMAKE_BUILD_RPATH is what a binary copied straight out of
    # the build tree carries, and that is exactly how the release bundles are
    # assembled. Without it the only thing baked in is the build machine's
    # absolute third_party path, and the caller is forced to set
    # LD_LIBRARY_PATH / DYLD_LIBRARY_PATH to launch the engine at all.
    if(APPLE)
        set(HIVEMIND_ORIGIN "@loader_path")
    else()
        set(HIVEMIND_ORIGIN "$ORIGIN")
    endif()
    set(HIVEMIND_RPATH "${HIVEMIND_ORIGIN};${HIVEMIND_ORIGIN}/../lib;${ONNXRuntime_LIBDIR}")
    set(CMAKE_BUILD_RPATH "${HIVEMIND_RPATH}")
    set(CMAKE_INSTALL_RPATH "${HIVEMIND_RPATH}")
else()
    message(FATAL_ERROR "Unknown HIVEMIND_BACKEND '${HIVEMIND_BACKEND}'")
endif()

find_package(Threads REQUIRED)

add_subdirectory(src/Fairy-Stockfish)

if(HIVEMIND_USE_UNITY_BUILD)
    set_target_properties(Fairy-Stockfish PROPERTIES UNITY_BUILD ON UNITY_BUILD_BATCH_SIZE 8)
endif()

add_library(${PROJECT_NAME}_lib STATIC ${LIB_SOURCES})
if(HIVEMIND_BACKEND STREQUAL "tensorrt")
    target_include_directories(${PROJECT_NAME}_lib SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
    target_compile_definitions(${PROJECT_NAME}_lib PUBLIC HIVEMIND_BACKEND_TENSORRT)
else()
    target_include_directories(${PROJECT_NAME}_lib SYSTEM PUBLIC ${ONNXRuntime_INCLUDE_DIR})
    target_compile_definitions(${PROJECT_NAME}_lib PUBLIC HIVEMIND_BACKEND_ONNXRUNTIME)
endif()
target_include_directories(${PROJECT_NAME}_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(${PROJECT_NAME}_lib PUBLIC
    TENSORRT_LIBRARY_DIR="${TensorRT_DIR}/lib"
    HIVEMIND_WORKSPACE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    HIVEMIND_FP16_CONVERTER_SCRIPT="${CMAKE_CURRENT_SOURCE_DIR}/scripts/convert_onnx_fp16.py"
)
if(HIVEMIND_USE_UNITY_BUILD)
    set_target_properties(${PROJECT_NAME}_lib PROPERTIES UNITY_BUILD ON UNITY_BUILD_BATCH_SIZE 8)
endif()
if(HIVEMIND_BACKEND STREQUAL "tensorrt")
    target_link_libraries(${PROJECT_NAME}_lib PUBLIC ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES})
else()
    target_link_libraries(${PROJECT_NAME}_lib PUBLIC ${ONNXRuntime_LIBRARY})
endif()
target_link_libraries(${PROJECT_NAME}_lib PUBLIC ${CMAKE_THREAD_LIBS_INIT} Fairy-Stockfish Threads::Threads)
target_link_libraries(${PROJECT_NAME}_lib PUBLIC ${CMAKE_DL_LIBS})

add_executable(${PROJECT_NAME} src/main.cc)
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME hivemind.bin)
if(HIVEMIND_USE_UNITY_BUILD)
    set_target_properties(${PROJECT_NAME} PROPERTIES UNITY_BUILD ON UNITY_BUILD_BATCH_SIZE 8)
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_lib)

# Windows resolves DLLs from the executable's own directory and has no RPATH
# equivalent, so the runtime is copied rather than pointed at.
if(WIN32 AND HIVEMIND_BACKEND STREQUAL "onnxruntime")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRuntime_ROOT}/lib/onnxruntime.dll"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        COMMENT "Copying onnxruntime.dll next to hivemind.exe")
endif()

# The sh wrapper is a POSIX convenience; on Windows the .exe is the entry point.
if(NOT WIN32)
    if(HIVEMIND_BACKEND STREQUAL "tensorrt")
        file(WRITE "${CMAKE_BINARY_DIR}/hivemind"
        "#!/bin/sh\n"
        "set -e\n"
        "SCRIPT_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
        "export LD_LIBRARY_PATH=\"${TensorRT_DIR}/lib:${CUDA_TOOLKIT_ROOT_DIR}/lib64:${CUDA_TOOLKIT_ROOT_DIR}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\"\n"
        "exec \"$SCRIPT_DIR/hivemind.bin\" \"$@\"\n")
    else()
        # RPATH already points at the runtime; the wrapper just keeps the same entry point.
        file(WRITE "${CMAKE_BINARY_DIR}/hivemind"
        "#!/bin/sh\n"
        "set -e\n"
        "SCRIPT_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
        "exec \"$SCRIPT_DIR/hivemind.bin\" \"$@\"\n")
    endif()
    # chmod, not /bin/chmod: macOS and Linux agree on the name, not the path.
    execute_process(COMMAND chmod 755 "${CMAKE_BINARY_DIR}/hivemind")
endif()

include(CTest)
enable_testing()
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
