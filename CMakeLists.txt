# GEMM Project CMake Configuration
# =================================
# This CMakeLists.txt file configures the build system for the GEMM project,
# including CUDA compilation, optimization flags, and documentation generation.
#
# Key Features:
# - CUDA architecture targeting for various GPU generations
# - Optimized NVCC compilation flags
# - Build configuration options (LOOPS_COUNT_AS_TEMPLATE_ARGS, etc.)
# - Doxygen documentation generation
# - Test framework integration
#
# Author: GEMM Project
# Date: 2025

cmake_minimum_required(VERSION 3.17)

# CMP0146 (CMake >= 3.27): the FindCUDA module is removed under NEW.
# We never call find_package(CUDA) ourselves, but OpenCV's config does when OpenCV is built
# with CUDA (OpenCV_CUDA ON); it also provides CUDA_INCLUDE_DIRS used below. Keep FindCUDA
# available (OLD) and silence the dev warning until OpenCV's config stops using it.
# The default also covers scripts that reset policies (OpenCV's config, Qt Creator's
# package-manager provider, which is where the warning is reported from).
set(CMAKE_POLICY_DEFAULT_CMP0146 OLD)
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

# CUDA Architecture Configuration
# Available options:
# 86: RTX 3090, 3080
# 89: RTX 4090
# 80: A100
# 90: Hopper H100, H200
# 100-virtual: Blackwell B200
# 120-virtual: Blackwell 5090
# A list is allowed: "89;90a" builds one binary for the 4090 and the H100. "-real" (e.g.
# "90a-real") ships machine code only, no PTX: use it for binaries that leave this machine
# (the H100 cluster), so the kernels cannot be read back as PTX. 90a (not 90) is Hopper with its
# architecture-specific features (wgmma, setmaxnreg); a 90a binary runs on H100 / H200 only.
set(CUDA_ARCH "89" CACHE STRING "CUDA architectures, e.g. 89 (RTX 4090), 90a-real (H100), \"89;90a-real\"")

message(STATUS "Build mode ${CMAKE_BUILD_TYPE}")
message(STATUS "Build arch ${CUDA_ARCH}")

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_RULE_MESSAGES ON)
set(CMAKE_OBJECT_PATH_MAX 1024)
set(CMAKE_EXPORT_COMPILE_COMMANDS true)

if(CUDA_ARCH STREQUAL "120-virtual")
	message(STATUS "Targeting Blackwell (PTX only)")

	unset(CMAKE_CUDA_ARCHITECTURES)
	# Set arch manually
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
	  -gencode arch=compute_120,code=compute_120 \
	  -Xcompiler -Wall")
else()
	set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
endif()
# for print_build_config(): "89;90a-real" -> "89+90a-real" (a ';' would split the definition)
string(REPLACE ";" "+" GEMM_CUDA_ARCH_TEXT "${CUDA_ARCH}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY         ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY         ${CMAKE_CURRENT_SOURCE_DIR}/lib/${CMAKE_BUILD_TYPE})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY         ${CMAKE_CURRENT_SOURCE_DIR}/lib/${CMAKE_BUILD_TYPE})

SET(CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS} "/NODEFAULTLIBS:LIBCMT")

set(CMAKE_CXX_STANDARD 17 CACHE STRING "Host compiler C++ version")
set(CMAKE_CUDA_STANDARD 17 CACHE STRING "CUDA compiler C++ version")

message(STATUS "CUDA standard " ${CMAKE_CUDA_STANDARD})

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Option for production builds (skip lineinfo for maximum performance)
option(PRODUCTION_CODE "Build for production (no debug symbols)" OFF)

# Configuration options for GEMM behavior
option(LOOPS_COUNT_AS_TEMPLATE_ARGS "Use loop counts as template arguments" ON)
option(REPORT_STATISTICS "Enable detailed statistics reporting" OFF)
option(L2_CACHE_POLICY "Enable L2 cache policy optimization" ON)
option(USE_NVTX "Enable NVIDIA Tools Extension (NVTX) for profiling" OFF)
option(FULL_UNROLL_ROW_LOOPS "Fully unroll the BFS row loops (old behaviour, ~6x slower bfs_caps.cu compile)" OFF)
option(LAST_USE_HINTS "BFS passes: last accesses of the encoder, decoder and forward sums as __ldlu / __stcs (leave L2 first)" OFF)
# leaf GEMM of the CAPS pipeline (BFS::batched_leaf()): cublas = batched cublasGemmEx,
# wgmma = our Hopper kernel (TMA + wgmma, include/wgmma_gemm.hpp; H100 / H200 only, needs 90a-real
# in CUDA_ARCH; R = 0 then runs on cuBLAS). (The former mma.sync leaf kernel: archive/mma/.)
set(LEAF_GEMM "cublas" CACHE STRING "Leaf GEMM of the CAPS pipeline: cublas or wgmma (Hopper)")
set_property(CACHE LEAF_GEMM PROPERTY STRINGS cublas wgmma)
if(LEAF_GEMM STREQUAL "mma") # a cache from before the mma.sync kernel moved to archive/mma/
    message(WARNING "LEAF_GEMM=mma is gone (archive/mma/): using cublas")
    set(LEAF_GEMM "cublas" CACHE STRING "Leaf GEMM of the CAPS pipeline: cublas or wgmma (Hopper)" FORCE)
endif()
if(NOT LEAF_GEMM MATCHES "^(cublas|wgmma)$")
    message(FATAL_ERROR "LEAF_GEMM must be cublas or wgmma, got '${LEAF_GEMM}'")
endif()
if(LEAF_GEMM STREQUAL "wgmma" AND NOT CUDA_ARCH MATCHES "90a")
    message(FATAL_ERROR "LEAF_GEMM=wgmma needs sm_90a code: add 90a-real to CUDA_ARCH (e.g. \"90a-real\" or \"89;90a-real\")")
endif()
# warps along y in a block of the BFS passes (bfs_kernel): blocks of
# 32 x BFS_WARPS_Y threads, each virtual block BFS_WARPS_Y rows high. A power of two (shift / mask
# index math), at most 32 (1024 threads). Not tuned yet; the loop counts of gen/tuning_table.hpp
# were tuned with 4, so retune them for another value.
set(BFS_WARPS_Y "4" CACHE STRING "Warps along y in a block of the BFS passes: 1, 2, 4, 8, 16 or 32")
set_property(CACHE BFS_WARPS_Y PROPERTY STRINGS 1 2 4 8 16 32)
if(NOT BFS_WARPS_Y MATCHES "^(1|2|4|8|16|32)$")
    message(FATAL_ERROR "BFS_WARPS_Y must be 1, 2, 4, 8, 16 or 32, got '${BFS_WARPS_Y}'")
endif()
# The L2 schedule / cache model and its viewer are Python tools now: py/ (see py/README.md).

# Production builds: force the options that must not differ from the benchmarked configuration.
# A normal variable shadows the cache entry, so the forced value is what the build (and the
# summary below) uses; the cache keeps what was requested and a message says it was overridden.
macro(gemm_production_force var value reason)
    if(${value})
        if(NOT ${var})
            message(STATUS "PRODUCTION_CODE: forcing ${var}=${value} (${reason})")
        endif()
    elseif(${var})
        message(STATUS "PRODUCTION_CODE: forcing ${var}=${value} (${reason})")
    endif()
    set(${var} ${value})
endmacro()

if(PRODUCTION_CODE)
    # -G/-O0 debug device code (or no optimisation flags at all for RelWithDebInfo/MinSizeRel,
    # see the flag setup below) contradicts a production build. Multi-config generators pick
    # the configuration at build time, so only single-config builds can be checked here.
    if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE STREQUAL "Release")
        message(FATAL_ERROR "PRODUCTION_CODE=ON requires CMAKE_BUILD_TYPE=Release (got '${CMAKE_BUILD_TYPE}')")
    endif()
    gemm_production_force(USE_NVTX OFF "profiling markers on every launch")
    gemm_production_force(REPORT_STATISTICS OFF "prints and extra occupancy queries in the launch path")
    gemm_production_force(LOOPS_COUNT_AS_TEMPLATE_ARGS ON "tuned kernel table; OFF is for tuning")
    gemm_production_force(FULL_UNROLL_ROW_LOOPS ON "configuration the benchmarks were built with")
    gemm_production_force(L2_CACHE_POLICY ON "benchmarked configuration")
endif()

# Display build configuration information
message(STATUS "")
message(STATUS "=== GEMM Build Configuration ===")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Production Code: ${PRODUCTION_CODE}")
message(STATUS "")
message(STATUS "Optimization Flags:")
message(STATUS "  LOOPS_COUNT_AS_TEMPLATE_ARGS: ${LOOPS_COUNT_AS_TEMPLATE_ARGS}")
message(STATUS "  L2_CACHE_POLICY: ${L2_CACHE_POLICY}")
message(STATUS "  REPORT_STATISTICS: ${REPORT_STATISTICS}")
message(STATUS "  USE_NVTX: ${USE_NVTX}")
message(STATUS "  FULL_UNROLL_ROW_LOOPS: ${FULL_UNROLL_ROW_LOOPS}")
message(STATUS "  LAST_USE_HINTS: ${LAST_USE_HINTS}")
message(STATUS "  LEAF_GEMM: ${LEAF_GEMM}")
message(STATUS "  BFS_WARPS_Y: ${BFS_WARPS_Y}")
message(STATUS "")
message(STATUS "Half Precision Options:")
message(STATUS "")
message(STATUS "CUDA Configuration:")
if(WIN32)
    message(STATUS "  Platform: Windows")
    message(STATUS "  Compiler: MSVC")
else()
    message(STATUS "  Platform: Linux")
    message(STATUS "  Compiler: GCC/Clang")
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(PRODUCTION_CODE)
        message(STATUS "  Optimization: Maximum Performance (Production)")
    else()
        message(STATUS "  Optimization: High Performance (Development)")
    endif()
else()
    message(STATUS "  Optimization: Debug Mode")
endif()
message(STATUS "=================================")
message(STATUS "")

project(gemm LANGUAGES CXX CUDA)

# Keep windows.h from defining min/max macros (was a no-op #define in core.hpp, placed
# after the headers it was meant to affect)
if(WIN32)
    add_compile_definitions(NOMINMAX)
endif()

include(CheckLanguage)
check_language(CUDA)
check_language(CXX)

# Same stream semantics in every build type (was Release only, so Debug used the legacy default
# stream and could hide or show synchronisation bugs that Release did not). Affects .cu files;
# host .cpp files compiled by MSVC/GCC still use the legacy default stream for stream 0.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream=per-thread")
# Relocatable device code comes from CMAKE_CUDA_SEPARABLE_COMPILATION (per target), on every
# platform. It was also a global -rdc=true in CMAKE_CUDA_FLAGS on Windows; that compiled
# src/wgmma_gemm.cu with -rdc too, although its library turns separable
# compilation off, so both gemm.lib's device link and the executable's registered their device code
# (LNK2005 __cudaRegisterLinkedBinary_..._wgmma_gemm_cu already defined in cmake_device_link.obj).
# Optimized CUDA flags for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
        -O3 \
        --ftz=true \
        -Xptxas=-O3 \
        -Xptxas=-v \
        -Xptxas=-warn-lmem-usage \
        -Xptxas=-warn-spills")
    # no --maxrregcount: 255 registers per thread is the hardware limit and the default anyway;
    # the kernels that need a budget set it with __launch_bounds__ (and setmaxnreg in wgmma)
    
    # Platform-specific host compiler optimization
    if(WIN32)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/O2")
    else()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-O3")
    endif()
    
    # Add lineinfo unless building for production
    if(NOT PRODUCTION_CODE)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
    endif()
    
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
        -G \
        -g \
        -lineinfo \
        -Xptxas=-v")
    
    # Platform-specific host compiler debug flags
    if(WIN32)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Od")
    else()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-g -Xcompiler=-O0")
    endif()
endif()

# MKL (CPU reference GEMM).
# Linux: use the oneAPI install in /opt/intel/oneapi (2025.2 preferred, else the newest one), unless
# MKL_DIR was given on the command line / in the cache or MKLROOT is set (oneAPI setvars.sh).
# Windows: MKL is found through MKL_DIR / MKLROOT from the kit or environment, as before.
if(UNIX AND NOT APPLE AND NOT MKL_DIR AND NOT DEFINED ENV{MKLROOT})
    set(GEMM_MKL_PREFERRED_DIR /opt/intel/oneapi/mkl/2025.2/lib/cmake/mkl)
    if(EXISTS "${GEMM_MKL_PREFERRED_DIR}/MKLConfig.cmake")
        set(MKL_DIR "${GEMM_MKL_PREFERRED_DIR}" CACHE PATH "Folder containing MKLConfig.cmake")
    else()
        file(GLOB GEMM_MKL_CANDIDATES /opt/intel/oneapi/mkl/*/lib/cmake/mkl/MKLConfig.cmake)
        if(GEMM_MKL_CANDIDATES)
            list(SORT GEMM_MKL_CANDIDATES COMPARE NATURAL ORDER DESCENDING)
            list(GET GEMM_MKL_CANDIDATES 0 GEMM_MKL_CONFIG)
            get_filename_component(GEMM_MKL_FOUND_DIR "${GEMM_MKL_CONFIG}" DIRECTORY)
            message(WARNING "MKL 2025.2 not found in /opt/intel/oneapi/mkl, using ${GEMM_MKL_FOUND_DIR}")
            set(MKL_DIR "${GEMM_MKL_FOUND_DIR}" CACHE PATH "Folder containing MKLConfig.cmake")
        endif()
    endif()
endif()

set(MKL_HINTS)
if(DEFINED ENV{MKLROOT})
    file(TO_CMAKE_PATH "$ENV{MKLROOT}" MKL_ROOT_PATH)
    list(APPEND MKL_HINTS "${MKL_ROOT_PATH}/lib/cmake/mkl")
endif()
if(DEFINED ENV{ONEAPI_ROOT})
    file(TO_CMAKE_PATH "$ENV{ONEAPI_ROOT}" ONEAPI_ROOT_PATH)
    list(APPEND MKL_HINTS "${ONEAPI_ROOT_PATH}/mkl/latest/lib/cmake/mkl")
endif()
find_package(MKL CONFIG REQUIRED HINTS ${MKL_HINTS})
message(STATUS "MKL: ${MKL_DIR}")

message(STATUS "MKL: ${MKL_DIR}")
find_package(Threads)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_AUTOMOC ON)
# Qt: Core (QCoreApplication, QCommandLineParser, QDebug), Test (QtTest in tst/unit_tests.cpp,
# tst/reference.cpp). No Widgets / Gui: everything runs headless (compute nodes without a display)
set(QT_COMPONENTS
        Core
        Test)
find_package(Qt5 COMPONENTS ${QT_COMPONENTS} REQUIRED)
set(OpenCV_CUDA ON)
set(OpenCV_STATIC ON)

# only the modules the code includes: OpenCV_LIBS then leaves out the others, e.g. opencv_cvv,
# whose link interface needs Qt5::Gui
find_package(OpenCV REQUIRED COMPONENTS core cudaarithm cudev)
# OpenCV's static CUDA modules carry "-LIBPATH:<CUDA lib dir>" as a raw link item; with spaces in the
# path nvcc -dlink splits it. Replace it by an INTERFACE link directory, which CMake quotes/shortens.
# Only the modules used (OpenCV_LIBS): every imported OpenCV target would include opencv_cvv, and
# setting a property on it fails without Qt5::Gui ("link interface ... contains Qt5::Gui").
foreach(imported IN LISTS OpenCV_LIBS)
    if(NOT TARGET ${imported})
        continue()
    endif()
    get_target_property(imported_libs ${imported} INTERFACE_LINK_LIBRARIES)
    if(imported_libs)
        set(libpaths ${imported_libs})
        list(FILTER libpaths INCLUDE REGEX "-LIBPATH:")
        if(libpaths)
            list(FILTER imported_libs EXCLUDE REGEX "-LIBPATH:")
            set_target_properties(${imported} PROPERTIES INTERFACE_LINK_LIBRARIES "${imported_libs}")
            foreach(item IN LISTS libpaths)
                string(REGEX REPLACE "^.*-LIBPATH:([^>]*)>?$" "\\1" dir "${item}")
                set_property(TARGET ${imported} APPEND PROPERTY INTERFACE_LINK_DIRECTORIES "${dir}")
            endforeach()
        endif()
    endif()
endforeach()

include_directories(${Qt5Core_INCLUDE_DIRS})

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
# CUDA 13 moved Thrust / CUB / libcu++ into include/cccl: nvcc adds it by itself, the host compiler
# (the .cpp files that include thrust headers) does not. Only present from CUDA 13 on. Passed as a
# C++ compile option, not an include directory: Qt's moc (AUTOMOC) uses the include directories and
# fails on CCCL 3's namespace macros ("Parse error at THRUST_300102"), the compile options it ignores.
foreach(cuda_include_dir ${CUDA_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    if(EXISTS "${cuda_include_dir}/cccl")
        if(MSVC)
            add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/I${cuda_include_dir}/cccl>")
        else()
            add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-isystem${cuda_include_dir}/cccl>")
        endif()
    endif()
endforeach()
include_directories(${MKL_ROOT}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/gen)  # tuning_table.hpp (include/tuning.hpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

message(STATUS "Using OCV libs " ${OpenCV_LIBS})

set(TARGET_LIBRARIES
	${OpenCV_LIBS}
	cublas
	cublasLt # gemm_benchmark "CUBLAS best": the fastest candidate of cuBLASLt's heuristic
	cudart_static
)
# NVML (power / energy readings): libnvidia-ml on Linux, nvml.lib on Windows
if(UNIX)
	list(APPEND TARGET_LIBRARIES nvidia-ml)
else()
	list(APPEND TARGET_LIBRARIES nvml)
endif()

set(SRC_INCLUDE_DIR src)
file(GLOB CPP_SOURCES CONFIGURE_DEPENDS # new files in src/ re-run CMake
       "${SRC_INCLUDE_DIR}/*.cpp"
       "${SRC_INCLUDE_DIR}/*.cu"
)
file(GLOB CPP_HEADERS
       "${SRC_INCLUDE_DIR}/*.h"
       "${SRC_INCLUDE_DIR}/*.ini"
       "${SRC_INCLUDE_DIR}/*.hpp"
       "${SRC_INCLUDE_DIR}/*.cuh"
       "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
       "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
       "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cuh"
)

# gen/tuning_table.hpp (written by test_diagnostics tune): part of the project, and every
# translation unit that expands it is rebuilt when it changes (OBJECT_DEPENDS in addition to
# the compiler's header dependencies, which nvcc does not always report)
set(GEMM_TUNING_TABLE "${CMAKE_CURRENT_SOURCE_DIR}/gen/tuning_table.hpp")
list(APPEND CPP_HEADERS "${GEMM_TUNING_TABLE}")
set_source_files_properties("${SRC_INCLUDE_DIR}/bfs_caps.cu" "${SRC_INCLUDE_DIR}/tuning.cpp"
                            PROPERTIES OBJECT_DEPENDS "${GEMM_TUNING_TABLE}")

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Settings shared by the library and every test/tool executable (see tst/CMakeLists.txt).
# Configuration macros (L2_CACHE_POLICY, ...) and MKL flags are PUBLIC on gemm, so linking gemm
# (directly or through test_utils) is enough to get them.
function(gemm_apply_common_settings target)
    target_compile_options(${target} PRIVATE
            #$<$<COMPILE_LANGUAGE:CUDA>:-G -g>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo -Xptxas -v>      # Debug flags for CUDA
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v>   # Release flags for CUDA
            # -rdc=true is already set globally for Windows
    )
    set_property(TARGET ${target} PROPERTY
      MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,Release>:ProgramDatabase>")
endfunction()

set(TARGET_NAME gemm)
# src/wgmma_gemm.cu is built separately, without relocatable device code (gemm_wgmma below)
list(REMOVE_ITEM CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_INCLUDE_DIR}/wgmma_gemm.cu")
add_library(${TARGET_NAME} STATIC ${CPP_SOURCES} ${CPP_HEADERS})
gemm_apply_common_settings(${TARGET_NAME})

target_link_libraries(${TARGET_NAME} ${TARGET_LIBRARIES})

# Add compile definitions for configuration options
# Convert CMake boolean options to C++ boolean values (1/0)
# PUBLIC: every target that links gemm (test_utils, all tests) sees the same values as the
# library itself. With PRIVATE the tests silently fell back to the defaults in core.hpp.
target_compile_definitions(${TARGET_NAME} PUBLIC
    $<$<BOOL:${LOOPS_COUNT_AS_TEMPLATE_ARGS}>:LOOPS_COUNT_AS_TEMPLATE_ARGS=1>
    $<$<NOT:$<BOOL:${LOOPS_COUNT_AS_TEMPLATE_ARGS}>>:LOOPS_COUNT_AS_TEMPLATE_ARGS=0>
    $<$<BOOL:${REPORT_STATISTICS}>:REPORT_STATISTICS=1>
    $<$<NOT:$<BOOL:${REPORT_STATISTICS}>>:REPORT_STATISTICS=0>
    $<$<BOOL:${L2_CACHE_POLICY}>:L2_CACHE_POLICY=1>
    $<$<NOT:$<BOOL:${L2_CACHE_POLICY}>>:L2_CACHE_POLICY=0>
    $<$<BOOL:${USE_NVTX}>:USE_NVTX=1>
    $<$<NOT:$<BOOL:${USE_NVTX}>>:USE_NVTX=0>
    $<$<BOOL:${FULL_UNROLL_ROW_LOOPS}>:FULL_UNROLL_ROW_LOOPS=1>
    $<$<NOT:$<BOOL:${FULL_UNROLL_ROW_LOOPS}>>:FULL_UNROLL_ROW_LOOPS=0>
    $<$<BOOL:${LAST_USE_HINTS}>:LAST_USE_HINTS=1>
    $<$<NOT:$<BOOL:${LAST_USE_HINTS}>>:LAST_USE_HINTS=0>
    BFS_WARPS_Y=${BFS_WARPS_Y}
    $<$<STREQUAL:${LEAF_GEMM},wgmma>:LEAF_GEMM_WGMMA=1>
    $<$<NOT:$<STREQUAL:${LEAF_GEMM},wgmma>>:LEAF_GEMM_WGMMA=0>
    $<$<BOOL:${PRODUCTION_CODE}>:PRODUCTION_CODE=1>
    $<$<NOT:$<BOOL:${PRODUCTION_CODE}>>:PRODUCTION_CODE=0>
    # print_build_config() (include/common.hpp): stringified there, so no quotes here
    GEMM_BUILD_CONFIG=$<CONFIG>
    GEMM_CUDA_ARCH=${GEMM_CUDA_ARCH_TEXT}
)

# Our Hopper GEMM (src/wgmma_gemm.cu) as its own library WITHOUT relocatable device code: with
# -rdc=true (CMAKE_CUDA_SEPARABLE_COMPILATION, needed by the other .cu files) ptxas ignores
# setmaxnreg ("C7504 Potential Performance Loss: 'setmaxnreg' ignored to maintain compatibility
# across compilation units"), so the producer warpgroup kept its registers and the consumers were
# capped at 168 (65536 / 384 threads) instead of 232. The file calls no device code of other
# files, so it needs no device linking. Same configuration macros as gemm.
add_library(gemm_wgmma STATIC ${SRC_INCLUDE_DIR}/wgmma_gemm.cu)
set_target_properties(gemm_wgmma PROPERTIES CUDA_SEPARABLE_COMPILATION OFF CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
# OFF: the kernel without its register handover (setmaxnreg), for tests (src/wgmma_gemm.cu)
option(WGMMA_SETMAXNREG "wgmma kernel: producer / consumer register handover (setmaxnreg)" ON)
target_compile_definitions(gemm_wgmma PRIVATE $<IF:$<BOOL:${WGMMA_SETMAXNREG}>,WGMMA_SETMAXNREG=1,WGMMA_SETMAXNREG=0>)
# OFF: the kernel without prefetch.tensormap, for tests (src/wgmma_gemm.cu)
option(WGMMA_PREFETCH_TENSORMAP "wgmma kernel: prefetch the tensor maps (prefetch.tensormap)" ON)
target_compile_definitions(gemm_wgmma PRIVATE
    $<IF:$<BOOL:${WGMMA_PREFETCH_TENSORMAP}>,WGMMA_PREFETCH_TENSORMAP=1,WGMMA_PREFETCH_TENSORMAP=0>)
gemm_apply_common_settings(gemm_wgmma)
target_compile_definitions(gemm_wgmma PRIVATE $<TARGET_PROPERTY:${TARGET_NAME},INTERFACE_COMPILE_DEFINITIONS>)
target_link_libraries(gemm_wgmma ${TARGET_LIBRARIES})
target_link_libraries(${TARGET_NAME} gemm_wgmma)

target_compile_options(${TARGET_NAME} PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(${TARGET_NAME} PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)

# Documentation generation
find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/docs)
    
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_IN}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
    
    # Add documentation to all target
    add_custom_target(docs-all
        DEPENDS docs
        COMMENT "All documentation generated"
    )
    
    message(STATUS "Doxygen found: ${DOXYGEN_EXECUTABLE}")
    message(STATUS "Documentation target 'docs' available")
else()
    message(STATUS "Doxygen not found - documentation target not available")
    message(STATUS "Install Doxygen to generate API documentation")
endif()

message(LOG "${TARGET_LIBRARIES}")
list(APPEND TARGET_LIBRARIES
    gemm
    MKL::MKL
    Qt5::Core
)

enable_testing()
add_subdirectory(tst)
