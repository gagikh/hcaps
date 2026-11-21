# Test Framework CMake Configuration
# ==================================
# Builds the test utilities library, one executable per *.cpp / *.cu in this folder and the
# CPU-only reference tests.
#
# Everything shared with the library comes from the root CMakeLists.txt:
#   - configuration macros (L2_CACHE_POLICY, ...) and MKL flags: PUBLIC on `gemm`
#   - CUDA/MSVC compile settings:                gemm_apply_common_settings(<target>)
#   - link libraries:                            ${TARGET_LIBRARIES} (includes gemm, MKL, Qt)
#   - CUDA architectures:                        global CMAKE_CUDA_ARCHITECTURES / CUDA flags
#
# CTest labels:  ctest -L cpu   (no GPU needed)      ctest -L gpu
#
# Author: GEMM Project
# Date: 2025

# ---------------------------------------------------------------------------------------------
# Test utilities library
# ---------------------------------------------------------------------------------------------
set(TEST_UTILS_SOURCES
    test_utils.cpp
    results.cpp
    performance_tests.cpp
)

add_library(test_utils STATIC ${TEST_UTILS_SOURCES})
gemm_apply_common_settings(test_utils)
target_link_libraries(test_utils PUBLIC ${TARGET_LIBRARIES} Qt5::Test)
target_include_directories(test_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# ---------------------------------------------------------------------------------------------
# One executable per source file
# ---------------------------------------------------------------------------------------------
file(GLOB ALL_SOURCES "*.cpp" "*.cu")
list(REMOVE_ITEM ALL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/test_utils.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/results.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/performance_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/reference.cpp"         # built separately below
)
message(STATUS "Test sources: ${ALL_SOURCES}")

# executables that are tools, not tests (built, but not registered with CTest)
set(NOT_A_TEST diagnostics)

foreach (FILE_NAME ${ALL_SOURCES})
    get_filename_component(TEST_NAME "${FILE_NAME}" NAME_WE)
    set(TARGET_NAME test_${TEST_NAME})

    add_executable(${TARGET_NAME} ${FILE_NAME})
    set_target_properties(${TARGET_NAME} PROPERTIES
        AUTOGEN_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/autogen/${TARGET_NAME}_autogen
    )
    gemm_apply_common_settings(${TARGET_NAME})
    target_link_libraries(${TARGET_NAME} test_utils)
    if (TEST_NAME STREQUAL "diagnostics")
        # tune writes <source>/gen/tuning_table.hpp wherever it is started from
        get_filename_component(GEMM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
        target_compile_definitions(${TARGET_NAME} PRIVATE GEMM_SOURCE_DIR="${GEMM_SOURCE_DIR}")
    endif()

    if (NOT TEST_NAME IN_LIST NOT_A_TEST)
        add_test(
            NAME ${TARGET_NAME}
            COMMAND ${TARGET_NAME}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
        set_tests_properties(${TARGET_NAME} PROPERTIES LABELS "gpu")
    endif()
endforeach()

# ---------------------------------------------------------------------------------------------
# CPU-only reference tests: OpenCV core + QtTest, no CUDA/MKL/GPU needed at runtime.
#   ctest -L cpu      or      ./bin/test_reference -v2
# ---------------------------------------------------------------------------------------------
add_executable(test_reference reference.cpp)
set_target_properties(test_reference PROPERTIES
    AUTOGEN_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/autogen/test_reference_autogen
)
target_include_directories(test_reference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(test_reference opencv_core Qt5::Core Qt5::Test)
add_test(NAME test_reference COMMAND test_reference)
set_tests_properties(test_reference PROPERTIES LABELS "cpu")
