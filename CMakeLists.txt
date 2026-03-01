cmake_minimum_required(VERSION 3.18)
project(OvaSearch CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# ── Auto-detect Python virtual-env site-packages ──────────────────────
# Works with any Python 3.x version (3.8 … 3.14+)
# Prefer project-local venv if it exists
if(EXISTS "${CMAKE_SOURCE_DIR}/env/bin/python3")
    set(Python3_EXECUTABLE "${CMAKE_SOURCE_DIR}/env/bin/python3")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/env/Scripts/python.exe")
    set(Python3_EXECUTABLE "${CMAKE_SOURCE_DIR}/env/Scripts/python.exe")
endif()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
            "import sysconfig; print(sysconfig.get_path('purelib'))"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Python site-packages: ${PYTHON_SITE_PACKAGES}")

# ── OpenVINO (installed via pip) ──────────────────────────────────────
set(OpenVINO_DIR "${PYTHON_SITE_PACKAGES}/openvino/cmake")
find_package(OpenVINO REQUIRED)

# ── Fetch dependencies ────────────────────────────────────────────────
include(FetchContent)

# USearch – high-performance vector index
FetchContent_Declare(
    usearch
    GIT_REPOSITORY https://github.com/unum-cloud/usearch.git
    GIT_TAG        v2.13.0
)
FetchContent_MakeAvailable(usearch)

# OpenVINO GenAI C++ headers (only the include dir is needed)
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/openvino.genai/src/cpp/include")
    message(STATUS "OpenVINO GenAI headers not found locally, cloning...")
    execute_process(
        COMMAND git clone --depth 1
                https://github.com/openvinotoolkit/openvino.genai.git
                "${CMAKE_SOURCE_DIR}/openvino.genai"
        RESULT_VARIABLE _genai_clone_result
    )
    if(NOT _genai_clone_result EQUAL 0)
        message(FATAL_ERROR
            "Failed to clone OpenVINO GenAI headers. Clone manually:\n"
            "  git clone --depth 1 https://github.com/openvinotoolkit/openvino.genai.git")
    endif()
endif()
set(GENAI_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/openvino.genai/src/cpp/include")

# ── Locate the pre-compiled OpenVINO GenAI shared library from pip ────
# This avoids ABI issues from building GenAI from source; we link
# directly to the pip-installed binary instead.
file(GLOB GENAI_LIBS
    "${PYTHON_SITE_PACKAGES}/openvino_genai/libopenvino_genai*.dylib"
    "${PYTHON_SITE_PACKAGES}/openvino_genai/libopenvino_genai*.so"
)
if(NOT GENAI_LIBS)
    message(FATAL_ERROR
        "OpenVINO GenAI library not found in ${PYTHON_SITE_PACKAGES}/openvino_genai/\n"
        "Install it with:  pip install openvino-genai")
endif()
list(GET GENAI_LIBS 0 GENAI_LIB)
message(STATUS "OpenVINO GenAI library: ${GENAI_LIB}")

# ── Build ─────────────────────────────────────────────────────────────
add_executable(ovasearch main.cpp)
target_include_directories(ovasearch PRIVATE
    ${usearch_SOURCE_DIR}/include
    "${GENAI_INCLUDE_DIR}"
    "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(ovasearch PRIVATE
    openvino::runtime
    "${GENAI_LIB}"
)

set_target_properties(ovasearch PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${PYTHON_SITE_PACKAGES}/openvino_genai;${PYTHON_SITE_PACKAGES}/openvino/libs;${PYTHON_SITE_PACKAGES}/openvino_tokenizers"
)
