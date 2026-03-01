cmake_minimum_required(VERSION 3.18)
project(OvaSearch CXX)

set(CMAKE_CXX_STANDARD 17)

# OpenVINO paths
set(OpenVINO_DIR "${CMAKE_SOURCE_DIR}/env/lib/python3.14/site-packages/openvino/cmake")
find_package(OpenVINO REQUIRED)

# Removed openvino_genai component search due to ABI compilation issues
# Direct linking to the PIP pre-compiled macOS arm64 extensions instead

include(FetchContent)
FetchContent_Declare(
    usearch
    GIT_REPOSITORY https://github.com/unum-cloud/usearch.git
    GIT_TAG        v2.13.0
)
FetchContent_MakeAvailable(usearch)

add_executable(ovasearch main.cpp)
target_include_directories(ovasearch PRIVATE 
    ${usearch_SOURCE_DIR}/include 
    "${CMAKE_SOURCE_DIR}/openvino.genai/src/cpp/include"
    "${CMAKE_SOURCE_DIR}/include"
)

# Link against OpenVINO runtime from the user's master build,
# and link against the pre-compiled OpenVINO GenAI from pip
target_link_libraries(ovasearch PRIVATE 
    openvino::runtime 
    "${CMAKE_SOURCE_DIR}/env/lib/python3.14/site-packages/openvino_genai/libopenvino_genai.2600.dylib"
)

set_target_properties(ovasearch PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${CMAKE_SOURCE_DIR}/env/lib/python3.14/site-packages/openvino_genai;${CMAKE_SOURCE_DIR}/env/lib/python3.14/site-packages/openvino/libs;${CMAKE_SOURCE_DIR}/env/lib/python3.14/site-packages/openvino_tokenizers"
)
