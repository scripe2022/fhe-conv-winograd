cmake_minimum_required(VERSION 3.20)
project(openfhe_conv LANGUAGES CXX)
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenFHE CONFIG REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)

add_executable(openfhe_conv
    src/main.cpp
    src/utils.cpp
    src/orion.cpp
    src/winograd.cpp
    /home/jyh/.local/include/cpglib/print.o
)

target_link_libraries(openfhe_conv
    PRIVATE
        OPENFHEpke
        ${HDF5_CXX_LIBRARIES}
        ${HDF5_C_LIBRARIES}
)

target_include_directories(openfhe_conv
    PRIVATE
        ${OpenFHE_INCLUDE}
        ${OpenFHE_INCLUDE}/pke
        ${OpenFHE_INCLUDE}/core
        ${OpenFHE_INCLUDE}/binfhe
        ${HDF5_INCLUDE_DIRS}
)

