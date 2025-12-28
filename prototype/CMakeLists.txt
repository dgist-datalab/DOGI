cmake_minimum_required(VERSION 3.15)
project(dogi)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_definitions(
    OS_LINUX
    ROCKSDB_PLATFORM_POSIX
    ROCKSDB_LIB_IO_POSIX
)

add_subdirectory(src)
add_subdirectory(app)

