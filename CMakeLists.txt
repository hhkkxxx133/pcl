cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(pppcl)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
set( CMAKE_EXE_LINKER_FLAGS "-ljsoncpp " )

add_executable (pppcl test.cpp)
target_link_libraries (pppcl ${PCL_LIBRARIES})
