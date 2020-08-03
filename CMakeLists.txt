# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(mlcpp)




find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
link_directories( ${OpenCV_LIB_DIR} )

find_package(Boost REQUIRED)

#find_package (Python COMPONENTS Interpreter Development)
#include_directories(${Python_INCLUDE_DIRS})

include_directories(${Boost_INCLUDE_DIRS})

add_executable( mlcpp  main.cpp backprop.cpp)
set_property(TARGET mlcpp PROPERTY CXX_STANDARD 17)
set_property(TARGET mlcpp PROPERTY CXX_STANDARD_REQUIRED ON)

target_compile_options(mlcpp PRIVATE /ZI)

target_link_libraries(mlcpp ${OpenCV_LIBS} )
target_link_libraries(mlcpp ${Boost_LIBRARIES})
#target_link_libraries(mlcpp ${Python_LIBRARIES})



add_executable(testcpp testcpp.cpp)
target_link_libraries(testcpp ${OpenCV_LIBS} )
target_link_libraries(testcpp ${Boost_LIBRARIES})