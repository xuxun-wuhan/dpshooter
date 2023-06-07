cmake_minimum_required(VERSION 3.4.1)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../../sdk/native/jni/include)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../../sdk/native/libs/${ANDROID_ABI})
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../../sdk/native/staticlibs/${ANDROID_ABI})
# Set your source files here.
set(SOURCE_FILES native-lib.cpp)

add_library( # Specifies the name of the library.
        native-lib

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        ${SOURCE_FILES})

# Link the target library with the log library
# included in the NDK and with OpenCV.
target_link_libraries( # Specifies the target library.
        native-lib

        # Links the target library to the log library
        # included in the NDK.
        ${log-lib} m z android log
        opencv_java4 opencv_optflow)