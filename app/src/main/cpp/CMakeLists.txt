cmake_minimum_required(VERSION 3.4.1)

# Path to OpenCV
set(OpenCV_DIR ../../../../sdk/native/jni)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

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
target_link_libraries(native-lib ${log-lib} ${OpenCV_LIBS})