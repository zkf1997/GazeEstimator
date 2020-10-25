cmake_minimum_required(VERSION 3.9.0)
project(gaze_estimator LANGUAGES C CXX
    VERSION 0.1)

#####################################
# please modify the paths accordingly
add_subdirectory(D:/coding/libs/dlib-19.21/dlib dlib_build)# add dlib
set(OpenCV_DIR D:/coding/libs/opencv/build)#set opencv directory
set(CMAKE_PREFIX_PATH D:/coding/libs/libtorch)#set libtorch directory
#####################################

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(OpenCV QUIET)
if (OpenCV_FOUND)
   include_directories(${OpenCV_INCLUDE_DIRS})

   add_executable(gaze_estimator gaze_estimator.cpp normalizer.cpp)
   include_directories(gaze_estimator ./include)
   target_link_libraries(gaze_estimator dlib::dlib ${OpenCV_LIBS} "${TORCH_LIBRARIES}")
   set_property(TARGET gaze_estimator PROPERTY CXX_STANDARD 17)
else()
   message("OpenCV not found, so we won't build.")
endif()

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET gaze_estimator
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:gaze_estimator>)
endif (MSVC)
