cmake_minimum_required(VERSION 3.8)
project(racing_track_detection)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ai_msgs REQUIRED)
find_package(dnn_node REQUIRED)
find_package(foxglove_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(hbm_img_msgs REQUIRED)
find_package(hobot_cv REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

link_directories(
  ${CMAKE_SYSROOT}/usr/lib/hbbpu/
  /usr/hobot/lib
)

add_executable(${PROJECT_NAME} src/racing_track_detection.cpp)
ament_target_dependencies(
  ${PROJECT_NAME}
  ai_msgs
  dnn_node
  foxglove_msgs
  geometry_msgs
  hbm_img_msgs
  hobot_cv
  rclcpp
  std_msgs
)
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(
  ${PROJECT_NAME}
  ${OpenCV_LIBS}
  hbmem
)

install(TARGETS ${PROJECT_NAME} DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
