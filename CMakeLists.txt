cmake_minimum_required(VERSION 3.20)
# Derive the project version from include/vccl.h (single source of truth).
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/vccl.h" _vccl_hdr)
string(REGEX MATCH "#define VCCL_MAJOR ([0-9]+)" _m "${_vccl_hdr}")
set(_vccl_major "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define VCCL_MINOR ([0-9]+)" _m "${_vccl_hdr}")
set(_vccl_minor "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define VCCL_PATCH ([0-9]+)" _m "${_vccl_hdr}")
set(_vccl_patch "${CMAKE_MATCH_1}")
project(vccl VERSION ${_vccl_major}.${_vccl_minor}.${_vccl_patch} LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

option(VCCL_ENABLE_VERBS "Build the ibverbs (RoCE/RDMA) transport" ON)
option(VCCL_BUILD_TESTS "Build vccl tests" ON)
option(VCCL_BUILD_BENCH "Build vccl-bench" ON)

add_compile_options(-Wall -Wextra)

set(VCCL_SOURCES
  src/core/types.cc
  src/core/reduce.cc
  src/core/bootstrap.cc
  src/core/comm.cc
  src/core/collectives.cc
  src/core/group.cc
  src/transport/tcp.cc
  src/util/socket.cc
  src/util/log.cc
)

set(VCCL_HAVE_VERBS OFF)
if(VCCL_ENABLE_VERBS)
  include(CheckIncludeFile)
  check_include_file("infiniband/verbs.h" HAVE_VERBS_H)
  find_library(IBVERBS_LIBRARY ibverbs)
  if(HAVE_VERBS_H AND IBVERBS_LIBRARY)
    set(VCCL_HAVE_VERBS ON)
    list(APPEND VCCL_SOURCES src/transport/verbs.cc)
    message(STATUS "ibverbs transport: enabled (${IBVERBS_LIBRARY})")
  else()
    message(STATUS "ibverbs transport: disabled (libibverbs not found)")
  endif()
endif()

add_library(vccl SHARED ${VCCL_SOURCES})
add_library(vccl_static STATIC ${VCCL_SOURCES})

foreach(tgt vccl vccl_static)
  target_include_directories(${tgt}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  if(VCCL_HAVE_VERBS)
    target_compile_definitions(${tgt} PRIVATE VCCL_HAVE_VERBS=1)
    target_link_libraries(${tgt} PRIVATE ${IBVERBS_LIBRARY})
  endif()
  find_package(Threads REQUIRED)
  target_link_libraries(${tgt} PRIVATE Threads::Threads)
endforeach()

set_target_properties(vccl PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR})

install(TARGETS vccl vccl_static)
install(FILES include/vccl.h DESTINATION include)

if(VCCL_BUILD_TESTS)
  enable_testing()
  add_executable(test_collectives tests/test_collectives.cc)
  target_link_libraries(test_collectives PRIVATE vccl_static Threads::Threads)
  target_include_directories(test_collectives PRIVATE include src)
  add_test(NAME collectives COMMAND test_collectives)
  set_tests_properties(collectives PROPERTIES TIMEOUT 300)

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(test_dmabuf tests/test_dmabuf.cc)
    target_link_libraries(test_dmabuf PRIVATE vccl_static Threads::Threads)
    target_include_directories(test_dmabuf PRIVATE include src)
    add_test(NAME dmabuf COMMAND test_dmabuf)
    set_tests_properties(dmabuf PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
  endif()
endif()

if(VCCL_BUILD_BENCH)
  add_executable(vccl-bench bench/vccl_bench.cc)
  target_link_libraries(vccl-bench PRIVATE vccl_static Threads::Threads)
  target_include_directories(vccl-bench PRIVATE include src)

  if(VCCL_HAVE_VERBS)
    add_executable(vccl-info tools/vccl_info.cc)
    target_link_libraries(vccl-info PRIVATE ${IBVERBS_LIBRARY})
  endif()

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
      add_executable(vccl-vk-demo bench/vccl_vk_demo.cc)
      target_link_libraries(vccl-vk-demo PRIVATE vccl_static Vulkan::Vulkan
                                                 Threads::Threads)
      target_include_directories(vccl-vk-demo PRIVATE include src)
      message(STATUS "vccl-vk-demo: enabled")
    else()
      message(STATUS "vccl-vk-demo: disabled (Vulkan not found)")
    endif()
  endif()
endif()
