enable_testing()

add_executable(firefly)
target_sources(firefly
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/main.c
        ${CMAKE_CURRENT_SOURCE_DIR}/geometry.c
        ${CMAKE_CURRENT_SOURCE_DIR}/geometry.h
        ${CMAKE_CURRENT_SOURCE_DIR}/gpulib.c
        ${CMAKE_CURRENT_SOURCE_DIR}/gpulib.h
    )

configure_file(kernels/path-trace.cl kernels/path-trace.cl COPYONLY)
configure_file(kernels/camera-directions.cl kernels/camera-directions.cl COPYONLY)

find_package(OpenCL REQUIRED)
target_link_libraries(firefly OpenCL::OpenCL m)