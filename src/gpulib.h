#ifndef GPULIB_H
#define GPULIB_H

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

cl_int read_cl_source(const char *source_path, char **kernel_source, size_t *source_size);
cl_int build_cl_program(const cl_program program, const cl_device_id device);
cl_int setup_cl(cl_device_id *device, cl_context *context, cl_command_queue *command_queue);

#endif
