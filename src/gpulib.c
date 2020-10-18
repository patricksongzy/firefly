#include <stdio.h>
#include <errno.h>

#include "gpulib.h"

/**
 * @brief Reads an integer value from user input.
 * 
 * @param label the label for the value being read.
 * @param from the minimum value.
 * @param to the maximum value.
 * @return const cl_int the return code.
 */
const cl_int read_value(const char *label, const cl_int from, const cl_int to)
{
    char *line = NULL;
    size_t size;
    size_t success = 0;

    cl_int result;

    do
    {
        printf("Please enter a %s in [%d, %d]: ", label, from, to);
        if (getline(&line, &size, stdin) == -1)
            continue;

        errno = 0;

        char *endptr;
        result = strtol(line, &endptr, 10);
        if (endptr == line)
            printf("No input read.\n");
        else if (errno == ERANGE)
            printf("Input is out of range.\n");
        else if (result < from || result > to)
            printf("Input must be in range [%d, %d].\n", from, to);
        else
            success = 1;
    } while (!success);

    return result;
}

/**
 * @brief Gets the device name of a given device.
 * 
 * @param device the device.
 * @param device_name a pointer to the device name.
 * @return const cl_int the return code.
 */
const cl_int get_device_name(const cl_device_id device, char **device_name)
{
    cl_int ret;

    size_t parameter_size;

    ret = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &parameter_size);
    if (ret != CL_SUCCESS)
        return ret;

    *device_name = malloc(parameter_size * sizeof(char));

    ret = clGetDeviceInfo(device, CL_DEVICE_NAME, parameter_size, *device_name, NULL);
    return ret;
}

/**
 * @brief Gets the platform name of a given platform.
 * 
 * @param platform the platform.
 * @param platform_name a pointer to the platform name.
 * @return const cl_int the return code.
 */
const cl_int get_platform_name(const cl_platform_id platform, char **platform_name)
{
    cl_int ret;

    size_t parameter_size;

    ret = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &parameter_size);
    if (ret != CL_SUCCESS)
        return ret;

    *platform_name = malloc(parameter_size * sizeof(char));

    ret = clGetPlatformInfo(platform, CL_PLATFORM_NAME, parameter_size, *platform_name, NULL);
    return ret;
}

cl_int read_cl_source(const char *source_path, char **kernel_source, size_t *source_size)
{
    cl_int ret;

    FILE *fp;

    fp = fopen(source_path, "r");
    if (ret = (fp == NULL))
        return ret;

    // find the source size
    fseek(fp, 0, SEEK_END);
    *source_size = ftell(fp);
    rewind(fp);
    // read the source
    *kernel_source = (char *)malloc((*source_size + 1) * sizeof(char));
    *(*kernel_source + *source_size) = '\0';
    fread(*kernel_source, sizeof(char), *source_size, fp);
    fclose(fp);

    return CL_SUCCESS;
}

cl_int build_cl_program(const cl_program program, const cl_device_id device)
{
    cl_int ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (ret == CL_BUILD_PROGRAM_FAILURE)
    {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        char *log = malloc(log_size * sizeof(char));

        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        fprintf(stderr, "%s\n", log);
        free(log);
    }

    return ret;
}

cl_int setup_cl(cl_device_id *device, cl_context *context, cl_command_queue *command_queue)
{
    cl_int ret;

    cl_uint num_platforms;
    cl_uint num_devices;

    cl_uint platform_index;
    cl_uint device_index;

    cl_platform_id *platforms;
    cl_device_id *devices;

    // query the number of available platforms
    ret = clGetPlatformIDs(0, NULL, &num_platforms);
    if (ret != CL_SUCCESS)
        goto out;

    platforms = malloc(num_platforms * sizeof(cl_platform_id));
    if (ret = (platforms == NULL))
        goto out;

    // query the platform_ids
    ret = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (ret != CL_SUCCESS)
        goto cleanup_platforms;

    printf("num_platforms: %d\n", num_platforms);
    printf("Platforms:\n");

    for (cl_uint i = 0; i < num_platforms; i++)
    {
        char *platform_name;
        ret = get_platform_name(platforms[i], &platform_name);
        if (ret = (platform_name == NULL))
            goto cleanup_platforms;

        printf("%d: %s\n", i, platform_name);

        free(platform_name);

        if (ret != CL_SUCCESS)
            goto cleanup_platforms;
    }

    platform_index = num_platforms == 1 ? 0 : read_value("platform index", 0, num_platforms - 1);

    ret = clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    if (ret != CL_SUCCESS)
        goto cleanup_platforms;

    devices = malloc(num_devices * sizeof(cl_device_id));
    if (ret = (devices == NULL))
        goto cleanup_platforms;

    ret = clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    if (ret != CL_SUCCESS)
        goto cleanup_devices;

    printf("num_devices: %d\n", num_devices);
    printf("Devices:\n");

    for (cl_uint i = 0; i < num_devices; i++)
    {
        char *device_name;
        ret = get_device_name(devices[i], &device_name);
        if (ret = (device_name == NULL))
            goto cleanup_devices;

        printf("%d: %s\n", i, device_name);

        free(device_name);

        if (ret != CL_SUCCESS)
            goto cleanup_devices;
    }

    device_index = num_devices == 1 ? 0 : read_value("device index", 0, num_devices - 1);
    *device = devices[device_index];

    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[platform_index], 0};

    *context = clCreateContext(properties, 1, device, NULL, NULL, &ret);
    if (ret != CL_SUCCESS)
        goto cleanup_devices;

    *command_queue = clCreateCommandQueueWithProperties(*context, *device, NULL, &ret);
    if (ret != CL_SUCCESS)
        clReleaseContext(*context);
    
cleanup_devices:
    free(devices);
cleanup_platforms:
    free(platforms);
out:
    return ret;
}