#include <stdio.h>
#include <math.h>

#include "gpulib.h"
#include "geometry.h"

#define WIDTH 1024
#define HEIGHT 1024
#define NUM_SAMPLES 1024

static cl_device_id device;

static cl_context context;
static cl_command_queue command_queue;

// vertical field of view in radians
static cl_float fov = 1.2f;
// world coordinates
static cl_float3 camera_position = {160, 50, 52};
// Euler camera rotation to point forward with origin at the top left
// TODO THE EULER_TO_QUAT ORDER IS REVERSED
static cl_float3 camera_rotation = {CL_M_PI_2, 0, CL_M_PI_2};

struct sphere
{
    cl_float3 position;
    cl_float3 colour;
    cl_float3 emission;
    cl_float radius;
} __attribute__((packed));

const static inline unsigned int convert_pixel(float pixel)
{
    return (unsigned int)(255 * pixel);
}

cl_int generate_directions(cl_mem *directions_buf)
{
    cl_int ret;

    cl_program program;
    cl_kernel kernel;

    size_t local;
    size_t global = HEIGHT * WIDTH;

    cl_uint height = HEIGHT;
    cl_uint width = WIDTH;

    cl_float z_distance = -(height / (2.0f * tan(fov / 2.0f)));
    cl_float4 camera_quat = euler_to_quat(camera_rotation, "xyz");

    size_t source_size;
    char *kernel_source;
    read_cl_source("kernels/camera-directions.cl", &kernel_source, &source_size);

    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, &source_size, &ret);
    free(kernel_source);
    if (ret != CL_SUCCESS)
        goto cleanup_buf;

    ret = build_cl_program(program, device);
    if (ret != CL_SUCCESS)
        goto cleanup_program;

    kernel = clCreateKernel(program, "get_directions", &ret);
    if (ret != CL_SUCCESS)
        goto cleanup_program;

    *directions_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, HEIGHT * WIDTH * sizeof(cl_float3), NULL, &ret);
    if (ret != CL_SUCCESS)
        goto cleanup_kernel;

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), directions_buf);
    ret |= clSetKernelArg(kernel, 1, sizeof(cl_float4), &camera_quat);
    ret |= clSetKernelArg(kernel, 2, sizeof(cl_float), &z_distance);
    ret |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &height);
    ret |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &width);
    if (ret != CL_SUCCESS)
        goto cleanup_buf;

    ret = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &local, NULL);
    if (ret != CL_SUCCESS)
        goto cleanup_buf;

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
        goto cleanup_buf;

    ret = clFinish(command_queue);
    if (ret != CL_SUCCESS)
    cleanup_buf:
        clReleaseMemObject(*directions_buf);

cleanup_kernel:
    clReleaseKernel(kernel);
cleanup_program:
    clReleaseProgram(program);
out:
    return ret;
}

cl_int render(cl_float3 **image, cl_mem *directions_buf)
{
    cl_int ret;

    cl_program program;
    cl_kernel kernel;

    size_t local;
    size_t global = HEIGHT * WIDTH;

    cl_uint height = HEIGHT;
    cl_uint width = WIDTH;
    cl_uint num_samples = NUM_SAMPLES;

    // scene courtesy of smallpt
    struct sphere s1;
    s1.position = (cl_float3){81.6f, 1e4 + 1, 40.8f};
    s1.colour = (cl_float3){0.75, 0.25, 0.25};
    s1.emission = (cl_float3){0, 0, 0};
    s1.radius = 1e4f;

    struct sphere s2;
    s2.position = (cl_float3){81.6f, -1e4 + 99, 40.8f};
    s2.colour = (cl_float3){0.25, 0.25, 0.75};
    s2.emission = (cl_float3){0, 0, 0};
    s2.radius = 1e4f;

    struct sphere s3;
    s3.position = (cl_float3){1e4, 50, 40.8f};
    s3.colour = (cl_float3){0.75, 0.75, 0.75};
    s3.emission = (cl_float3){0, 0, 0};
    s3.radius = 1e4f;

    struct sphere s4;
    s4.position = (cl_float3){-1e4 + 170, 50, 40.8f};
    s4.colour = (cl_float3){0, 0, 0};
    s4.emission = (cl_float3){0, 0, 0};
    s4.radius = 1e4f;

    struct sphere s5;
    s5.position = (cl_float3){81.6f, 50, 1e4};
    s5.colour = (cl_float3){0.75, 0.75, 0.75};
    s5.emission = (cl_float3){0, 0, 0};
    s5.radius = 1e4f;

    struct sphere s6;
    s6.position = (cl_float3){81.6f, 50, -1e4 + 81.6f};
    s6.colour = (cl_float3){0.75, 0.75, 0.75};
    s6.emission = (cl_float3){0, 0, 0};
    s6.radius = 1e4f;

    struct sphere s7;
    s7.position = (cl_float3){47, 27, 16.5f};
    s7.colour = (cl_float3){1, 1, 1};
    s7.emission = (cl_float3){0, 0, 0};
    s7.radius = 16.5f;

    struct sphere s8;
    s8.position = (cl_float3){78, 73, 16.5f};
    s8.colour = (cl_float3){1, 1, 1};
    s8.emission = (cl_float3){0, 0, 0};
    s8.radius = 16.5f;

    struct sphere s9;
    s9.position = (cl_float3){81.6, 50, 681.6f - 0.27f};
    s9.colour = (cl_float3){0, 0, 0};
    s9.emission = (cl_float3){12, 12, 12};
    s9.radius = 600;

    struct sphere scene_spheres[] = {s1, s2, s3, s4, s5, s6, s7, s8, s9};
    size_t num_spheres = sizeof(scene_spheres) / sizeof(struct sphere);

    *image = calloc(HEIGHT * WIDTH, sizeof(cl_float3));

    size_t source_size;
    char *kernel_source;
    read_cl_source("kernels/path-trace.cl", &kernel_source, &source_size);

    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, &source_size, &ret);
    free(kernel_source);
    if (ret != CL_SUCCESS)
        goto cleanup_program;

    ret = build_cl_program(program, device);
    if (ret != CL_SUCCESS)
        goto cleanup_program;

    kernel = clCreateKernel(program, "render", &ret);
    if (ret != CL_SUCCESS)
        goto cleanup_program;

    cl_mem image_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, HEIGHT * WIDTH * sizeof(cl_float3), NULL, &ret);
    if (ret != CL_SUCCESS)
        goto cleanup_kernel;
    
    cl_mem scene_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, num_spheres * sizeof(struct sphere), NULL, &ret);
    if (ret != CL_SUCCESS)
        goto cleanup_image;
    
    // ensure no undefined behaviour by copying the zeroed values
    ret = clEnqueueWriteBuffer(command_queue, image_buf, CL_TRUE, 0, HEIGHT * WIDTH * sizeof(cl_float3), *image, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
        goto cleanup_buf;

    ret = clEnqueueWriteBuffer(command_queue, scene_buf, CL_TRUE, 0, num_spheres * sizeof(struct sphere), scene_spheres, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
        goto cleanup_buf;

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &image_buf);
    ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &scene_buf);
    ret |= clSetKernelArg(kernel, 2, sizeof(cl_uint), &num_spheres);
    ret |= clSetKernelArg(kernel, 3, sizeof(cl_mem), directions_buf);
    ret |= clSetKernelArg(kernel, 4, sizeof(cl_float3), &camera_position);
    ret |= clSetKernelArg(kernel, 5, sizeof(cl_uint), &height);
    ret |= clSetKernelArg(kernel, 6, sizeof(cl_uint), &width);
    ret |= clSetKernelArg(kernel, 7, sizeof(cl_uint), &num_samples);
    if (ret != CL_SUCCESS)
        goto cleanup_buf;

    ret = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &local, NULL);
    if (ret != CL_SUCCESS)
        goto cleanup_buf;

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
        goto cleanup_buf;

    ret = clFinish(command_queue);
    if (ret != CL_SUCCESS)
        goto cleanup_buf;

    ret = clEnqueueReadBuffer(command_queue, image_buf, CL_TRUE, 0, HEIGHT * WIDTH * sizeof(cl_float3), *image, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
        goto cleanup_buf;

cleanup_buf:
    clReleaseMemObject(scene_buf);
cleanup_image:
    clReleaseMemObject(image_buf);
cleanup_kernel:
    clReleaseKernel(kernel);
cleanup_program:
    clReleaseProgram(program);
out:
    return ret;
}

int main(void)
{
    cl_int ret;

    ret = setup_cl(&device, &context, &command_queue);
    if (ret != CL_SUCCESS)
        goto out;

    cl_mem directions_buf;
    ret = generate_directions(&directions_buf);
    if (ret != CL_SUCCESS)
        goto cleanup_context;

    cl_float3 *image;
    ret = render(&image, &directions_buf);
    if (ret != CL_SUCCESS)
        goto cleanup;

    FILE *image_file = fopen("result.pgm", "wb");
    // write the magic number, dimensions, and max greyscale value
    fprintf(image_file, "P3\n%d %d\n%d\n", WIDTH, HEIGHT, 255);

    for (size_t i = 0; i < HEIGHT * WIDTH; i++)
    {
        fprintf(image_file, "%d %d %d ", convert_pixel(image[i].x), convert_pixel(image[i].y), convert_pixel(image[i].z));
    }

    fclose(image_file);

cleanup:
    clReleaseMemObject(directions_buf);
    free(image);
cleanup_context:
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
out:
    if (ret != CL_SUCCESS)
        fprintf(stderr, "Program exited with non-zero exit code: '%d'.\n", ret);
    else
        printf("Program exited successfully.\n");

    return ret;
}
