#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "gpulib.h"
#include "geometry.h"

#define EPSILON 1E-5

int approximatelty_equal(float out, float target)
{
    int is_close = fabs(target - out) <= EPSILON * fmax(fabs(out), fabs(target));

    if (!is_close)
        printf("%f is not equal to %f.\n", out, target);

    return is_close;
}

void test_euler_to_quaternion_xyz(void)
{
    cl_float3 euler = (cl_float4){0.5, 0.8, 0.2};
    cl_float4 target = (cl_float4){0.2644041, 0.3526778, 0.1849564, 0.8783507};
    cl_float4 quaternion = euler_to_quat(euler, "xyz");

    assert(approximatelty_equal(quaternion.w, target.w));
    assert(approximatelty_equal(quaternion.x, target.x));
    assert(approximatelty_equal(quaternion.y, target.y));
    assert(approximatelty_equal(quaternion.z, target.z));
}

void test_euler_to_quaternion_zyx(void)
{
    cl_float3 euler = (cl_float4){0.5, 0.8, 0.2};
    cl_float4 target = (cl_float4){0.1890673, 0.3981767, -0.0067682, 0.8975873};
    cl_float4 quaternion = euler_to_quat(euler, "zyx");

    assert(approximatelty_equal(quaternion.w, target.w));
    assert(approximatelty_equal(quaternion.x, target.x));
    assert(approximatelty_equal(quaternion.y, target.y));
    assert(approximatelty_equal(quaternion.z, target.z));
}

void test_euler_to_quaternion_rand(void)
{
    cl_float3 euler = (cl_float4){0.5, 0.8, 0.2};
    cl_float4 target = (cl_float4){0.2644041, 0.3526778, 0.1849564, 0.8783507};
    cl_float4 quaternion = euler_to_quat(euler, "abc");

    assert(approximatelty_equal(quaternion.w, target.w));
    assert(approximatelty_equal(quaternion.x, target.x));
    assert(approximatelty_equal(quaternion.y, target.y));
    assert(approximatelty_equal(quaternion.z, target.z));
}

void test_multiply_quat(void)
{
    cl_float4 lhs = (cl_float4){0.5, 0.6, 0.9, 0.2};
    cl_float4 rhs = (cl_float4){0.3, 0.2, 0.5, 0.8};

    cl_float4 target = (cl_float4){0.58, 0.54, 0.74, -0.56};
    cl_float4 quaternion = multiply_quat(lhs, rhs);

    assert(approximatelty_equal(quaternion.w, target.w));
    assert(approximatelty_equal(quaternion.x, target.x));
    assert(approximatelty_equal(quaternion.y, target.y));
    assert(approximatelty_equal(quaternion.z, target.z));
}

void test_rotate_quat(void)
{
    cl_float3 reference_rotation = (cl_float3){0, -0.2, 0.2};
    cl_float3 local_rotation = (cl_float3){0, 0.2, 0.5};

    cl_float4 reference_quat = euler_to_quat(reference_rotation, "xyz");
    cl_float4 local_quat = euler_to_quat(local_rotation, "xyz");

    cl_float4 target = (cl_float4){-0.044016, 0.099709, 0.242252, 0.964072};
    cl_float4 transformed = rotate_quat(reference_quat, local_quat);
    assert(approximatelty_equal(transformed.x, target.x));
    assert(approximatelty_equal(transformed.y, target.y));
    assert(approximatelty_equal(transformed.z, target.z));
    assert(approximatelty_equal(transformed.w, target.w));
}

int main(void)
{
    /*
     * Test geometry functions
     */
    test_euler_to_quaternion_xyz();
    test_euler_to_quaternion_zyx();
    test_euler_to_quaternion_rand();

    test_multiply_quat();

    test_rotate_quat();
}
