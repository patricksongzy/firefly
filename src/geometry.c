#include <math.h>
#include <stdio.h>
#include <ctype.h>

#include "geometry.h"

inline cl_float4 multiply_quat(const cl_float4 lhs, const cl_float4 rhs)
{
    cl_float4 result;
    result.w = -lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z + lhs.w * rhs.w;
    result.x = lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y + lhs.w * rhs.x;
    result.y = -lhs.x * rhs.z + lhs.y * rhs.w + lhs.z * rhs.x + lhs.w * rhs.y;
    result.z = lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w + lhs.w * rhs.z;

    return result;
}

inline cl_float4 multiply_quat_components(const cl_float4 q, const float s)
{
    cl_float4 result;
    result.w = q.w * s;
    result.x = q.x * s;
    result.y = q.y * s;
    result.z = q.z * s;

    return result;
}

inline cl_float4 divide_quat_components(const cl_float4 q, const float s)
{
    cl_float4 result;
    result.w = q.w / s;
    result.x = q.x / s;
    result.y = q.y / s;
    result.z = q.z / s;

    return result;
}

inline cl_float4 conjugate_quat(const cl_float4 q)
{
    cl_float4 result;
    result.w = q.w;
    result.x = -q.x;
    result.y = -q.y;
    result.z = -q.z;

    return result;
}

inline float quat_magnitude(const cl_float4 q)
{
    return sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
}

inline cl_float4 norm_quat(const cl_float4 q)
{
    return divide_quat_components(q, quat_magnitude(q));
}

cl_float4 euler_to_quat(const cl_float3 euler, const char *order)
{
    float roll = euler.x;
    float pitch = euler.y;
    float yaw = euler.z;

    cl_float4 sign;
    switch (tolower(order[1]))
    {
    case 'x':
        sign = SIGN_YXZ;
        
        if (tolower(order[0]) == 'z')
            sign = multiply_quat_components(sign, -1.0);

        break;
    case 'z':
        sign = SIGN_XZY;

        if (tolower(order[0]) == 'y')
            sign = multiply_quat_components(sign, -1.0);
        
        break;
    case 'y':
    default:
        sign = SIGN_XYZ;

        if (tolower(order[0]) == 'z')
            sign = multiply_quat_components(sign, -1.0);

        break;
    }

    float cp = cos(pitch * 0.5);
    float sp = sin(pitch * 0.5);
    float cr = cos(roll * 0.5);
    float sr = sin(roll * 0.5);
    float cy = cos(yaw * 0.5);
    float sy = sin(yaw * 0.5);

    cl_float4 quaternion;
    quaternion.w = cp * cr * cy + sign.w * sp * sr * sy;
    quaternion.x = cp * sr * cy + sign.x * sp * cr * sy;
    quaternion.y = sp * cr * cy + sign.y * cp * sr * sy;
    quaternion.z = cp * cr * sy + sign.z * sp * sr * cy;

    return quaternion;
}

cl_float4 rotate_quat(const cl_float4 rotation, const cl_float4 unrotated)
{
    cl_float4 conjugate = conjugate_quat(rotation);
    return multiply_quat(rotation, multiply_quat(unrotated, conjugate));
}
