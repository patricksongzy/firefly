#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "gpulib.h"

static const cl_float4 SIGN_XYZ = {-1, 1, -1, 1};
static const cl_float4 SIGN_XZY = {-1, -1, 1, 1};
static const cl_float4 SIGN_YXZ = {1, -1, -1, 1};

cl_float4 multiply_quat(const cl_float4 lhs, const cl_float4 rhs);
cl_float4 multiply_quat_components(const cl_float4 q, const float s);
cl_float4 divide_quat_components(const cl_float4 q, const float s);
cl_float4 conjugate_quat(const cl_float4 q);
float quat_magnitude(const cl_float4 q);
cl_float4 norm_quat(const cl_float4 q);
cl_float4 euler_to_quat(const cl_float3 euler, const char *order);
cl_float4 rotate_quat(const cl_float4 rotation, const cl_float4 unrotated);

#endif
