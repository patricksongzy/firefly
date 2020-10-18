inline float4 multiply_quat(const float4 lhs, const float4 rhs)
{
    float4 result;
    result.w = -lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z + lhs.w * rhs.w;
    result.x = lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y + lhs.w * rhs.x;
    result.y = -lhs.x * rhs.z + lhs.y * rhs.w + lhs.z * rhs.x + lhs.w * rhs.y;
    result.z = lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w + lhs.w * rhs.z;

    return result;
}

inline float4 conjugate_quat(const float4 q)
{
    float4 result;
    result.w = q.w;
    result.x = -q.x;
    result.y = -q.y;
    result.z = -q.z;

    return result;
}

inline float4 reverse_rotate_quat(const float4 rotation, const float4 unrotated)
{
    float4 conjugate = conjugate_quat(rotation);
    return multiply_quat(conjugate, multiply_quat(unrotated, rotation));
}