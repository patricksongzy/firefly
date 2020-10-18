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

kernel void get_directions(global float3 *camera_directions, const float4 camera_world_quat, const float z_distance, const uint height, const uint width)
{
    size_t i = get_global_id(0);
    size_t x = i % width;
    size_t y = i / width;

    float4 screen_coordinates = (float4){x - width / 2.0f, y - height / 2.0f, z_distance, 0};
    // tangent of half the field of view gives the ratio of the opposite and adjacent of the right angle triangle one half the height of the screen height
    float3 direction = normalize(reverse_rotate_quat(camera_world_quat, screen_coordinates).xyz);

    camera_directions[i] = direction;
}