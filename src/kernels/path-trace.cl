#define EPSILON 1e-2f

struct ray
{
    float3 origin;
    float3 direction;
};

struct sphere
{
    float3 position;
    float3 colour;
    float3 emission;
    float radius;
} __attribute__((packed));

inline bool intersect_sphere(const struct sphere s, const struct ray *r, float *t)
{
    float3 centre_ray = s.position - r->origin;

    // solve the quadratic, where a = 1
    float b = dot(centre_ray, r->direction);
    float c = dot(centre_ray, centre_ray) - s.radius * s.radius;

    float disc = b * b - c;

    if (disc < 0.0f)
        return false;

    float disc_root = sqrt(disc);
    if ((*t = b - disc_root) > EPSILON)
        return true;

    if ((*t = b + disc_root) > EPSILON)
        return true;

    return false;
}

inline bool intersect_scene(constant struct sphere *scene_spheres, const size_t n, const struct ray *r, size_t *hit_index, float *t)
{
    float min_distance = INFINITY;
    
    for (size_t i = 0; i < n; i++)
    {
        float hit_distance;

        if (intersect_sphere(scene_spheres[i], r, &hit_distance) && hit_distance < min_distance)
        {
            min_distance = hit_distance;
            *hit_index = i;
        }
    }

    *t = min_distance;

    return min_distance < INFINITY;
}

inline uint rand(ulong *seed)
{
    // Java Random number algorithm
    *seed = (*seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
    return *seed >> 16;
}

inline float randf(ulong *seed)
{
    return rand(seed) / (float) UINT_MAX;
}

kernel void render(global float3 *output, constant struct sphere *scene_spheres, const uint num_spheres, constant float3 *camera_directions, const float3 camera_position, const uint height, const uint width, const uint num_samples)
{
    size_t i = get_global_id(0);
    size_t x = i % width;
    size_t y = i / width;

    struct ray camera_ray;
    camera_ray.origin = camera_position;
    camera_ray.direction = camera_directions[i];

    for (size_t s = 0; s < num_samples; s++)
    {
        // not the best pseudo-random
        ulong seed = i * x * y * s;

        float3 accumulated_colour = (float3){0, 0, 0};
        float3 mask = (float3){1.0, 1.0, 1.0};
        struct ray cast_ray = camera_ray;
        for (size_t bounce = 0; bounce < 16; bounce++)
        {
            float t;
            size_t hit_index;
            if (!intersect_scene(scene_spheres, num_spheres, &cast_ray, &hit_index, &t))
                break;

            struct sphere hit_sphere = scene_spheres[hit_index];
            float3 hit_point = cast_ray.origin + cast_ray.direction * t;
            // a ray from the centre of a sphere, to the point on the surface will have the direction of the normal
            float3 normal = normalize(hit_point - hit_sphere.position);
            // normal flipping technique
            float3 oriented_normal = dot(normal, cast_ray.direction) < 0.0f ? normal : normal * -1.0f;

            float3 w = oriented_normal;
            // use the smallest component as the axis
            float3 axis = fabs(w.x) < fabs(w.y) && fabs(w.x) < fabs(w.z) ? (float3){1.0, 0, 0} : fabs(w.y) < fabs(w.z) ? (float3){0, 1.0, 0} : (float3){0, 0, 1.0};
            float3 u = normalize(cross(axis, w));
            float3 v = cross(w, u);

            float random_angle =  2 * M_PI * randf(&seed);
            float random_number = randf(&seed);
            float random_distance = sqrt(random_number);
            float3 bounce_direction = normalize(u * cos(random_angle) * random_distance + v * sin(random_angle) * random_distance + w * sqrt(1 - random_number));

            float p = max(hit_sphere.colour.x, max(hit_sphere.colour.y, hit_sphere.colour.z));
            if (bounce > 5) {
                if (randf(&seed) > p) {
                    accumulated_colour += mask * hit_sphere.emission;
                    break;
                } else {
                    hit_sphere.colour /= p;
                }
            }

            cast_ray.origin = hit_point + oriented_normal * EPSILON;
            cast_ray.direction = bounce_direction;

            accumulated_colour += mask * hit_sphere.emission;

            mask *= hit_sphere.colour;
        }

        output[i] += accumulated_colour / (float) num_samples;
    }
}
