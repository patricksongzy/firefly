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

inline bool intersect_scene(constant struct sphere *scene_spheres, const size_t n, const struct ray *r, int *hit_index, float *t)
{
    float min_distance = INFINITY;
    
    for (size_t i = 0; i < n; i++)
    {
        float hit_distance;
        
        if (intersect_sphere(scene_spheres[i], r, &hit_distance) && hit_distance < min_distance && *hit_index != i)
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
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    size_t i = x + width * y;

    if (x >= width || y >= height)
        return;

    // not the best pseudo-random
    ulong seed = i;
    for (size_t t = 0; t < (size_t) (64 * randf(&seed)); t++) {
        rand(&seed);
    }

    struct ray camera_ray;
    camera_ray.origin = camera_position;
    camera_ray.direction = camera_directions[i];

    for (size_t s = 0; s < num_samples; s++)
    {
        float light_weight = 1;
        float3 accumulated_colour = (float3){0, 0, 0};
        float3 mask = (float3){1.0, 1.0, 1.0};
        struct ray cast_ray = camera_ray;
        int hit_index = -1;
        for (size_t bounce = 0; bounce < 16; bounce++)
        {
            float t;
            if (!intersect_scene(scene_spheres, num_spheres, &cast_ray, &hit_index, &t))
                break;

            struct sphere hit_sphere = scene_spheres[hit_index];

            accumulated_colour += mask * hit_sphere.emission * light_weight;

            float p = max(mask.x, max(mask.y, mask.z));
            if (bounce > 5) {
                if (randf(&seed) > p) {
                    break;
                } else {
                    mask /= p;
                }
            }

            float3 hit_point = cast_ray.origin + cast_ray.direction * t;
            // a ray from the centre of a sphere, to the point on the surface will have the direction of the normal
            float3 normal = normalize(hit_point - hit_sphere.position);
            // normal flipping technique
            float3 oriented_normal = dot(normal, cast_ray.direction) < 0.0f ? normal : normal * -1.0f;

            // create axes about the normal
            float3 w = oriented_normal;
            // use the smallest component as the axis
            float3 axis = fabs(w.x) < fabs(w.y) && fabs(w.x) < fabs(w.z) ? (float3){1.0, 0, 0} : fabs(w.y) < fabs(w.z) ? (float3){0, 1.0, 0} : (float3){0, 0, 1.0};
            float3 u = normalize(cross(axis, w));
            float3 v = cross(w, u);

            // cosine hemisphere sampling
            float random_angle =  2 * M_PI_F * randf(&seed);
            float random_number = randf(&seed);
            float random_distance = sqrt(random_number);
            float3 bounce_direction = normalize(u * cos(random_angle) * random_distance + v * sin(random_angle) * random_distance + w * sqrt(1 - random_number));
            float3 bounce_start = hit_point + oriented_normal * EPSILON;

            cast_ray.origin = bounce_start;
            cast_ray.direction = bounce_direction;

            // this snippet of code is translated from smallpt for now
            // smallpt is by Kevin Beason, released under the MIT licence
            // TODO make this code more clear
            float3 e = (float3){0, 0, 0};
            for (size_t j = 0; j < num_spheres; j++) {
                struct sphere sphere = scene_spheres[j];
                if (length(sphere.emission) == 0)
                    continue;
                
                float3 light_w = sphere.position - bounce_start;
                float3 light_axis = fabs(light_w.x) < fabs(light_w.y) && fabs(light_w.x) < fabs(light_w.z) ? (float3){1.0, 0, 0} : fabs(light_w.y) < fabs(light_w.z) ? (float3){0, 1.0, 0} : (float3){0, 0, 1.0};
                float3 light_u = normalize(cross(light_axis, light_w));
                float3 light_v = cross(light_w, light_u);
                float cos_a_max = sqrt(1 -  sphere.radius * sphere.radius / dot(bounce_start - sphere.position, bounce_start - sphere.position));
                float eps1 = randf(&seed);
                float eps2 = randf(&seed);
                float cos_a = 1 - eps1 + eps1 * cos_a_max;
                float sin_a = sqrt(1 - cos_a * cos_a);
                float phi = 2 * M_PI_F * eps2;
                float3 l = normalize(light_u * cos(phi) * sin_a + light_v * sin(phi) * sin_a + light_w * cos_a);
                int hit_light = -1;
                float t_light;
                if (intersect_scene(scene_spheres, num_spheres, &(struct ray){bounce_start, l}, &hit_light, &t_light)) {
                    if (hit_light == j && hit_light != hit_index) {
                        float omega = 2 * M_PI_F * (1 - cos_a_max);
                        e += hit_sphere.colour * (sphere.emission * dot(l, oriented_normal) * omega) * M_1_PI_F;
                    }
                }
            }

            accumulated_colour += mask * e;
            mask *= hit_sphere.colour;

            light_weight = 0;
        }

        output[i] += accumulated_colour / (float) num_samples;
    }
}
