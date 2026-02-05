import jax
from jax import numpy as jnp, vmap, jit

import polyscope as ps
from icecream import ic


@jit
def normalize(x):
    return x / (jnp.linalg.norm(x) + 1e-8)


# Note the phi, theta have different convention as in rendering
@jit
def cartesian_to_spherical(v):
    return jnp.arctan2(jnp.sqrt((v[1] ** 2 + v[0] ** 2)), v[2]), jnp.arctan2(v[1], v[0])


@jit
def spherical_to_cartesian(phi, theta):
    return jnp.array(
        [jnp.sin(phi) * jnp.cos(theta), jnp.sin(phi) * jnp.sin(theta), jnp.cos(phi)]
    )


def ray_triangle_intersection(O, D, VF):
    V0 = VF[0]
    V1 = VF[1]
    V2 = VF[2]
    E1 = V1 - V0
    E2 = V2 - V0
    T = O - V0
    P = jnp.cross(D, E2)
    Q = jnp.cross(T, E1)
    return jnp.array([jnp.dot(Q, E2), jnp.dot(P, T), jnp.dot(Q, D)]) / jnp.dot(P, E1)


if __name__ == "__main__":
    origin = jnp.array([0.0, 0.0, 0.0])
    center = jnp.array([1.0, 1.0, 1.0])

    t_max = 3
    n_rays = 10
    n_triangles = 9

    key_tri = jax.random.PRNGKey(0)
    V_tris = 0.5 * jax.random.normal(key_tri, (3 * n_triangles, 3)) + center[None, :]
    F_tris = 3 * jnp.arange(n_triangles)[:, None] + jnp.array([0, 1, 2])[None, :]

    phi_ref, theta_ref = cartesian_to_spherical(center)

    key_ray = jax.random.PRNGKey(0)
    key_phi, key_theta = jax.random.split(key_ray)
    phis = phi_ref + jnp.deg2rad(5) * jax.random.normal(key_phi, (n_rays))
    thetas = theta_ref + jnp.deg2rad(5) * jax.random.normal(key_theta, (n_rays))
    ray_targets = t_max * vmap(spherical_to_cartesian)(phis, thetas)
    ray_origins = jnp.repeat(origin[None, :], n_rays, axis=0)
    ray_dirs = vmap(normalize)(ray_targets - ray_origins)

    V_ray = jnp.stack([ray_origins, ray_targets], axis=1).reshape(-1, 3)
    E_ray = jnp.stack([2 * jnp.arange(n_rays), 2 * jnp.arange(n_rays) + 1], axis=-1)

    res = vmap(
        vmap(ray_triangle_intersection, in_axes=(None, None, 0)), in_axes=(0, 0, None)
    )(ray_origins, ray_dirs, V_tris[F_tris])

    def validate(res):
        t = res[0]
        u = res[1]
        v = res[2]

        return jnp.logical_and(
            jnp.logical_and(
                jnp.logical_and(
                    jnp.logical_and(t >= 0, t <= t_max), jnp.logical_and(u >= 0, u <= 1)
                ),
                jnp.logical_and(v >= 0, v <= 1),
            ),
            jnp.logical_and((1 - u - v) >= 0, (1 - u - v) <= 1),
        )

    mask = vmap(vmap(validate))(res)

    hits = ray_origins[:, None, :] + res[..., 0, None] * ray_dirs[:, None, :]
    hits = hits[mask]

    ps.init()
    ps.register_point_cloud("origin", origin[None, :])
    ps.register_point_cloud("hits", hits)
    ps.register_surface_mesh("V_tris", V_tris, F_tris)
    ps.register_curve_network("ray", V_ray, E_ray, radius=1e-3)
    ps.show()
