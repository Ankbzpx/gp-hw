import jax
from jax import numpy as jnp, vmap, jit

import polyscope as ps
import polyscope.imgui as psim

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


@jit
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


def main():
    origin = jnp.array([0.0, 0.0, 0.0])
    center = jnp.array([1.0, 1.0, 1.0])

    t_max = 3.0
    n_rays = 10
    n_triangles = 9
    angle = 5

    tri_seed = 0
    ray_seed = 0

    def demo(t_max, n_rays, n_triangles, angle, tri_seed, ray_seed):
        key_tri = jax.random.PRNGKey(tri_seed)
        V_tris = (
            0.5 * jax.random.normal(key_tri, (3 * n_triangles, 3)) + center[None, :]
        )
        F_tris = 3 * jnp.arange(n_triangles)[:, None] + jnp.array([0, 1, 2])[None, :]

        phi_ref, theta_ref = cartesian_to_spherical(center)

        key_ray = jax.random.PRNGKey(ray_seed)
        key_phi, key_theta = jax.random.split(key_ray)
        phis = phi_ref + jnp.deg2rad(angle) * jax.random.normal(key_phi, (n_rays))
        thetas = theta_ref + jnp.deg2rad(angle) * jax.random.normal(key_theta, (n_rays))
        ray_targets = t_max * vmap(spherical_to_cartesian)(phis, thetas)
        ray_origins = jnp.repeat(origin[None, :], n_rays, axis=0)
        ray_dirs = vmap(normalize)(ray_targets - ray_origins)

        V_ray = jnp.stack([ray_origins, ray_targets], axis=1).reshape(-1, 3)
        E_ray = jnp.stack([2 * jnp.arange(n_rays), 2 * jnp.arange(n_rays) + 1], axis=-1)

        res = vmap(
            vmap(ray_triangle_intersection, in_axes=(None, None, 0)),
            in_axes=(0, 0, None),
        )(ray_origins, ray_dirs, V_tris[F_tris])

        @jit
        def validate(res):
            t = res[0]
            u = res[1]
            v = res[2]

            return jnp.logical_and(
                jnp.logical_and(
                    jnp.logical_and(
                        jnp.logical_and(t >= 0, t <= t_max),
                        jnp.logical_and(u >= 0, u <= 1),
                    ),
                    jnp.logical_and(v >= 0, v <= 1),
                ),
                jnp.logical_and((1 - u - v) >= 0, (1 - u - v) <= 1),
            )

        mask = vmap(vmap(validate))(res)

        hits = ray_origins[:, None, :] + res[..., 0, None] * ray_dirs[:, None, :]
        hits = hits[mask]

        return V_tris, F_tris, V_ray, E_ray, hits

    V_tris, F_tris, V_ray, E_ray, hits = demo(
        t_max, n_rays, n_triangles, angle, tri_seed, ray_seed
    )

    def callback():
        nonlocal t_max, n_rays, n_triangles, angle, tri_seed, ray_seed

        c0, t_max = psim.SliderFloat("t_max", t_max, 0.1, 5.0)
        c1, n_rays = psim.SliderInt("n_rays", n_rays, 1, 20)
        c2, n_triangles = psim.SliderInt("n_triangles", n_triangles, 1, 20)
        c3, angle = psim.SliderFloat("angle", angle, 0, 20)
        c4, tri_seed = psim.SliderInt("tri_seed", tri_seed, 0, 20)
        c5, ray_seed = psim.SliderInt("ray_seed", ray_seed, 0, 20)

        update = c0 | c1 | c2 | c3 | c4 | c5
        if update:
            V_tris, F_tris, V_ray, E_ray, hits = demo(
                t_max, n_rays, n_triangles, angle, tri_seed, ray_seed
            )
            ps.register_point_cloud("hits", hits)
            ps.register_surface_mesh("V_tris", V_tris, F_tris)
            ps.register_curve_network("ray", V_ray, E_ray, radius=1e-3)

    ps.init()
    ps.set_user_callback(callback)
    ps.register_point_cloud("origin", origin[None, :])
    ps.register_point_cloud("hits", hits)
    ps.register_surface_mesh("V_tris", V_tris, F_tris)
    ps.register_curve_network("ray", V_ray, E_ray, radius=1e-3)
    ps.show()


if __name__ == "__main__":
    main()
