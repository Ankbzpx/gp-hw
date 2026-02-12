import numpy as np

import jax
from jax import numpy as jnp, vmap, jit
import trimesh
import igl

import robust_laplacian

import scipy.sparse.linalg
from scipy.spatial import KDTree

import polyscope as ps
import polyscope.imgui as psim

from icecream import ic


@jit
def normalize(x):
    return x / (jnp.linalg.norm(x) + 1e-8)


@jit
def face_normal(vf):
    fn = jnp.cross(vf[1] - vf[0], vf[2] - vf[0])
    return normalize(fn)


@jit
def face_area(vf):
    fn = jnp.cross(vf[1] - vf[0], vf[2] - vf[0])
    return 0.5 * jnp.linalg.norm(fn)


@jit
def angle(v0, v1, v2):
    d0 = v1 - v0
    d1 = v2 - v0
    cos = jnp.dot(d0, d1) / (jnp.linalg.norm(d0) * jnp.linalg.norm(d0) + 1e-8)
    return jnp.arccos(jnp.clip(cos, 0, 1))


@jit
def vertex_angle(vf):
    return jnp.array(
        [
            angle(vf[0], vf[1], vf[2]),
            angle(vf[1], vf[2], vf[0]),
            angle(vf[2], vf[0], vf[1]),
        ]
    )


@jit
def normal_svd(x, neigh_idx, samples):
    x_bar = samples[neigh_idx] - x
    _, _, Vt = jnp.linalg.svd(x_bar)
    return Vt[-1]


@jit
def normal_pca(x, neigh_idx, samples):
    x_bar = samples[neigh_idx] - x
    _, eigenvectors = jnp.linalg.eigh(x_bar.T @ x_bar)
    return eigenvectors[:, 0]


def aabb_compute(V, scale=0.9):
    V_aabb_max = V.max(0, keepdims=True)
    V_aabb_min = V.min(0, keepdims=True)
    V_center = 0.5 * (V_aabb_max + V_aabb_min)
    scale = (V_aabb_max - V_center).max() / scale
    return V_center, scale, (V_aabb_max - V_aabb_min)


def normalize_aabb(V, scale=0.9):
    V_center, scale, _ = aabb_compute(V, scale)
    return (V - V_center) / scale


def main():

    model_idx = 0
    model_list = [
        "bunny.obj",
        "fandisk.ply",
        "rbf_clover_ring_two_stars.obj",
        "rocker_arm.ply",
        "cube_twist.ply",
        "terrain.obj",
        "bumpy_plane.off",
        "cactus.off",
        "camel_head.off",
        "hand.off",
    ]

    V_cube = np.array(
        [
            [-1, -1, 1],
            [1, -1, 1],
            [-1, 1, 1],
            [1, 1, 1],
            [-1, -1, -1],
            [1, -1, -1],
            [-1, 1, -1],
            [1, 1, -1],
        ]
    )

    F_cube = np.array(
        [
            [7, 6, 2],
            [2, 3, 7],
            [0, 4, 5],
            [5, 1, 0],
            [0, 2, 6],
            [6, 4, 0],
            [7, 3, 1],
            [1, 5, 7],
            [3, 2, 0],
            [0, 1, 3],
            [4, 6, 7],
            [7, 5, 4],
        ]
    )

    def compute_normals(model_name, sample_size, k, noise_level, seed, use_svd, robust):
        V, F = igl.read_triangle_mesh(f"assets/{model_name}")
        V = normalize_aabb(V)
        ms_tri = trimesh.Trimesh(V, F)
        samples, _ = trimesh.sample.sample_surface(
            mesh=ms_tri, count=sample_size, seed=seed
        )

        key = jax.random.PRNGKey(seed)
        noise = noise_level * jax.random.normal(key, (sample_size, 3))
        samples = samples + noise

        _, neigh_indices = KDTree(samples).query(samples, k=k + 1)
        neigh_indices = neigh_indices[:, 1:]

        if use_svd:
            sample_normals = vmap(normal_svd, in_axes=(0, 0, None))(
                samples, neigh_indices, samples
            )
        else:
            sample_normals = vmap(normal_pca, in_axes=(0, 0, None))(
                samples, neigh_indices, samples
            )

        h = vmap(jnp.linalg.norm)(samples[neigh_indices[:, 0]] - samples).mean().item()

        samples = np.array(samples)
        sample_normals = np.array(sample_normals)

        _, idx = KDTree(ms_tri.vertices).query(samples)

        dps = np.einsum("ni,ni->n", ms_tri.vertex_normals[idx], sample_normals)
        sample_normals = sample_normals * np.sign(dps)[:, None]

        if robust:
            L, M = robust_laplacian.point_cloud_laplacian(samples)
            solve = scipy.sparse.linalg.factorized(M + ((0.5 * h) ** 2) * L)
            dps = np.einsum("ni,ni->n", solve(sample_normals), sample_normals)
            sample_normals = sample_normals * np.sign(dps)[:, None]

        return V, F, samples, sample_normals, neigh_indices

    sample_size = 1000
    k = 30
    sigma = 0.01
    seed = 0
    use_svd = True
    robust = True

    V, F, samples, sample_normals, neigh_indices = compute_normals(
        model_list[model_idx], sample_size, k, sigma, seed, use_svd, robust
    )

    def pca_bbox(V, F):
        V_center = V.mean(0)
        V_bar = V - V_center
        _, eigvecs = np.linalg.eigh(V_bar.T @ V_bar)

        V_proj = np.einsum("bi,ij", V_bar, eigvecs)

        half_extend = 0.5 * (V_proj.max(0) - V_proj.min(0))
        center = 0.5 * (V_proj.max(0) + V_proj.min(0))

        return (V_center + center @ eigvecs.T)[None, :] + np.einsum(
            "bj, ij->bi", V_cube * half_extend[None, :], eigvecs
        )

    V_obb = pca_bbox(V, F)

    def callback():
        nonlocal \
            model_idx, \
            sample_size, \
            k, \
            sigma, \
            seed, \
            use_svd, \
            robust, \
            samples, \
            neigh_indices

        c0, model_idx = psim.Combo("Models", model_idx, model_list)
        c1, sample_size = psim.SliderInt("Sample Size", sample_size, 50, 10000)
        c2, k = psim.SliderInt("k", k, 3, 100)
        c3, sigma = psim.SliderFloat("Sigma", sigma, 0, 1e-1)
        c4, seed = psim.SliderInt("Seed", seed, 0, 20)
        c5, use_svd = psim.Checkbox("Use SVD", use_svd)
        c6, robust = psim.Checkbox("Robust", robust)

        if c0 or c1 or c2 or c3 or c4 or c5 or c6:
            ps.remove_curve_network("knn", False)
            V, F, samples, sample_normals, neigh_indices = compute_normals(
                model_list[model_idx], sample_size, k, sigma, seed, use_svd, robust
            )
            ps.register_surface_mesh("Mesh", V, F)
            ps.register_point_cloud(
                "samples", samples, radius=3e-3
            ).add_vector_quantity("sample_normals", sample_normals, enabled=True)

            V_obb = pca_bbox(V, F)
            ps.register_surface_mesh("OBB", V_obb, F_cube, transparency=0.2)
            if c0:
                ps.reset_camera_to_home_view()

        io = psim.GetIO()
        if io.MouseClicked[0]:
            screen_coords = io.MousePos
            pick_result = ps.pick(screen_coords=screen_coords)

            if pick_result.is_hit and pick_result.structure_name == "samples":
                idx = pick_result.structure_data["index"]
                neigh_idx = neigh_indices[idx]
                V_neigh = samples[np.concatenate([np.array([idx]), neigh_idx])]
                E_neigh = np.stack(
                    [np.zeros((len(neigh_idx),)), np.arange(len(neigh_idx)) + 1],
                    axis=-1,
                )
                ps.register_curve_network("knn", V_neigh, E_neigh, radius=2e-3)

    ps.init()
    ps.set_user_callback(callback)
    ps.register_surface_mesh("OBB", V_obb, F_cube, transparency=0.2)
    ps.register_surface_mesh("Mesh", V, F)
    ps.register_point_cloud("samples", samples, radius=3e-3).add_vector_quantity(
        "sample_normals", sample_normals, enabled=True
    )
    ps.show()


if __name__ == "__main__":
    main()
