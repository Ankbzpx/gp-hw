import polyscope as ps
import polyscope.imgui as psim
from icecream import ic
import jax
from jax import numpy as jnp, vmap, jit
import numpy as np

import igl


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
        "hand.off"
    ]

    def compute_normals(model_name):
        V, F = igl.read_triangle_mesh(f"assets/{model_name}")

        VF = V[F]
        FN = vmap(face_normal)(VF)
        FA = vmap(face_area)(VF)
        VA = vmap(vertex_angle)(VF)

        VN_uniform = jnp.zeros_like(V)
        VN_area = jnp.zeros_like(V)
        VN_angle = jnp.zeros_like(V)

        # Mapping from vertex to faces
        vids = F.reshape(
            -1,
        )
        fids = jnp.repeat(jnp.arange(len(F))[:, None], axis=1, repeats=3).reshape(
            -1,
        )

        VN_uniform = VN_uniform.at[vids].add(FN[fids])
        VN_uniform = vmap(normalize)(VN_uniform)

        VN_area = VN_uniform.at[vids].add((FA[:, None] * FN)[fids])
        VN_area = vmap(normalize)(VN_area)

        VN_angle = VN_angle.at[vids].add(
            VA.reshape(
                -1,
            )[:, None]
            * FN[fids]
        )
        VN_angle = vmap(normalize)(VN_angle)

        return V, F, FN, VN_uniform, VN_area, VN_angle

    V, F, FN, VN_uniform, VN_area, VN_angle = compute_normals(model_list[model_idx])

    def callback():
        nonlocal model_idx

        changed, model_idx = psim.Combo("Models", model_idx, model_list)
        if changed:
            V, F, FN, VN_uniform, VN_area, VN_angle = compute_normals(
                model_list[model_idx]
            )
            ms_vis = ps.register_surface_mesh("Mesh", V, F)
            ms_vis.add_vector_quantity("Face normals", FN, defined_on="faces")
            ms_vis.add_vector_quantity("Vertex normals (uniform)", VN_uniform)
            ms_vis.add_vector_quantity("Vertex normals (area)", VN_area)
            ms_vis.add_vector_quantity("Vertex normals (angle)", VN_angle)
            ps.reset_camera_to_home_view()

    ps.init()
    ps.set_user_callback(callback)
    ms_vis = ps.register_surface_mesh("Mesh", V, F)
    ms_vis.add_vector_quantity("Face normals", FN, defined_on="faces", enabled=True)
    ms_vis.add_vector_quantity("Vertex normals (uniform)", VN_uniform)
    ms_vis.add_vector_quantity("Vertex normals (area)", VN_area)
    ms_vis.add_vector_quantity("Vertex normals (angle)", VN_angle)
    ps.show()


if __name__ == "__main__":
    main()
