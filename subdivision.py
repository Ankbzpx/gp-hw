import igl
import trimesh
import numpy as np
from joblib import Memory

import polyscope as ps
import polyscope.imgui as psim
from icecream import ic

memory = Memory("__pycache__", verbose=0)

MAX_DIV = 4


def subdivide(V, F):
    # Compute connectivity
    FF, _ = igl.triangle_triangle_adjacency(F)

    vids_new = np.arange(len(F)) + len(V)

    mask_0 = FF[:, 0] != -1
    mask_1 = FF[:, 1] != -1
    mask_2 = FF[:, 2] != -1

    F_intermediate = np.vstack(
        [
            np.stack([F[:, 1], vids_new, F[:, 0]], axis=-1),
            np.stack([F[:, 2], vids_new, F[:, 1]], axis=-1),
            np.stack([F[:, 0], vids_new, F[:, 2]], axis=-1),
        ]
    )

    F_final = np.vstack(
        [
            np.stack([vids_new, F[:, 0], vids_new[FF[:, 0]]], axis=-1)[mask_0],
            np.stack([vids_new, F[:, 1], vids_new[FF[:, 1]]], axis=-1)[mask_1],
            np.stack([vids_new, F[:, 2], vids_new[FF[:, 2]]], axis=-1)[mask_2],
            np.stack([F[:, 1], vids_new, F[:, 0]], axis=-1)[np.logical_not(mask_0)],
            np.stack([F[:, 2], vids_new, F[:, 1]], axis=-1)[np.logical_not(mask_1)],
            np.stack([F[:, 0], vids_new, F[:, 2]], axis=-1)[np.logical_not(mask_2)],
        ]
    )

    # Compute new vertices
    V_bary = V[F].mean(1)

    # Update old vertices
    A = igl.adjacency_matrix(F)
    n = A.sum(0)
    a_n = (4.0 - 2.0 * np.cos(2.0 * np.pi / n)) / 9.0

    # No idea why is this needed...
    n = np.array(n).reshape(
        -1,
    )
    a_n = np.array(a_n).reshape(
        -1,
    )
    V_final = np.vstack([V * (1 - a_n)[:, None] + (A @ V) * (a_n / n)[:, None], V_bary])
    return V_final, F_intermediate, F_final


@memory.cache
def load_and_subdivide(model_name):
    model_path = f"assets/{model_name}"
    V, F = igl.read_triangle_mesh(model_path)

    V_list = []
    F_list = []
    V_sub_list = []
    F_sub_im_list = []
    F_sub_list = []

    for _ in range(MAX_DIV):
        V_sub, F_sub_im, F_sub = subdivide(V, F)

        V_list.append(V)
        F_list.append(F)
        V_sub_list.append(V_sub)
        F_sub_im_list.append(F_sub_im)
        F_sub_list.append(F_sub)

        V = V_sub
        F = F_sub

    return V_list, F_list, V_sub_list, F_sub_im_list, F_sub_list


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

    # Cache
    for model_name in model_list:
        load_and_subdivide(model_name)

    level = 0

    V_list, F_list, V_sub_list, F_sub_im_list, F_sub_list = load_and_subdivide(
        model_list[model_idx]
    )

    def callback():
        nonlocal V_list, F_list, V_sub_list, F_sub_im_list, F_sub_list, model_idx, level

        c0, model_idx = psim.Combo("Models", model_idx, model_list)
        c1, level = psim.SliderInt("Level", level, 0, MAX_DIV - 1)

        if c1 or c0:
            if c0:
                V_list, F_list, V_sub_list, F_sub_im_list, F_sub_list = (
                    load_and_subdivide(model_list[model_idx])
                )

            ps.register_surface_mesh(
                "0 mesh", V_list[level], F_list[level], edge_width=1
            )
            ps.register_surface_mesh(
                "1 mesh_sub_intermediate",
                V_sub_list[level],
                F_sub_im_list[level],
                edge_width=1,
            )
            ps.register_surface_mesh(
                "2 mesh_sub", V_sub_list[level], F_sub_list[level], edge_width=1
            )

            if c0:
                ps.reset_camera_to_home_view()

    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_user_callback(callback)
    ps.register_surface_mesh("0 mesh", V_list[level], F_list[level], edge_width=1)
    ps.register_surface_mesh(
        "1 mesh_sub_intermediate", V_sub_list[level], F_sub_im_list[level], edge_width=1
    )
    ps.register_surface_mesh(
        "2 mesh_sub", V_sub_list[level], F_sub_list[level], edge_width=1
    )
    ps.show()


if __name__ == "__main__":
    main()
