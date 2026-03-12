import numpy as np
import igl
import polyscope as ps
import polyscope.imgui as psim
import trimesh
import networkx as nx
import scipy.sparse.linalg
from joblib import Memory
from normal_estimation import normalize_aabb

from icecream import ic


memory = Memory("__pycache__", verbose=0)


def unpack_stiffness(L):
    V_cot_adj_coo = scipy.sparse.coo_array(L)
    # We don't need diagonal
    valid_entries_mask = V_cot_adj_coo.col != V_cot_adj_coo.row
    E_i = V_cot_adj_coo.col[valid_entries_mask]
    E_j = V_cot_adj_coo.row[valid_entries_mask]
    return np.stack([E_i, E_j], axis=-1)


@memory.cache
def compute_spectral(model_name):
    V, F = igl.read_triangle_mesh(f"assets_low/{model_name}")
    mesh_tri = trimesh.Trimesh(V, F)

    A = igl.adjacency_matrix(F)
    c, _, _ = igl.connected_components(A)

    A_sum = np.array(np.sum(A, axis=1))
    A_diag = scipy.sparse.diags(
        A_sum.reshape(
            -1,
        )
    )
    U = A_diag - A

    L = igl.cotmatrix(V, F)
    L = -L

    graph = nx.from_edgelist(mesh_tri.face_adjacency)
    mst: nx.Graph = nx.minimum_spanning_tree(graph)

    FE = np.stack([np.array(e) for e in mst.edges()])
    V_bary = V[F].mean(1)
    LF = nx.laplacian_matrix(mst)

    _, eigvec_LF = scipy.sparse.linalg.eigsh(LF, k=c + 2, which="SM")
    _, eigvecs_L = scipy.sparse.linalg.eigsh(L, k=c + 2, which="SM")
    _, eigvecs_U = scipy.sparse.linalg.eigsh(U, k=c + 2, which="SM")

    E_LF = unpack_stiffness(LF)
    E_L = unpack_stiffness(L)
    E_U = unpack_stiffness(U)
    return (
        V,
        F,
        V_bary,
        FE,
        eigvec_LF[:, c:],
        E_LF,
        eigvecs_L[:, c:],
        E_L,
        eigvecs_U[:, c:],
        E_U,
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
        "hand.off",
    ]

    # Trigger cache
    for model_name in model_list:
        print(model_name)
        compute_spectral(model_name)

    V, F, V_bary, E_bary, V_mst, E_mst, V_co, E_co, V_u, E_u = compute_spectral(
        model_list[model_idx]
    )
    V_mst = normalize_aabb(V_mst)
    V_co = normalize_aabb(V_co)
    V_u = normalize_aabb(V_u)

    def callback():
        nonlocal model_idx
        c0, model_idx = psim.Combo("Models", model_idx, model_list)

        if c0:
            V, F, V_bary, E_bary, V_mst, E_mst, V_co, E_co, V_u, E_u = compute_spectral(
                model_list[model_idx]
            )
            V_mst = normalize_aabb(V_mst)
            V_co = normalize_aabb(V_co)
            V_u = normalize_aabb(V_u)

            ps.register_surface_mesh("Mesh", V, F)
            ps.register_curve_network("Mesh spectral CO", V_co, E_co)
            ps.register_curve_network("Mesh spectral U", V_u, E_u)
            ps.register_curve_network("MST", V_mst, E_mst)
            ps.register_curve_network("MST spectral", V_bary, E_bary)

    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_user_callback(callback)
    ps.register_surface_mesh("Mesh", V, F)
    ps.register_curve_network("Mesh spectral CO", V_co, E_co)
    ps.register_curve_network("Mesh spectral U", V_u, E_u)
    ps.register_curve_network("MST", V_mst, E_mst)
    ps.register_curve_network("MST spectral", V_bary, E_bary)
    ps.show()


if __name__ == "__main__":
    main()
