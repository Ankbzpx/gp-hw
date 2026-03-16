import igl
from normal_estimation import normalize_aabb, normalize
import jax
from jax import Array, jit, numpy as jnp, vmap
import numpy as np
import scipy
from joblib import Memory
from functools import partial
from typing import Callable

import polyscope as ps
import polyscope.imgui as psim

from icecream import ic

memory = Memory("__pycache__", verbose=0)


# Remove unreference vertices and assign new vertex indices
def rm_unref_vertices(V, F):
    V_unique, V_unique_idx, V_unique_idx_inv = np.unique(
        F.flatten(), return_index=True, return_inverse=True
    )
    V_id_new = np.arange(len(V_unique))
    V_map = V_id_new[np.argsort(V_unique_idx)]
    V_map_inv = np.zeros((np.max(V_map) + 1,), dtype=np.int64)
    V_map_inv[V_map] = V_id_new

    F = V_map_inv[V_unique_idx_inv].reshape(F.shape)
    V = V[V_unique][V_map]

    return V, F


@jit
@partial(vmap)
def per_face_basis(verts):
    e01 = verts[1] - verts[0]
    e02 = verts[2] - verts[0]
    fn = normalize(jnp.cross(e01, e02))
    u = normalize(e01)
    v = normalize(jnp.cross(e01, fn))

    # T_l_w
    T = jnp.stack([u, v])

    return fn, T


def surface_vertex_topology(V, F):
    E = np.stack(
        [
            np.stack([F[:, 0], F[:, 1]], -1),
            np.stack([F[:, 1], F[:, 2]], -1),
            np.stack([F[:, 2], F[:, 0]], -1),
        ],
        1,
    ).reshape(-1, 2)

    # Use row-wise unique to filter boundary and nonmanifold vertices
    E_row_sorted = np.sort(E, axis=1)
    _, ue_inv, ue_count = np.unique(
        E_row_sorted, axis=0, return_counts=True, return_inverse=True
    )

    V_boundary = np.full((len(V)), False)
    V_boundary[list(np.unique(E[(ue_count == 1)[ue_inv]][:, 0]))] = True

    V_nonmanifold = np.full((len(V)), False)
    V_nonmanifold[list(np.unique(E[(ue_count > 2)[ue_inv]][:, 0]))] = True

    return E, V_boundary, V_nonmanifold


def build_traversal_graph(V, F):
    V, F = rm_unref_vertices(V, F)
    E, V_boundary, V_nonmanifold = surface_vertex_topology(V, F)

    E_id = np.arange(F.size)
    # Not sure if there is a more efficient approach
    E2Eid = dict([(f"{e[0]}_{e[1]}", e_id) for e, e_id in zip(E, E_id)])

    def opposite_edge_id(e_id):
        v0, v1 = E[e_id]
        key = f"{v1}_{v0}"
        return E2Eid[key] if key in E2Eid else -1

    E2E = -np.ones(F.size, dtype=np.int64)
    E2E[E_id] = np.array(list(map(opposite_edge_id, E_id)))

    # build V2E map
    # https://stackoverflow.com/questions/38277143/sort-2d-numpy-array-lexicographically
    E_sort_idx = np.lexsort(E.T[::-1])
    E_sorted = E[E_sort_idx]
    E_id_sorted = E_id[E_sort_idx]

    # Since edges are directed, `e_count` directly match the number of incident faces
    _, e_count = np.unique(E_sorted[:, 0], return_counts=True)
    pad_width = np.max(e_count)

    split_indices = np.cumsum(e_count)[:-1]
    V2E_list = np.split(E_id_sorted, split_indices)
    V2E = np.array(
        [np.concatenate([el, -np.ones(pad_width - len(el))]) for el in V2E_list]
    ).astype(np.int64)

    return V, F, E, V2E, E2E, V_boundary, V_nonmanifold


@jit
def prev_edge_id(e_id):
    return jnp.where(e_id % 3 == 0, e_id + 2, e_id - 1)


@jit
def next_edge_id(e_id):
    return jnp.where(e_id % 3 == 2, e_id - 2, e_id + 1)


@jit
@partial(vmap, in_axes=(0, None, None))
def edge_call(e_id, f: Callable, val_default: Array):
    return jnp.where(e_id == -1, val_default, f(e_id))


@jit
@partial(vmap, in_axes=(0, None, None))
def one_ring_traversal(e_ids, func: Callable, val_default: Array):
    return edge_call(e_ids, func, val_default)


# "Computing Vertex Normals from Polygonal Facets" by Grit Thuermer and Charles A. Wuethrich, JGT 1998, Vol 3
@jit
def angle_weighted_face_normal(e_id, V, E, FN):
    f_id = e_id // 3
    cur_edge = E[e_id]
    prev_edge = E[prev_edge_id(e_id)]

    d_0 = V[cur_edge[1]] - V[cur_edge[0]]
    d_1 = V[prev_edge[0]] - V[prev_edge[1]]

    angle = jnp.arccos(jnp.dot(d_0, d_1) / jnp.linalg.norm(d_0) / jnp.linalg.norm(d_1))

    angle = jnp.where(jnp.isnan(angle), 0, angle)

    return angle * FN[f_id]


@jit
def per_vertex_normal(V, E, V2E, FN):
    vn = one_ring_traversal(
        V2E,
        jax.tree_util.Partial(angle_weighted_face_normal, V=V, E=E, FN=FN),
        jnp.zeros((3,)),
    )
    vn = vmap(normalize)(jnp.sum(vn, 1))
    return vn


# Assume normal to be tangent plane, x axis is one of its connected edges
@jit
@partial(vmap, in_axes=(0, None, None, None))
def local_vertex_basis(eid, VN, V, E):
    v0, v1 = E[eid[0]]
    u = normalize(V[v1] - V[v0])
    v = jnp.cross(VN[v0], u)
    T = jnp.stack([u, v])
    return T


@jit
def per_vertex_basis(V, E, V2E, FN):
    VN = per_vertex_normal(V, E, V2E, FN)
    T = local_vertex_basis(V2E, VN, V, E)
    return VN, T


# Get rotation matrix from a to b. Reference: https://math.stackexchange.com/a/897677
@jit
def rotate_coplanar(a, b):
    cos = jnp.dot(a, b)
    cross = jnp.cross(a, b)
    sin = jnp.linalg.norm(cross)

    # pure rotation matrix
    G = jnp.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])

    # transformation of coordinate: a, normal (a x b), binormal
    F = jnp.array([a, normalize(b - cos * a), normalize(cross)])

    return F.T @ G @ F


# Reference: https://en.wikipedia.org/wiki/Heron%27s_formula
@jit
def face_area(verts):
    v0, v1, v2 = verts

    a = jnp.linalg.norm(v1 - v0)
    b = jnp.linalg.norm(v2 - v0)
    c = jnp.linalg.norm(v2 - v1)

    area_2 = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    return jnp.sqrt(area_2) / 4


# Reference: https://www.alecjacobson.com/weblog/?p=1146
@jit
def hybrid_area(a, b, c):
    e0 = a - b
    e1 = a - c
    e2 = b - c

    e0_norm = jnp.linalg.norm(e0)
    e1_norm = jnp.linalg.norm(e1)
    e2_norm = jnp.linalg.norm(e2)

    cos_a = jnp.einsum("i,i", e0, e1) / e0_norm / e1_norm
    cos_b = jnp.einsum("i,i", -e0, e2) / e0_norm / e2_norm
    cos_c = jnp.einsum("i,i", -e1, -e2) / e1_norm / e2_norm
    cos_sum = cos_a + cos_b + cos_c

    area = 0.5 * jnp.linalg.norm(jnp.cross(e0, e1))

    is_obtuse = (cos_a < 0.0) | (cos_b < 0.0) | (cos_c < 0.0)
    is_obtuse_angle = cos_a < 0.0

    area = jnp.where(is_obtuse, 0.25 * area, 0.5 * area * (cos_c + cos_b) / cos_sum)
    area = jnp.where(is_obtuse_angle, 2 * area, area)

    return area


@jit
def vertex_area(e_id, E, V):
    cur_edge = E[e_id]
    prev_edge = E[prev_edge_id(e_id)]
    return hybrid_area(V[cur_edge[0]], V[cur_edge[1]], V[prev_edge[0]])


@jit
def vertex_angle(e_id, E, V):
    cur_edge = E[e_id]
    prev_edge = E[prev_edge_id(e_id)]

    d_0 = V[cur_edge[1]] - V[cur_edge[0]]
    d_1 = V[prev_edge[0]] - V[prev_edge[1]]
    angle = jnp.arccos(jnp.dot(d_0, d_1) / jnp.linalg.norm(d_0) / jnp.linalg.norm(d_1))
    return angle


@jit
def principal_curvature(T, TM):
    eigvals, eigvecs = vmap(jnp.linalg.eigh)(TM)
    eigvecs = jnp.einsum("bij,bni->bnj", T, eigvecs)
    return eigvals, eigvecs


@jit
def cotangent(ei, ej, E, V, FA):
    A = V[E[ej][0]]
    B = V[E[ej][1]]
    C = V[E[ei][1]]
    area = FA[ei // 3]

    len2 = lambda x: jnp.einsum("i,i", x, x)

    a = len2(B - C)
    b = len2(A - C)
    c = len2(B - A)

    return 0.25 * (b + c - a) / area


# For a boundary vertex of n adjacent vertices, only n-1 half-edges are included in V2E
# Here we retrieve the missing one to support cotangent weight calculation
@jit
def boundary_auxillary_edge(eids, E2E):
    ref_eid = jnp.where(eids == -1, -1, vmap(prev_edge_id)(eids))
    src_eid = jnp.where(eids == -1, -1, E2E[eids])
    in_bool = vmap(jnp.isin, in_axes=(0, None))(ref_eid, src_eid) == 0
    e_aux = ref_eid[jnp.argwhere(in_bool, size=1)[0][0]]
    # Assume boundary has less valence
    e_replace = jnp.argwhere(eids == -1, size=1)[0][0]
    return eids.at[e_replace].set(e_aux)


@jit
def cotangent_edge_weight(e_id, E, E2E, V, FA):
    cot_a = cotangent(e_id, prev_edge_id(e_id), E, V, FA)
    e_id_op = E2E[e_id]
    cot_b = jnp.where(
        e_id_op == -1, 0.0, cotangent(e_id_op, prev_edge_id(e_id_op), E, V, FA)
    )
    return 0.5 * (cot_a + cot_b)


def cotangent_weight(V, E, FA, V2E, E2E, V_boundary):
    V2E_aux = np.copy(V2E)
    V2E_aux[V_boundary] = vmap(boundary_auxillary_edge, in_axes=(0, None))(
        V2E[V_boundary], E2E
    )
    return one_ring_traversal(
        V2E_aux,
        jax.tree_util.Partial(cotangent_edge_weight, E=E, E2E=E2E, V=V, FA=FA),
        0.0,
    )


@jit
def cotangent_laplacian_entry(ws, e_ids, E):
    edges = E[e_ids]
    vid = edges[0, 0]
    vid_i = edges[:, 0]
    vid_j = edges[:, 1]
    w_sum = jnp.sum(ws)

    idx_i = jnp.concat([jnp.array([vid]), vid_i])
    idx_j = jnp.concat([jnp.array([vid]), vid_j])
    weights = jnp.concat([-jnp.array([w_sum]), ws])
    return idx_i, idx_j, weights


@jit
def uniform_laplacian_entry(e_ids, E):
    edges = E[e_ids]
    vid = edges[0, 0]
    vid_i = edges[:, 0]
    vid_j = edges[:, 1]

    ws = jnp.where(e_ids != -1.0, 1.0, 0)
    w_sum = jnp.sum(ws)

    idx_i = jnp.concat([jnp.array([vid]), vid_i])
    idx_j = jnp.concat([jnp.array([vid]), vid_j])
    weights = jnp.concat([-jnp.array([w_sum]), ws])
    return idx_i, idx_j, weights


@jit
def directional_curvature(xi, xj, n):
    return 2 * jnp.dot(xi - xj, n) / jnp.dot(xi - xj, xi - xj)


@jit
def project_n(x, n):
    return x - jnp.dot(x, n) * n


@jit
def direction(xi, xj, n):
    return normalize(project_n(xj - xi, n))


# Reference: https://www.multires.caltech.edu/pubs/diffGeoOps.pdf
@jit
def curvature_tensor(e_ids, E, V, VN):
    edge = E[e_ids]

    xi = V[edge[:, 0]]
    xj = V[edge[:, 1]]
    n = VN[edge[:, 0]]

    ei = normalize(project_n(xj[0] - xi[0], n[0]))
    ej = normalize(jnp.cross(ei, n[0]))
    basis = jnp.stack([ei, ej])

    kappa = vmap(directional_curvature)(xi, xj, n)
    d = vmap(direction)(xi, xj, n) @ basis.T

    A = jnp.stack(
        [d[:, 0] * d[:, 0], 2 * d[:, 0] * d[:, 1], d[:, 1] * d[:, 1]], axis=-1
    )

    A = jnp.where(e_ids[:, None] != -1, A, 0)
    kappa = jnp.where(e_ids != -1, kappa, 0)
    X, _, _, _ = jnp.linalg.lstsq(A, kappa, rcond=None)

    TM = jnp.array([[X[0], X[1]], [X[1], X[2]]])

    eigvals, eigvecs = jnp.linalg.eigh(TM)

    return eigvals[0], eigvals[1], eigvecs[:, 0] @ basis, eigvecs[:, 1] @ basis


@memory.cache
def compute_laplacian(model_name):
    V, F = igl.read_triangle_mesh(f"assets/{model_name}")
    V = normalize_aabb(V)

    V, F, E, V2E, E2E, V_boundary, V_nonmanifold = build_traversal_graph(V, F)

    NV = len(V)
    FN, T_f = per_face_basis(V[F])
    VN, T_v = per_vertex_basis(V, E, V2E, FN)
    FA = vmap(face_area)(V[F])
    Ws = cotangent_weight(V, E, FA, V2E, E2E, V_boundary)
    VA = one_ring_traversal(V2E, jax.tree_util.Partial(vertex_area, E=E, V=V), 0.0)
    VA = np.asarray(VA.sum(1))
    M_inv = scipy.sparse.diags(1.0 / VA, 0)

    vert_angles = one_ring_traversal(
        V2E, jax.tree_util.Partial(vertex_angle, E=E, V=V), 0.0
    )
    kappa_G = M_inv @ (2 * jnp.pi - vert_angles.sum(1))

    kappa_1, kappa_2, dir_1, dir_2 = vmap(
        curvature_tensor, in_axes=(0, None, None, None)
    )(V2E, E, V, VN)

    _, _, weights_uniform = vmap(uniform_laplacian_entry, in_axes=(0, None))(V2E, E)

    idx_i, idx_j, weights_cotangent = vmap(
        cotangent_laplacian_entry, in_axes=(0, 0, None)
    )(Ws, V2E, E)
    idx_i = idx_i.reshape(
        -1,
    )
    idx_j = idx_j.reshape(
        -1,
    )
    weights_cotangent = weights_cotangent.reshape(
        -1,
    )
    weights_uniform = weights_uniform.reshape(
        -1,
    )

    mask = weights_cotangent != 0

    idx_i = idx_i[mask]
    idx_j = idx_j[mask]
    weights_cotangent = weights_cotangent[mask]
    weights_uniform = weights_uniform[mask]

    L = scipy.sparse.coo_array(
        (weights_cotangent, (idx_i, idx_j)), shape=(NV, NV)
    ).tocsc()

    A = scipy.sparse.coo_array(
        (weights_uniform, (idx_i, idx_j)), shape=(NV, NV)
    ).tocsc()

    kappa_H = vmap(jnp.linalg.norm)(0.5 * M_inv @ L @ V)

    VN_L = -L @ V
    VN_A = -A @ V

    # kappa_G = kappa_1 * kappa_2
    # kappa_H = 0.5 * (kappa_1 + kappa_2)

    return (
        V,
        F,
        kappa_G,
        kappa_H,
        kappa_1,
        dir_1,
        kappa_2,
        dir_2,
        VN,
        VN_L,
        VN_A,
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
        compute_laplacian(model_name)

    V, F, kappa_G, kappa_H, kappa_1, dir_1, kappa_2, dir_2, VN, VN_L, VN_A = (
        compute_laplacian(model_list[model_idx])
    )

    def callback():
        nonlocal model_idx

        c0, model_idx = psim.Combo("Models", model_idx, model_list)

        if c0:
            V, F, kappa_G, kappa_H, kappa_1, dir_1, kappa_2, dir_2, VN, VN_L, VN_A = (
                compute_laplacian(model_list[model_idx])
            )
            ps.remove_all_structures()
            ms_viz = ps.register_surface_mesh("ms", V, F)
            ms_viz.add_vector_quantity("0 Vertex normal", VN)
            ms_viz.add_vector_quantity("1 Mean curvature normal (cotangent)", VN_L)
            ms_viz.add_vector_quantity("2 Mean curvature normal (uniform)", VN_A)
            ms_viz.add_scalar_quantity("3 Curvature Gaussian", kappa_G)
            ms_viz.add_scalar_quantity("4 Curvature Mean", kappa_H)
            ms_viz.add_vector_quantity("5 Principal direction 1", dir_1)
            ms_viz.add_scalar_quantity("5 Curvature Principal 1", kappa_1)
            ms_viz.add_vector_quantity("6 Principal direction 2", dir_2)
            ms_viz.add_scalar_quantity("6 Curvature Principal 2", kappa_2)
            ps.reset_camera_to_home_view()

    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_user_callback(callback)
    ms_viz = ps.register_surface_mesh("ms", V, F)
    ms_viz.add_vector_quantity("0 Vertex normal", VN, enabled=True)
    ms_viz.add_vector_quantity(
        "1 Mean curvature normal (cotangent)", VN_L, enabled=True
    )
    ms_viz.add_vector_quantity("2 Mean curvature normal (uniform)", VN_A)
    ms_viz.add_scalar_quantity("3 Curvature Gaussian", kappa_G, cmap="turbo")
    ms_viz.add_scalar_quantity("4 Curvature Mean", kappa_H, cmap="turbo")
    ms_viz.add_vector_quantity("5 Principal direction 1", dir_1)
    ms_viz.add_scalar_quantity("5 Curvature Principal 1", kappa_1, cmap="turbo")
    ms_viz.add_vector_quantity("6 Principal direction 2", dir_2)
    ms_viz.add_scalar_quantity("6 Curvature Principal 2", kappa_2, cmap="turbo")
    ps.show()


if __name__ == "__main__":
    main()
