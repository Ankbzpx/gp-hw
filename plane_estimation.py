import numpy as np
import jax
from jax import numpy as jnp, vmap, jit
import open3d as o3d
from normal_estimation import normalize
from joblib import Memory

import polyscope as ps
import polyscope.imgui as psim
from icecream import ic


memory = Memory("__pycache__", verbose=0)


def huber_loss(d, eps):
    return jnp.where(
        jnp.abs(d) > eps, eps * (jnp.abs(d) - 0.5 * eps), 0.5 * jnp.square(d)
    ).sum()


@jit
def fit_plane(x):
    centroid = x.mean(0)
    _, _, v_t = jnp.linalg.svd(x - centroid)
    n = v_t[-1]
    n = n / (jnp.linalg.norm(n) + 1e-8)
    c = -n @ centroid
    return n, c


@jit
def fit_plane_ransac_iter(key, pc, eps):
    pc_pick = jax.random.choice(key, pc, shape=(3,), replace=False)
    n, c = fit_plane(pc_pick)
    d = pc @ n + c
    inliner_mask = jnp.abs(d) < eps
    return inliner_mask, inliner_mask.sum()


@memory.cache
def fit_plane_ransac(pc, eps, K=1000):
    key = jax.random.PRNGKey(0)
    mask, count = vmap(fit_plane_ransac_iter, in_axes=(0, None, None))(
        jax.random.split(key, K), pc, eps
    )
    idx_best = jnp.argmax(count)
    inliers = pc[mask[idx_best]]

    n, c = fit_plane(inliers)
    mask = jnp.abs(pc @ n + c) < eps
    return n, c, mask


@memory.cache
def load_and_downsample_pc(pc_path):
    pc_o3d = o3d.io.read_point_cloud(pc_path)
    pc = np.asarray(pc_o3d.points)
    pc_proj = pc[:, :2]

    dim_min = np.min(pc_proj.max(0) - pc_proj.min(0))
    voxel_size = dim_min / 128

    pc_o3d = pc_o3d.voxel_down_sample(voxel_size=voxel_size)
    # pc_o3d = pc_o3d.farthest_point_down_sample(16384)
    pc = np.asarray(pc_o3d.points)
    return pc, voxel_size


@memory.cache
def cluster_planes(pc, eps, n_clusters=20):
    pc_clusters = []
    for _ in range(n_clusters):
        n, c, mask = fit_plane_ransac(pc, eps)
        pc_clusters.append(pc[mask])
        pc = pc[jnp.logical_not(mask)]
        if len(pc) < 3:
            break
    return pc_clusters, pc


def main():
    max_clusters = 50

    idx_pick = 0
    tags = ["01", "02", "03", "04", "05"]

    for tag in tags:
        print(tag)
        pc_path = f"assets/pc/{tag}.ply"
        pc, voxel_size = load_and_downsample_pc(pc_path)
        eps = voxel_size
        pc_clusters, reminder = cluster_planes(pc, eps, max_clusters)

    pc_path = f"assets/pc/{tags[idx_pick]}.ply"

    pc, voxel_size = load_and_downsample_pc(pc_path)
    eps = voxel_size
    pc_clusters, reminder = cluster_planes(pc, eps, max_clusters)

    K = max_clusters

    def callback():
        nonlocal idx_pick, group, K, pc_clusters, reminder
        c0, idx_pick = psim.Combo("Models", idx_pick, tags)
        c1, K = psim.SliderInt("K", K, 1, max_clusters)

        if c0:
            pc_path = f"assets/pc/{tags[idx_pick]}.ply"
            pc, voxel_size = load_and_downsample_pc(pc_path)
            eps = voxel_size
            pc_clusters, reminder = cluster_planes(pc, eps, max_clusters)
            ps.reset_camera_to_home_view()
            for i in range(len(pc_clusters)):
                ps.register_point_cloud(f"plane_{i}", pc_clusters[i]).add_to_group(
                    group
                )
                ps.register_point_cloud("reminder", reminder)

        if c1:
            for i in range(max_clusters):
                pc_tag = f"plane_{i}"

                if i >= len(pc_clusters):
                    return

                if i >= K:
                    if ps.has_point_cloud(pc_tag):
                        ps.remove_point_cloud(pc_tag)
                else:
                    ps.register_point_cloud(pc_tag, pc_clusters[i]).add_to_group(group)

        io = psim.GetIO()
        if io.MouseClicked[1]:
            screen_coords = io.MousePos
            pick_result = ps.pick(screen_coords=screen_coords)

            if pick_result.is_hit and pick_result.structure_name.startswith("plane"):
                pc_viz = ps.get_point_cloud(pick_result.structure_name)
                pc_viz.set_enabled(False)

    ps.init()
    ps.set_user_callback(callback)
    group = ps.create_group("planes")
    for i in range(len(pc_clusters)):
        ps.register_point_cloud(f"plane_{i}", pc_clusters[i]).add_to_group(group)
    ps.register_point_cloud("reminder", reminder)
    ps.show()


if __name__ == "__main__":
    main()
