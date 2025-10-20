import numpy as np
import trimesh
import open3d as o3d


def save_mesh(a_trimesh, save_path):
    mesh_txt = trimesh.exchange.obj.export_obj(a_trimesh, include_normals=False, include_color=False, include_texture=False, return_texture=False, write_texture=False, resolver=None, digits=8)
    with open(save_path, "w") as fp:
        fp.write(mesh_txt)


def simplify_mesh(a_trimesh):
    a_o3d = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(a_trimesh.vertices), triangles=o3d.utility.Vector3iVector(a_trimesh.faces))
    simple_a_o3d = a_o3d.simplify_quadric_decimation(target_number_of_triangles=a_trimesh.faces.shape[0] // 20)
    simple_a_trimesh = trimesh.Trimesh(vertices=np.float32(simple_a_o3d.vertices), faces=np.int32(simple_a_o3d.triangles))

    return simple_a_trimesh


def downsample_points_from_mesh(a_trimesh, N_point=10000):
    a_o3d = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(a_trimesh.vertices), triangles=o3d.utility.Vector3iVector(a_trimesh.faces))
    a_downsampled_pcd = a_o3d.sample_points_uniformly(number_of_points=N_point)
    a_downsampled_points = np.float32(a_downsampled_pcd.points)
    assert a_downsampled_points.shape == (N_point, 3)
    return a_downsampled_points


def compute_bbox(world_points, intrinsic, extrinsic, alpha=1.0):
    """
    world_points: a numpy array, shape = (N, 3)
    alpha: resize rate

    return: bbox: [row_min, row_max, column_min, column_max], shape = (4,)
    """
    p = np.concatenate((world_points, np.ones((world_points.shape[0], 1))), axis=1)  # (N, 4)
    p = (p @ extrinsic.T)[:, :3]  # (N, 3)
    p = p @ intrinsic.T  # (N, 3)
    uv = p[:, :2] / p[:, 2:]  # (N, 2)
    rows, cols = uv[:, 1], uv[:, 0]

    rows_c, rows_halflen = (rows.min() + rows.max()) / 2, (rows.max() - rows.min()) / 2
    cols_c, cols_halflen = (cols.min() + cols.max()) / 2, (cols.max() - cols.min()) / 2
    rows_min = rows_c - rows_halflen * alpha
    rows_max = rows_c + rows_halflen * alpha
    cols_min = cols_c - cols_halflen * alpha
    cols_max = cols_c + cols_halflen * alpha

    bbox = np.int32([rows_min, rows_max, cols_min, cols_max])
    return bbox


def change_bbox(bbox, H=None, W=None):
    r_min, r_max, c_min, c_max = bbox
    if (not H is None) and (not W is None):
        r_min = max(r_min, 0)
        r_max = min(r_max, H-1)
        c_min = max(c_min, 0)
        c_max = min(c_max, W-1)
    return [int(c_min), int(r_min), int(c_max-c_min+1), int(r_max-r_min+1)]
