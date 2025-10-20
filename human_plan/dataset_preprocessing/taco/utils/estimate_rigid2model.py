import os
from os.path import join, isdir, isfile
import sys
sys.path.append("..")
import numpy as np
import cv2
import imageio
import trimesh
import torch
from torch import nn
import time
from tqdm import tqdm
from prepare_2Dmask.utils.pyt3d_wrapper import Pyt3DWrapper
from prepare_2Dmask.utils.colors import FAKE_COLOR_LIST, MARKER_COLOR_LIST
from utils.video_utils import mp42imgs, overlay
from utils.camera_params import txt2intrinsic
from utils.parse_NOKOV import get_obj_names, get_obj_model_paths, parse_trc, parse_xrs, get_NOKOV_data_paths, compute_marker2obj, get_optimize_cfg, get_static_marker_trc_path
from utils.parse_object import load_obj_mesh
from utils.time_align import align_frames, load_aligned_frames
from utils.pose import batch_rodrigues
from knn_cuda import KNN
import open3d as o3d


class Rigid_to_Object(nn.Module):
    def __init__(self, opt_R, opt_t, pre_rotation, init_axangle, init_t):
        super(Rigid_to_Object, self).__init__()
        self.ax = nn.Parameter(torch.from_numpy(init_axangle[0:1]), requires_grad=opt_R[0])
        self.ay = nn.Parameter(torch.from_numpy(init_axangle[1:2]), requires_grad=opt_R[1])
        self.az = nn.Parameter(torch.from_numpy(init_axangle[2:3]), requires_grad=opt_R[2])
        self.x = nn.Parameter(torch.from_numpy(init_t[0:1]), requires_grad=opt_t[0])
        self.y = nn.Parameter(torch.from_numpy(init_t[1:2]), requires_grad=opt_t[1])
        self.z = nn.Parameter(torch.from_numpy(init_t[2:3]), requires_grad=opt_t[2])
        self.pre_rotation = nn.Parameter(batch_rodrigues(torch.from_numpy(pre_rotation).unsqueeze(0))[0], requires_grad=False)  # (3, 3)
    def forward(self):
        return self.pre_rotation, torch.cat((self.ax, self.ay, self.az), axis=0), torch.cat((self.x, self.y, self.z), axis=0)  # (3, 3), (3,), (3,)


def optimize_rigid_to_model(marker_pos, obj_mesh, marker_radius=0.004, cfg=None, save_dir=".", save_suffix="", device="cuda:0"):
    
    os.makedirs(save_dir, exist_ok=True)
    
    obj_mesh_o3d = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(obj_mesh.vertices), triangles=o3d.utility.Vector3iVector(obj_mesh.faces))
    obj_mesh_o3d.compute_vertex_normals()
    obj_vertices = torch.from_numpy(obj_mesh.vertices).to(device)  # (N_obj, 3)
    obj_vertex_normals = torch.from_numpy(np.float32(obj_mesh_o3d.vertex_normals)).to(device)  # (N_obj, 3)
    marker_pos_np = marker_pos.copy()
    marker_pos = torch.from_numpy(marker_pos).to(device)
    
    # save initial markers
    vs, fs = None, None
    for pos in marker_pos_np:
        mesh = trimesh.primitives.Sphere()
        v = mesh.vertices
        f = mesh.faces
        v = v * marker_radius + pos
        if vs is None:
            vs, fs = v, f
        else:
            fs = np.concatenate((fs, f + vs.shape[0]), axis=0)
            vs = np.concatenate((vs, v), axis=0)
    marker_meshes = trimesh.Trimesh(vertices=vs, faces=fs)
    mesh_txt = trimesh.exchange.obj.export_obj(marker_meshes, include_normals=False, include_color=False, include_texture=False, return_texture=False, write_texture=False, resolver=None, digits=8)
    with open(join(save_dir, "ex1_{}.obj".format(save_suffix)), "w") as fp:
        fp.write(mesh_txt)
    
    optim_model = Rigid_to_Object(opt_R=cfg["opt_R"], opt_t=cfg["opt_t"], pre_rotation=cfg["pre_rotation"], init_axangle=cfg["init_axangle"], init_t=cfg["init_t"])
    optim_model.to(device)
    optimizer = torch.optim.Adam(optim_model.parameters(), lr=1e-4)
    optim_model.train()
    
    knn = KNN(k=1, transpose_mode=True)
    
    N_epoch = 2000
    for epoch in range(N_epoch):
        pre_rotation, axangle, t = optim_model()  # (3, 3),  (3,), (3,)
        R = torch.matmul(batch_rodrigues(axangle.unsqueeze(0))[0], pre_rotation)  # (3, 3)
        
        pos = (marker_pos - t) @ R
        
        dist, closest_point = knn(obj_vertices.unsqueeze(0), pos.unsqueeze(0))  # dist不可微!!!
        dist = dist.squeeze(0).squeeze(-1)  # (N_marker,)
        closest_point = closest_point.squeeze(0).squeeze(-1)  # (N_marker,)
        
        contact_flag = ((pos - obj_vertices[closest_point])**2).sum(dim=-1)**(0.5) < 0.01  # < 1cm
        outlier_cnt = contact_flag.shape[0] - contact_flag.sum()
        
        contact_loss = torch.sum((((pos[contact_flag] - obj_vertices[closest_point][contact_flag])**2).sum(dim=-1)**(0.5) - marker_radius)**2) * 1e6  # unit: mm^2
        penetration_loss = torch.sum(torch.clamp(-((pos[contact_flag] - obj_vertices[closest_point][contact_flag]) * obj_vertex_normals[closest_point][contact_flag]).sum(dim=-1) + marker_radius, 0, None)) * 1e3  # unit: mm
        
        loss = contact_loss + penetration_loss
        
        if (epoch % 50 == 0) or (epoch == N_epoch - 1):
            print(epoch, outlier_cnt.item(), loss.item(), contact_loss.item(), penetration_loss.item(), axangle.detach().cpu().numpy(), t.detach().cpu().numpy())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    pre_rotation, axangle, t = optim_model()  # (3, 3),  (3,), (3,)
    T = np.eye(4)
    T[:3, :3] = batch_rodrigues(axangle.unsqueeze(0))[0].detach().cpu().numpy() @ pre_rotation.detach().cpu().numpy()
    T[:3, 3] = t.detach().cpu().numpy()
    
    # save final markers
    marker_pos_np = (marker_pos_np - T[:3, 3]) @ (T[:3, :3] @ np.linalg.inv(pre_rotation.detach().cpu().numpy()))
    vs, fs = None, None
    for pos in marker_pos_np:
        mesh = trimesh.primitives.Sphere()
        v = mesh.vertices
        f = mesh.faces
        v = v * marker_radius + pos
        if vs is None:
            vs, fs = v, f
        else:
            fs = np.concatenate((fs, f + vs.shape[0]), axis=0)
            vs = np.concatenate((vs, v), axis=0)
    marker_meshes = trimesh.Trimesh(vertices=vs, faces=fs)
    mesh_txt = trimesh.exchange.obj.export_obj(marker_meshes, include_normals=False, include_color=False, include_texture=False, return_texture=False, write_texture=False, resolver=None, digits=8)
    with open(join(save_dir, "ex2_{}.obj".format(save_suffix)), "w") as fp:
        fp.write(mesh_txt)
    
    return T


def rearrange(xrs_paths):
    N = len(xrs_paths)
    new_paths = []
    for idx in range(2, N+2):
        new_p = None
        for p in xrs_paths:
            if "_{}-".format(idx) in p:
                new_p = p
                break
        assert not new_p is None
        new_paths.append(new_p)
    return new_paths


def aggregate(marker_pos, xrs_data, static_marker_pos):
    overall_marker_pos = marker_pos.copy()
    for xrs_clip in xrs_data:
        rigid2world = xrs_clip["poses"][0]
        world2rigid = np.linalg.inv(rigid2world)
        pos = static_marker_pos.reshape(1, 3) @ world2rigid[:3, :3].T + world2rigid[:3, 3]  # (1, 3)
        overall_marker_pos = np.concatenate((overall_marker_pos, pos), axis=0)
    
    return overall_marker_pos


if __name__ == "__main__":
    """
    拿到建刚体时录制的小段trc和xrs, 跑一组rigid_to_model出来
    运行方法: 登录liuyun账号, conda activate HHO, 改下面这段信息, 改get_optimize_cfg的"marker_radius"(默认是0.004,即使用直径8mm的marker点), python3 estimate_rigid2model.py
    供参考:
        * 优化时实时输出outlier_cnt和loss的数值, [outlier_cnt <= 1] 且 [loss <= 10] 时可直接使用结果, 否则人工检查结果的质量
        * 物体名称和物体照片对应表: https://docs.qq.com/doc/DSldJR09FeWhEa0FP
    """
    #################################################################################
    obj_name = "198"  # 物体名称
    static_marker_trc_path = "/share/datasets/HOI-mocap/obj_NOKOV_info/calib_marker/calib_20230921_1-Unnamed.trc"  # 场景中一个固定marker点的坐标信息
    device = "cuda:0"
    
    obj_dataset_dir = "/share/datasets/HOI-mocap/object_models_final"  # 扫描的物体模型
    obj_NOKOV_info_dir = "/share/datasets/HOI-mocap/obj_NOKOV_info"  # 物体接触固定marker点的若干组pose
    save_root_dir = "/share/datasets/HOI-mocap/precomputed_rigid2model"  # 结果保存路径
    save_root_dir = "/home/liuyun/HOI-mocap_tmp"
    #################################################################################
    
    obj_model_path = get_obj_model_paths(obj_dataset_dir, [obj_name])[0]
    
    NOKOV_data_dir = join(obj_NOKOV_info_dir, "obj_" + obj_name)
    
    # NOKOV data
    trc_paths, xrs_paths = get_NOKOV_data_paths(NOKOV_data_dir, [obj_name], multi_path=True)
    xrs_paths = rearrange(xrs_paths[0])
    trc_data = parse_trc([trc_paths[0][0]])[0]  # {...}
    xrs_data = parse_xrs(xrs_paths)  # [{"poses": (N_0, 4, 4)}, ..., {"poses": (N_9, 4, 4)}], len = 10
    static_marker_pos = parse_trc([static_marker_trc_path])[0]["markers"][0, 0]  # (3,), unit: m
    
    # object model
    obj_mesh = load_obj_mesh(obj_model_path, unit=0.01)  # unit: cm
    
    # compute rigid_to_model and assistant data
    r2m_save_dir = join(save_root_dir, obj_name)
    os.makedirs(r2m_save_dir, exist_ok=True)
    
    paired_xrs_idx = int(trc_paths[0][0].split("/")[-1].split("-")[0].split("_")[-1]) - 2
    print(trc_data["markers"].shape, xrs_data[paired_xrs_idx]["poses"].shape)
    marker_pos_frame_0, marker_pos, marker_pos_std, marker_pos_max_deviation = compute_marker2obj(trc_data, xrs_data[paired_xrs_idx])
    overall_marker_pos = aggregate(marker_pos_frame_0, xrs_data, static_marker_pos)
    # overall_marker_pos = marker_pos_frame_0
    np.savetxt(join(r2m_save_dir, obj_name + "_marker_pos.txt"), overall_marker_pos)
        
    # finetune rigid_to_object (nearly np.eye(4))
    optimize_cfg = get_optimize_cfg(obj_name)
    rigid_to_model = optimize_rigid_to_model(overall_marker_pos, obj_mesh, marker_radius=optimize_cfg["marker_radius"], cfg=optimize_cfg, save_dir=r2m_save_dir, save_suffix=obj_name, device=device)
    print("rigid_to_model", rigid_to_model)
    np.savetxt(join(r2m_save_dir, obj_name + "_rigid_to_model.txt"), rigid_to_model)
