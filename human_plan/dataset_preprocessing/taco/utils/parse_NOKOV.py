import os
from os.path import join, isfile
import numpy as np
import torch
from transforms3d.quaternions import quat2mat, mat2quat


def get_obj_names(sequence_dir):
    obj_names = None
    for fn in os.listdir(sequence_dir):
        names = fn.split("-")[0].split("_")[1:-1]
        if obj_names is None:
            obj_names = names
        else:
            assert len(names) == len(obj_names)
            for i in range(len(names)):
                assert names[i] == obj_names[i]
    return obj_names


def get_obj_model_paths(obj_dataset_dir, obj_names=[]):
    obj_model_paths = []
    for obj_name in obj_names:
        obj_model_path = None
        for fn in os.listdir(obj_dataset_dir):
            if (fn[-9:] == "object" + obj_name) or ((obj_name[0] == '0') and (fn[-8:] == "object" + obj_name[1:])):
                for mesh_name in os.listdir(join(obj_dataset_dir, fn)):
                    if "_cm.obj" in mesh_name:
                        assert obj_model_path is None
                        obj_model_path = join(obj_dataset_dir, fn, mesh_name)
                    if "-cm.obj" in mesh_name:
                        assert obj_model_path is None
                        obj_model_path = join(obj_dataset_dir, fn, mesh_name)
                assert isfile(obj_model_path)
        obj_model_paths.append(obj_model_path)
    return obj_model_paths

def get_simplied_obj_model_paths(obj_dataset_dir, obj_names=[]):
    obj_model_paths = []
    for obj_name in obj_names:
        obj_model_path = join(obj_dataset_dir, f'{obj_name}_cm.obj')
        assert os.path.isfile(obj_model_path)
        obj_model_paths.append(obj_model_path)
    return obj_model_paths


def parse_trc(trc_paths):
    data_list = []
    for trc_path in trc_paths:
        cnt = 0
        data = {
            "timestamps": [],
            "markers": [],
        }
        N_marker = None
        with open(trc_path, "r") as f:
            for line in f:
                cnt += 1
                line = line.strip()
                if cnt == 4:
                    while line.find("\t\t") > -1:
                        line = line.replace("\t\t", "\t")
                    N_marker = len(line.split("\t")) - 3
                    if N_marker == 0:
                        N_marker = 10
                    print("[parse_trc] file {}: N_marker = {}".format(trc_path, N_marker))
                if cnt <= 6:
                    continue
                # assert N_marker >= 3
                values = line.split("\t")
                data["timestamps"].append(int(values[2]))
                markers = np.ones((N_marker, 3)).astype(np.float32) * 10000
                for i in range(3, len(values), 3):
                    x, y, z = values[i : i + 3]
                    if len(x) > 0:
                        markers[(i//3) - 1, 0] = float(x) / 1000
                        markers[(i//3) - 1, 1] = float(y) / 1000
                        markers[(i//3) - 1, 2] = float(z) / 1000
                data["markers"].append(markers)
    
        data["timestamps"] = np.uint64(data["timestamps"])
        data["markers"] = np.float32(data["markers"])
        
        data_list.append(data)
            
    return data_list


def parse_xrs(xrs_paths):
    
    data_list = []
    
    for xrs_path in xrs_paths:
        cnt = 0
        data = {
            "poses": [],
        }
        with open(xrs_path, "r") as f:
            for line in f:
                cnt += 1
                line = line.strip()
                if cnt <= 11:
                    continue
                values = line.split("\t")
                
                assert len(values) == 17
                
                pose = np.eye(4).astype(np.float32)
                pose[:3, 3] = 10000
                t = np.float32([float(values[2]), float(values[3]), float(values[4])]) / 1000
                pose[:3, 3] = t
                q = np.float32([float(values[8]), float(values[5]), float(values[6]), float(values[7])])  # xyzw -> wxyz
                R = quat2mat(q)
                pose[:3, :3] = R
                
                data["poses"].append(pose)
        
        data["poses"] = np.float32(data["poses"])  # (N, 4, 4)
        
        data_list.append(data)
    
    return data_list


def get_NOKOV_data_paths(NOKOV_data_dir, obj_names=[], multi_path=False):
    trc_path_dict = {}
    xrs_path_dict = {}
    for fn in os.listdir(NOKOV_data_dir):
        if fn[-4:] == ".trc":
            obj_name = fn.split(".")[0].split("_")[-1]
            if multi_path:
                if not obj_name in trc_path_dict:
                    trc_path_dict[obj_name] = []
                trc_path_dict[obj_name].append(join(NOKOV_data_dir, fn))
            else:
                trc_path_dict[obj_name] = join(NOKOV_data_dir, fn)
        if fn[-4:] == ".xrs":
            obj_name = fn.split(".")[0].split("_")[-1]
            if multi_path:
                if not obj_name in xrs_path_dict:
                    xrs_path_dict[obj_name] = []
                xrs_path_dict[obj_name].append(join(NOKOV_data_dir, fn))
            else:
                xrs_path_dict[obj_name] = join(NOKOV_data_dir, fn)
    
    trc_paths = [trc_path_dict[x] for x in obj_names]
    xrs_paths = [xrs_path_dict[x] for x in obj_names]
    return trc_paths, xrs_paths


def get_static_marker_trc_path(NOKOV_data_dir):
    trc_path = None
    for fn in os.listdir(NOKOV_data_dir):
        if fn[:6] != "calib_":
            continue
        trc_path = join(NOKOV_data_dir, fn)
        break
    return trc_path


def get_ego_rigid_xrs_path(NOKOV_data_dir):
    xrs_path = None
    for fn in os.listdir(NOKOV_data_dir):
        if (fn[-4:] == ".xrs") and ("helmet" in fn):
            assert xrs_path is None
            xrs_path = join(NOKOV_data_dir, fn)
    return xrs_path


def compute_marker2obj(trc_data, xrs_data):
    N_frame = min(trc_data["markers"].shape[0], xrs_data["poses"].shape[0])
    trc_data["markers"] = trc_data["markers"][:N_frame]
    xrs_data["poses"] = xrs_data["poses"][:N_frame]
    # check data quality
    N_marker = trc_data["markers"].shape[1]
    flag = trc_data["markers"].max(axis=-1).max(axis=-1) < 9999
    integral_markers = trc_data["markers"][flag]  # (N_integral, N_marker, 3)
    integral_objposes = xrs_data["poses"][flag]  # (N_integral, 4, 4)
    print(integral_markers.shape, integral_objposes.shape)
    if integral_markers.shape[0] == 0:
        print("invalid trc data !!!")
        return None, None, None, None
    N_integral = integral_markers.shape[0]
    
    integral_world2obj = np.linalg.inv(integral_objposes)
    integral_world2obj = torch.from_numpy(integral_world2obj)
    integral_markers = torch.from_numpy(integral_markers)
    integral_markers = torch.cat((integral_markers, torch.ones(N_integral, N_marker, 1)), dim=2)
    
    pos_hypothesis = torch.einsum('bij,bjk->bik', integral_markers, integral_world2obj.permute(0, 2, 1))
    pos_hypothesis = pos_hypothesis[:, :, :3].numpy()
    
    mean_pos = pos_hypothesis.mean(axis=0)
    std = np.std(pos_hypothesis - mean_pos, axis=0)
    max_deviation = np.abs(pos_hypothesis - mean_pos).max(axis=0)
    
    print("############ mean marker info ############")
    print("hypothesis 0:", pos_hypothesis[0])
    print("mean_pos:", mean_pos)
    print("std:", std)
    print("max_deviation:", max_deviation)
    print("##########################################")
    
    return pos_hypothesis[0], mean_pos, std, max_deviation


def lerp(T0, T1, alpha=0.5):
    """
    T0 * alpha + T1 * (1 - alpha)
    """
    
    T = np.eye(4)
    T[:3, 3] = T0[:3, 3] * alpha + T1[:3, 3] * (1 - alpha)
    
    q0 = mat2quat(T0[:3, :3])
    q1 = mat2quat(T1[:3, :3])
    if np.sum(q0 * q1) < np.sum(-q0 * q1):
        q0 = - q0
    q = q0 * alpha + q1 * (1 - alpha)
    T[:3, :3] = quat2mat(q)
    
    return T


def denoise_objpose(obj_poses, invalid_threshould=0.10):
    """
    obj_poses: shape = (N_frame, 4, 4), from original xrs data
    """
    N = obj_poses.shape[0]
    
    valid = abs(obj_poses[:, 2, 3]) > invalid_threshould
    if (valid.sum() == 0) or (valid.sum() == N):
        return obj_poses, 0
    
    print("denoising object poses ...")
    
    prev_valid_ids = np.zeros(N).astype(np.int32)
    p = -1
    for i in range(N):
        if valid[i]:
            p = i
        prev_valid_ids[i] = p
    
    next_valid_ids = np.zeros(N).astype(np.int32)
    p = N
    for i in range(N-1, -1, -1):
        if valid[i]:
            p = i
        next_valid_ids[i] = p
    
    max_range = 0
    for i in range(N):
        if valid[i]:
            continue
        L_idx = prev_valid_ids[i]
        R_idx = next_valid_ids[i]
        if L_idx == -1:
            max_range = max(max_range, R_idx - i)
            obj_poses[i] = obj_poses[R_idx]
            continue
        if R_idx == N:
            max_range = max(max_range, i - L_idx)
            obj_poses[i] = obj_poses[L_idx]
            continue
        max_range = max(max_range, R_idx - L_idx - 1)
        obj_poses[i] = lerp(obj_poses[L_idx], obj_poses[R_idx], alpha=(R_idx-i)/(R_idx-L_idx))
    
    return obj_poses, max_range


def get_optimize_cfg(obj_name):
    cfg = {
        "marker_radius": 0.004,
        "opt_R": [True, True, True],
        "opt_t": [True, True, True],
        "pre_rotation": np.float32([0, 0, 0]),
        "init_axangle": np.float32([0, 0, 0]),
        "init_t": np.float32([0, 0, 0.007]),  # NOKOV默认坐标系和实际的在z轴上差了7mm
    }
    
    # case-by-case settings
    # if obj_name == "005":
    #     cfg["marker_radius"] = 0.005
    #     cfg["opt_R"] = [False, False, True]
    #     cfg["pre_rotation"] = np.float32([-np.pi/2, 0, 0])  # 建刚体时的obj_model是y轴朝上的, 现在obj_model都统一成了z轴朝上
    #     cfg["init_t"] = np.float32([0, 0.007, 0])  # 当时模型是y轴朝上的, 所以yz分量互换
    if obj_name == "006":
        cfg["marker_radius"] = 0.005
        cfg["pre_rotation"] = np.float32([-np.pi/2, 0, 0])  # 建刚体时的obj_model是y轴朝上的, 现在obj_model都统一成了z轴朝上
        cfg["init_t"] = np.float32([0.005, 0, 0])  # 当时模型是y轴朝上的, 所以yz分量互换
    # if obj_name == "008":
    #     cfg["marker_radius"] = 0.005
    #     cfg["opt_R"] = [False, False, False]
    #     cfg["opt_t"] = [True, False, True]
    #     cfg["pre_rotation"] = np.float32([-np.pi/2, 0, 0])  # 建刚体时的obj_model是y轴朝上的, 现在obj_model都统一成了z轴朝上
    #     cfg["init_t"] = np.float32([0, 0.007, 0])  # 当时模型是y轴朝上的, 所以yz分量互换
    if obj_name == "038":
        cfg["init_t"] = np.float32([0, 0, 0.080])  # 建刚体时填错了z轴偏移量
    if obj_name == "047":
        cfg["init_axangle"] = np.float32([0, 0, -np.pi/2])  # 建刚体时摆错了朝向
    if obj_name == "171":
        cfg["init_t"] = np.float32([-0.006, 0, 0.007])
    if obj_name == "105":
        cfg["init_t"] = np.float32([0.010, 0, 0.007])
    if obj_name == "103":
        cfg["init_t"] = np.float32([0.030, 0, 0.007])
    if obj_name == "074":
        cfg["init_axangle"] = np.float32([0.03363231, 0.02953327, 0.04008295])
        cfg["init_t"] = np.float32([-0.00208643, -0.00374925, 0.00344347])
    if obj_name == "021":
        cfg["init_t"] = np.float32([0.010, 0, 0.007])
    if obj_name == "195":
        cfg["init_axangle"] = np.float32([0, 3/180*np.pi, 0])
    if obj_name == "218":
        cfg["init_t"] = np.float32([0.010, 0, 0.007])
    
    return cfg


if __name__ == "__main__":
    trc_path = "/share/datasets/HOI-mocap/20230829/NOKOV/obj_010_1/20230829_obj010_1-obj_010.trc"
    xrs_path = "/share/datasets/HOI-mocap/20230829/NOKOV/obj_010_1/20230829_obj010_1-obj_010.xrs"
    N_marker = 4
    
    trc_data = parse_trc(trc_path, N_marker=N_marker)
    xrs_data = parse_xrs(xrs_path, N_rigid=1)
    mean_pos, std, max_deviation = compute_marker2obj(trc_data, xrs_data)
    
    print("mean_pos =", mean_pos)
    print("std =", std)
    print("max deviation =", max_deviation)
