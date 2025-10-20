import os
from os.path import join, isdir, isfile
import sys
sys.path.append("..")
import numpy as np
import cv2
import trimesh
import json
import time
from tqdm import tqdm
from utils.video_utils import mp42imgs, overlay, overlay_bboxes
from utils.camera_params import txt2intrinsic
from utils.parse_NOKOV import get_obj_names, get_obj_model_paths, parse_xrs, get_ego_rigid_xrs_path, get_simplied_obj_model_paths
from utils.parse_object import load_obj_mesh
from utils.time_align import load_sequence_names
from utils.hand import mano_params_to_hand_info
from utils.mesh import simplify_mesh, compute_bbox


def load_sequence_names_from_organized_record(path: str, date: str):
    organized_sequence_list = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 0 and parts[0].startswith(date):
            organized_sequence_list.append(parts[0])
            
    organized_sequence_list = list(set(organized_sequence_list))
    organized_sequence_list.sort(key=lambda x:int(x))
    
    return organized_sequence_list


def process_action_recognition_data(date_dir, obj_pose_dir, hand_pose_dir, save_dir, sequence_name, obj_dataset_dir, ego_cam_info, N_rigid=2, overall_overlay_vw=None):
    
    # data integrity check
    ego_rgb_path = join(date_dir, sequence_name, "rgbd", "color.mp4")
    if not isfile(ego_rgb_path):
        print("[error] no egocam data !!!")
        return
    
    nokov_dir = join(date_dir, sequence_name, "nokov")
    if (not isdir(nokov_dir)) or (len(os.listdir(nokov_dir)) != 2 + N_rigid * 2):
        print("[error] incomplete nokov data !!!")
        return
    obj_names = get_obj_names(join(date_dir, sequence_name, "nokov"))  # [工具, 对象]
    assert len(obj_names) == N_rigid

    aligned_frame_path = join(date_dir, sequence_name, "aligned_frames_with_common_timestamps.json")
    if not isfile(aligned_frame_path):
        print("[error] no aligned_frame file !!!")
        return
    aligned_frames = json.load(open(aligned_frame_path, "r"))  # aligned frame data
    N_frame = len(aligned_frames)
    # aligned_frames = aligned_frames[:5]

    obj_pose_files = [join(obj_pose_dir, sequence_name, "objpose", obj_name + ".npy") for obj_name in obj_names]
    for i, f in enumerate(obj_pose_files):
        if not isfile(f):
            print("[error] no object pose for object {} !!!".format(obj_names[i]))
            return
    obj_poses_list = [np.load(f) for f in obj_pose_files]  # object pose data, order = [工具, 对象]
    for i, single_obj_poses in enumerate(obj_poses_list):
        if single_obj_poses.shape[0] != N_frame:
            print("[error] incomplete object pose !!!")
            return
    
    hand_pose_dirs = [join(hand_pose_dir, sequence_name, "mano_wo_contact", "right_hand.pkl"), join(hand_pose_dir, sequence_name, "mano_wo_contact", "left_hand.pkl")]  # hand pose data, order = [右手, 左手]
    
    if (not isfile(hand_pose_dirs[0])) or (not isfile(hand_pose_dirs[1])):  # 优先用整合的文件, 如果没有则用单独的文件
        print("(not recommended) search for separate files ...")
        hand_pose_dirs = [join(hand_pose_dir, sequence_name, "mano_wo_contact", "right_hand"), join(hand_pose_dir, sequence_name, "mano_wo_contact", "left_hand")]  # hand pose data, order = [右手, 左手]
        
        if (not isdir(hand_pose_dirs[0])) or (len(os.listdir(hand_pose_dirs[0])) != N_frame):
            print("[error] incomplete right hand pose !!!")
            return
        if (not isdir(hand_pose_dirs[1])) or (len(os.listdir(hand_pose_dirs[1])) != N_frame):
            print("[error] incomplete left hand pose !!!")
            return
    st = time.time()
    right_hand_vertices, _ = mano_params_to_hand_info(hand_pose_dirs[0], mano_beta=np.zeros(10), side="right", max_cnt=len(aligned_frames))  # right hand data, shape = (N_frame, 778, 3)
    left_hand_vertices, _ = mano_params_to_hand_info(hand_pose_dirs[1], mano_beta=np.zeros(10), side="left", max_cnt=len(aligned_frames))  # left hand data, shape = (N_frame, 778, 3)
    print("read hand_data time = ", time.time() - st)
    if (right_hand_vertices.shape[0] != N_frame) or (left_hand_vertices.shape[0] != N_frame):
        print("[error] incomplete hand pose !!!")
        return
    
    # get video data
    ego_rgb_list = mp42imgs(ego_rgb_path, return_rgb=True, max_cnt=None)
    ego_rigid_xrs_path = get_ego_rigid_xrs_path(nokov_dir)
    ego_rigid_xrs_data = parse_xrs([ego_rigid_xrs_path])[0]  # (N, 4, 4)
    # get object data
    # obj_model_paths = get_obj_model_paths(obj_dataset_dir, obj_names)
    obj_model_paths = get_simplied_obj_model_paths(obj_dataset_dir, obj_names)
    obj_vertices_list = []
    for obj_model_path in obj_model_paths:
        obj_mesh = load_obj_mesh(obj_model_path, unit=0.01)
        if obj_mesh.vertices.shape[0] > 20000:
            obj_mesh = simplify_mesh(obj_mesh)
        obj_vertices_list.append(obj_mesh.vertices)
    
    # compute bboxes and visualize
    os.makedirs(save_dir, exist_ok=True)
    # os.makedirs(join(save_dir, "bboxes"), exist_ok=True)
    # os.makedirs(join(save_dir, "processed_egocentric_video"), exist_ok=True)
    # os.makedirs(join(save_dir, "overlay"), exist_ok=True)

    H, W = ego_rgb_list[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    processed_egovideo_vw = cv2.VideoWriter(join(save_dir, "processed_egocentric_video.mp4"), fourcc, 30, (W, H))
    overlay_vw = cv2.VideoWriter(join(save_dir, "overlay.mp4"), fourcc, 30, (W, H))

    bboxes_list = []
    for idx, paired_frame in tqdm(enumerate(aligned_frames)):

        if (paired_frame["ego_rgb"] is None) or (paired_frame["NOKOV"] is None):
            print("[error] frame {} has incomplete paired_frame !!!".format(idx))
            continue

        ego_rgb = ego_rgb_list[paired_frame["ego_rgb"]]
        intrinsic = ego_cam_info["intrinsic"]
        extrinsic = np.linalg.inv(ego_rigid_xrs_data["poses"][paired_frame["NOKOV"]] @ ego_cam_info["camera2helmet"])
        bbox_list = []  # [右手, 左手, 工具, 对象]
        bbox_list.append(compute_bbox(right_hand_vertices[idx], intrinsic, extrinsic, alpha=1.1))
        bbox_list.append(compute_bbox(left_hand_vertices[idx], intrinsic, extrinsic, alpha=1.1))
        obj2world = obj_poses_list[0][idx]
        bbox_list.append(compute_bbox(obj_vertices_list[0] @ obj2world[:3, :3].T + obj2world[:3, 3], intrinsic, extrinsic, alpha=1.1))
        obj2world = obj_poses_list[1][idx]
        bbox_list.append(compute_bbox(obj_vertices_list[1] @ obj2world[:3, :3].T + obj2world[:3, 3], intrinsic, extrinsic, alpha=1.1))
        # print(bbox_list)
        bboxes_list.append(bbox_list)

        ego_bgr = ego_rgb[:, :, ::-1].astype(np.uint8)
        processed_egovideo_vw.write(ego_bgr)
        overlay_rgb = overlay_bboxes(ego_rgb, bbox_list)
        overlay_bgr = overlay_rgb[:, :, ::-1].astype(np.uint8)
        overlay_vw.write(overlay_bgr)
        if not overall_overlay_vw is None:
            labeled_overlay_bgr = cv2.putText(overlay_bgr, "{} {}".format(sequence_name, str(idx)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            overall_overlay_vw.write(labeled_overlay_bgr)
    
    processed_egovideo_vw.release()
    overlay_vw.release()
    bboxes_np = np.int32(bboxes_list)
    np.save(join(save_dir, "bboxes.npy"), bboxes_np)


if __name__ == "__main__":
    
    ##################################################################
    # date_list = ['20230917', '20230919', '20230923', '20230926', '20230927', '20230928', '20230929', '20231002', '20231005', '20231006', '20231010']
    # date_list = ['20231013', '20231015']
    # date_list = ['20231019', '20231026', '20231027']
    # date_list = ['20231020']
    # date_list = ['20231031', '20231102']
    date_list = ['20231024']
    intrinsics_foldername = "intrinsics"
    camera2helmet_foldername = "camera2helmet"
    
    # obj_dataset_dir = "/share/datasets/HOI-mocap/object_models_final"
    # obj_pose_dataset_dir = "/share/datasets/HOI-mocap/HO_poses"
    # hand_pose_dataset_dir = "/share/hlyang/results/dataset"
    # save_data_dir = "/share/datasets/HOI-mocap/action_recognition_data"
    
    # obj_dataset_dir = "/data2/HOI-mocap/object_models_final"
    obj_dataset_dir = "/data2/HOI-mocap/object_models_final_simplied" # use simplied objs
    obj_pose_dataset_dir = "/data2/HOI-mocap/HO_poses"
    hand_pose_dataset_dir = "/data2/hlyang/results/dataset"
    save_data_dir = "/data2/HOI-mocap/action_recognition_data"
    
    N_rigid = 2
    ##################################################################

    for date in date_list:

        # date_dir = join("/share/datasets/HOI-mocap", date)
        date_dir = join("/data2/HOI-mocap", date)
        
        obj_pose_dir = join(obj_pose_dataset_dir, date)
        hand_pose_dir = join(hand_pose_dataset_dir, date)
        
        # egocentric camera info
        ego_cam_info = {}
        ego_cam_info["intrinsic"], _ = txt2intrinsic(join(date_dir, "camera_params", intrinsics_foldername, "l515.txt"))
        ego_cam_info["camera2helmet"] = np.loadtxt(join(date_dir, "camera_params", camera2helmet_foldername, "l515.txt"))

        # find sequences
        hand_pose_organized_record_path = join(hand_pose_dataset_dir, 'organized_record.txt')
        
        sequence_names = []
        # if isfile(join(date_dir, date + "_valid_video_id.txt")):
        #     print("use prepared sequence names !!!")
        #     sequence_names = load_sequence_names(join(date_dir, date + "_valid_video_id.txt"))
        if isfile(hand_pose_organized_record_path):
            print("use prepared sequence names !!!")
            sequence_names = load_sequence_names_from_organized_record(hand_pose_organized_record_path, date)
        else:
            raise NotImplementedError

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        overall_overlay_vw = cv2.VideoWriter(join(save_data_dir, date + "_overall_overlay.mp4"), fourcc, 30, (1920, 1080))
        
        for sequence_name in sequence_names:
            try:
                seq_idx = int(sequence_name.split("_")[-1])
            except:
                print("[invalid seqence name]", sequence_name)
                continue
            
            print("#################################### [action recognition data] start processing {} ... ####################################".format(sequence_name))
            save_dir = join(save_data_dir, date, sequence_name)
            process_action_recognition_data(date_dir, obj_pose_dir, hand_pose_dir, save_dir, sequence_name, obj_dataset_dir, ego_cam_info=ego_cam_info, N_rigid=N_rigid, overall_overlay_vw=overall_overlay_vw)
            print("#################################### [action recognition data] finish processing {} !!! ####################################".format(sequence_name))

        overall_overlay_vw.release()
