#!/usr/bin/env python3
"""
HDF5到Rerun RRD转换脚本
将机器人操作的HDF5数据转换为可视化的RRD格式
支持图像、机器人轨迹、手部姿态等数据的可视化
"""

import h5py
import numpy as np
import argparse
import rerun as rr
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import cv2


def quaternion_to_rotation_matrix(quat_xyzw):
    """将四元数(xyzw格式)转换为旋转矩阵"""
    rot = R.from_quat(quat_xyzw)
    return rot.as_matrix()


def pose_to_transform3d(pose_7d):
    """
    将7D姿态(x,y,z,qw,qx,qy,qz)转换为4x4变换矩阵
    注意：Isaac Lab使用WXYZ格式的四元数
    """
    translation = pose_7d[:3]
    quat_wxyz = pose_7d[3:]
    # 转换为XYZW格式
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    
    rot_matrix = quaternion_to_rotation_matrix(quat_xyzw)
    
    transform = np.eye(4)
    transform[:3, :3] = rot_matrix
    transform[:3, 3] = translation
    
    return transform


def log_transform3d(entity_path, transform_matrix, static=False):
    """记录3D变换到Rerun"""
    # Rerun需要旋转矩阵和平移向量
    translation = transform_matrix[:3, 3]
    rotation = transform_matrix[:3, :3]
    
    # 使用Transform3D记录
    rr.log(
        entity_path,
        rr.Transform3D(
            translation=translation,
            mat3x3=rotation,
        ),
        static=static
    )


def log_camera_intrinsics(entity_path, intrinsics, width, height):
    """记录相机内参"""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    rr.log(
        entity_path,
        rr.Pinhole(
            resolution=[width, height],
            focal_length=[fx, fy],
            principal_point=[cx, cy],
        ),
        static=True
    )


def log_hand_skeleton(entity_path, keypoints_3d, color=None):
    """记录手部骨骼关键点"""
    # MANO手部关键点连接关系
    # 21个关键点: 0=腕关节, 1-4=拇指, 5-8=食指, 9-12=中指, 13-16=无名指, 17-20=小指
    connections = [
        # 腕关节到各个手指基部
        [0, 1], [0, 5], [0, 9], [0, 13], [0, 17],
        # 拇指
        [1, 2], [2, 3], [3, 4],
        # 食指
        [5, 6], [6, 7], [7, 8],
        # 中指
        [9, 10], [10, 11], [11, 12],
        # 无名指
        [13, 14], [14, 15], [15, 16],
        # 小指
        [17, 18], [18, 19], [19, 20],
    ]
    
    # 记录关键点
    if color is None:
        color = [255, 255, 255]
    
    rr.log(
        entity_path + "/points",
        rr.Points3D(
            positions=keypoints_3d,
            colors=[color] * len(keypoints_3d),
            radii=0.01
        )
    )
    
    # 记录骨骼连接
    if len(keypoints_3d) >= 21:  # 确保是MANO手部关键点
        lines = []
        for connection in connections:
            if connection[0] < len(keypoints_3d) and connection[1] < len(keypoints_3d):
                lines.append([keypoints_3d[connection[0]], keypoints_3d[connection[1]]])
        
        if lines:
            rr.log(
                entity_path + "/skeleton",
                rr.LineStrips3D(
                    strips=lines,
                    colors=[color] * len(lines)
                )
            )

def project_3d_to_2d(points_3d, camera_intrinsics, camera_extrinsics):
    """
    将3D点投影到2D图像坐标
    
    参数:
        points_3d: Nx3的3D点数组（世界坐标系）或列表
        camera_intrinsics: 3x3相机内参矩阵
        camera_extrinsics: 4x4相机外参矩阵（世界到相机的变换）
    
    返回:
        Nx2的2D图像坐标数组, N维深度数组
    """
    if len(points_3d) == 0:
        return np.array([]), np.array([])
    
    # 确保输入是numpy数组
    points_3d = np.array(points_3d, dtype=np.float64)
    if points_3d.ndim == 1:
        points_3d = points_3d.reshape(1, -1)
    
    # 按照 otv_isaaclab/utils.py 中 to_cam_frame 的方式处理
    # points = CAM_AXIS_TRANSFORM @ np.linalg.inv(main_cam_transformation) @ homogeneous_coord(points)
    
    # CAM_AXIS_TRANSFORM 定义（来自 human_plan/utils/transformation.py）
    CAM_AXIS_TRANSFORM = np.array([
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    
    # 转换为齐次坐标 (N, 4)
    points_3d_homo = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    
    # 世界坐标转相机坐标：CAM_AXIS_TRANSFORM @ inv(camera_extrinsics) @ points
    camera_inv = np.linalg.inv(camera_extrinsics)
    # points_3d_homo 是 (N, 4)，需要转置为 (4, N) 进行矩阵乘法
    points_cam_homo = CAM_AXIS_TRANSFORM @ camera_inv @ points_3d_homo.T  # (4, N)
    points_cam = points_cam_homo[:3, :].T  # (N, 3)
    
    # 调试：打印变换后的坐标
    if len(points_cam) > 0:
        print(f"\n=== 坐标变换调试 ===")
        print(f"世界坐标（前2个点）:\n{points_3d[:min(2, len(points_3d))]}")
        print(f"相机坐标（前2个点）:\n{points_cam[:min(2, len(points_cam))]}")
        print(f"深度值（Z）: {points_cam[:min(2, len(points_cam)), 2]}")
    
    # 手动实现投影变换（不使用cv2.projectPoints）
    # 相机坐标系下的3D点投影到2D图像平面
    # 公式: [u, v, 1]^T = K * [X, Y, Z]^T / Z
    # 其中 K 是相机内参矩阵
    
    # 提取相机坐标系下的深度（Z值）
    depths = points_cam[:, 2].copy()
    
    # 过滤掉深度为0或负值的点（避免除零错误）
    valid_mask = depths > 0
    
    # 初始化2D点数组
    points_2d = np.zeros((len(points_cam), 2), dtype=np.float64)
    
    if np.any(valid_mask):
        # 提取有效的3D点
        valid_points = points_cam[valid_mask]
        
        # 投影到图像平面: [u, v, w]^T = K @ [X, Y, Z]^T
        # 转换为齐次坐标并与内参矩阵相乘
        projected = (camera_intrinsics @ valid_points.T).T  # shape: (N, 3)
        
        # 归一化得到像素坐标: u = projected[0]/projected[2], v = projected[1]/projected[2]
        points_2d[valid_mask, 0] = projected[:, 0] / projected[:, 2]
        points_2d[valid_mask, 1] = projected[:, 1] / projected[:, 2]
    
    return points_2d, depths


def draw_trajectory_on_image(image, trajectory_2d, depths, color, thickness=2):
    """
    在图像上绘制轨迹
    
    参数:
        image: 输入图像
        trajectory_2d: Nx2的2D轨迹点
        depths: N维深度数组（用于过滤相机后方的点）
        color: BGR格式的颜色元组
        thickness: 线条粗细
    
    返回:
        绘制了轨迹的图像
    """
    if len(trajectory_2d) < 2:
        return image
    
    img_with_traj = image.copy()
    h, w = img_with_traj.shape[:2]
    
    # 过滤掉在相机后方的点和超出图像范围的点
    valid_points = []
    for i, (pt, depth) in enumerate(zip(trajectory_2d, depths)):
        if depth > 0 and 0 <= pt[0] < w and 0 <= pt[1] < h:
            valid_points.append((i, pt))
    
    # 调试：打印有效点数量
    if len(trajectory_2d) > 0 and len(valid_points) == 0:
        print(f"警告：{len(trajectory_2d)} 个轨迹点都不在图像范围内")
        print(f"图像尺寸: {w}x{h}")
        if len(trajectory_2d) > 0:
            print(f"第一个投影点: {trajectory_2d[0]}, 深度: {depths[0]}")
    
    # 绘制轨迹线段
    for i in range(len(valid_points) - 1):
        idx1, pt1 = valid_points[i]
        idx2, pt2 = valid_points[i + 1]
        
        # 只连接连续的点
        if idx2 - idx1 == 1:
            pt1_int = (int(pt1[0]), int(pt1[1]))
            pt2_int = (int(pt2[0]), int(pt2[1]))
            cv2.line(img_with_traj, pt1_int, pt2_int, color, thickness, cv2.LINE_AA)
    
    # 绘制轨迹点
    for _, pt in valid_points:
        pt_int = (int(pt[0]), int(pt[1]))
        cv2.circle(img_with_traj, pt_int, 3, color, -1, cv2.LINE_AA)
    
    return img_with_traj


def convert_hdf5_to_rrd(
    hdf5_path,
    output_rrd_path=None,
    frame_skip=1,
    max_frames=None,
    clip_start=0,
    clip_end=None,
    show_hands=True,
    show_ee_pose=True,
    show_trajectory=True,
):
    """
    将HDF5文件转换为RRD格式
    
    参数:
        hdf5_path: 输入HDF5文件路径
        output_rrd_path: 输出RRD文件路径（如果为None则使用内存记录+显示）
        frame_skip: 帧跳过间隔
        max_frames: 最大帧数
        clip_start: 起始帧
        clip_end: 结束帧
        show_hands: 是否显示手部姿态
        show_ee_pose: 是否显示末端执行器姿态
        show_trajectory: 是否显示轨迹
    """
    
    print(f"正在读取HDF5文件: {hdf5_path}")
    
    # 初始化Rerun
    recording_name = Path(hdf5_path).stem
    rr.init(recording_name, spawn=True)
    
    with h5py.File(hdf5_path, 'r') as f:
        # 读取数据
        obs_group = f['observations']
        
        # 获取图像数据
        images = None
        if 'images' in obs_group:
            image_group = obs_group['images']
            if 'main' in image_group:
                images = image_group['main'][clip_start:clip_end:frame_skip]
                print(f"图像形状: {images.shape}")
        
        # 获取姿态数据
        left_ee_pose = None
        right_ee_pose = None
        
        if 'left_ee_pose' in obs_group:
            left_ee_pose = obs_group['left_ee_pose'][clip_start:clip_end:frame_skip]
            right_ee_pose = obs_group['right_ee_pose'][clip_start:clip_end:frame_skip]
        elif 'left_target_ee_pose' in obs_group:
            left_ee_pose = obs_group['left_target_ee_pose'][clip_start:clip_end:frame_skip]
            right_ee_pose = obs_group['right_target_ee_pose'][clip_start:clip_end:frame_skip]
        
        # 获取手指尖端位置
        left_finger_tip = None
        right_finger_tip = None
        if 'left_finger_tip_pos' in obs_group:
            left_finger_tip = obs_group['left_finger_tip_pos'][clip_start:clip_end:frame_skip]
            right_finger_tip = obs_group['right_finger_tip_pos'][clip_start:clip_end:frame_skip]
        
        # 获取动作数据
        actions = None
        if 'action' in f:
            actions = f['action'][clip_start:clip_end:frame_skip]
        
        # 获取qpos数据
        qpos = None
        if 'qpos' in obs_group:
            qpos = obs_group['qpos'][clip_start:clip_end:frame_skip]
        
        # 标准相机内参（1280x720）
        standard_intrinsics = np.array([
            [488.6662,   0.0000, 640.0000],
            [  0.0000, 488.6662, 360.0000],
            [  0.0000,   0.0000,   1.0000]
        ])
        standard_width, standard_height = 1280, 720
        
        # 根据实际图像尺寸调整内参
        if images is not None:
            img_height, img_width = images.shape[1], images.shape[2]
            print(f"图像尺寸: {img_width}x{img_height}")
            
            # 计算缩放比例
            scale_x = img_width / standard_width
            scale_y = img_height / standard_height
            
            # 调整内参
            camera_intrinsics = np.array([
                [standard_intrinsics[0, 0] * scale_x,   0.0000, img_width / 2.0],
                [  0.0000, standard_intrinsics[1, 1] * scale_y, img_height / 2.0],
                [  0.0000,   0.0000,   1.0000]
            ])
            print(f"缩放比例: x={scale_x:.3f}, y={scale_y:.3f}")
            print(f"调整后的相机内参:\n{camera_intrinsics}")
        else:
            camera_intrinsics = standard_intrinsics
        
        # 相机外参（主相机）
        main_cam_trans = np.array([0.09, 0.0, 1.7])
        main_cam_quat_wxyz = (0.66446, 0.24184, -0.24184, -0.664464)
        main_cam_quat_xyzw = (0.24184, -0.24184, -0.664464, 0.66446)
        main_cam_rotmat = R.from_quat(main_cam_quat_xyzw).as_matrix()
        
        # Isaac Lab相机帧变换
        ISAAC_LAB_CAMERA_FRAME_CHANGE = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        
        main_cam_transformation = np.eye(4)
        main_cam_transformation[:3, :3] = ISAAC_LAB_CAMERA_FRAME_CHANGE @ main_cam_rotmat
        main_cam_transformation[:3, 3] = main_cam_trans
        
        # 记录世界坐标系
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        
        # 记录相机
        rr.log("world/camera", rr.Transform3D(
            translation=main_cam_trans,
            mat3x3=main_cam_transformation[:3, :3]
        ), static=True)
        
        # 记录相机内参
        if images is not None:
            height, width = images.shape[1], images.shape[2]
            log_camera_intrinsics("world/camera/image", camera_intrinsics, width, height)
        
        # 确定帧数
        num_frames = len(images) if images is not None else len(left_ee_pose) if left_ee_pose is not None else 0
        if max_frames:
            num_frames = min(num_frames, max_frames)
        
        print(f"总帧数: {num_frames}")
        
        # 用于存储轨迹
        left_ee_trajectory = []
        right_ee_trajectory = []
        
        # 逐帧记录数据
        for frame_idx in tqdm(range(num_frames), desc="转换帧"):
            # 设置时间戳
            rr.set_time("frame", sequence=clip_start + frame_idx * frame_skip)
            
            # 先更新末端执行器位置和轨迹数据
            # 记录左手末端执行器
            if left_ee_pose is not None and show_ee_pose:
                left_transform = pose_to_transform3d(left_ee_pose[frame_idx])
                log_transform3d("world/robot/left_ee", left_transform)
                
                # 添加到轨迹
                left_ee_trajectory.append(left_transform[:3, 3])
                
                # 记录手指尖端
                if left_finger_tip is not None and show_hands:
                    finger_positions = left_finger_tip[frame_idx]
                    if len(finger_positions.shape) == 1:
                        finger_positions = finger_positions.reshape(-1, 3)
                    
                    rr.log(
                        "world/robot/left_fingers",
                        rr.Points3D(
                            positions=finger_positions,
                            colors=[[0, 0, 255]] * len(finger_positions),
                            radii=0.02
                        )
                    )
            
            # 记录右手末端执行器
            if right_ee_pose is not None and show_ee_pose:
                right_transform = pose_to_transform3d(right_ee_pose[frame_idx])
                log_transform3d("world/robot/right_ee", right_transform)
                
                # 添加到轨迹
                right_ee_trajectory.append(right_transform[:3, 3])
                
                # 记录手指尖端
                if right_finger_tip is not None and show_hands:
                    finger_positions = right_finger_tip[frame_idx]
                    if len(finger_positions.shape) == 1:
                        finger_positions = finger_positions.reshape(-1, 3)
                    
                    rr.log(
                        "world/robot/right_fingers",
                        rr.Points3D(
                            positions=finger_positions,
                            colors=[[0, 255, 0]] * len(finger_positions),
                            radii=0.02
                        )
                    )
            
            # 然后记录图像（此时轨迹数据已更新，包含当前帧）
            if images is not None:
                image = images[frame_idx]
                
                # 记录图像到Rerun
                if len(image.shape) == 3 and image.shape[2] == 3:
                    rr.log("world/camera/image", rr.Image(image))
                
                # 使用Rerun的2D轨迹显示（不修改原始图像）
                if show_trajectory:
                    # 投影并记录左手轨迹
                    if len(left_ee_trajectory) > 1:
                        left_traj_2d, left_depths = project_3d_to_2d(
                            left_ee_trajectory, 
                            camera_intrinsics, 
                            main_cam_transformation
                        )
                        
                        # 在第一次成功投影时打印调试信息
                        if frame_idx == 1:
                            print(f"左手轨迹点数: {len(left_ee_trajectory)}")
                            print(f"左手投影点数: {len(left_traj_2d)}")
                            if len(left_traj_2d) > 0:
                                print(f"左手投影示例点: {left_traj_2d[0]}, 深度: {left_depths[0]}")
                        
                        # 过滤有效点（在相机前方且在图像范围内）
                        h, w = image.shape[:2]
                        valid_left_points = []
                        for pt, depth in zip(left_traj_2d, left_depths):
                            if depth > 0 and 0 <= pt[0] < w and 0 <= pt[1] < h:
                                valid_left_points.append(pt)
                        
                        # 使用Rerun的LineStrips2D在图像空间中显示轨迹
                        if len(valid_left_points) > 1:
                            rr.log(
                                "world/camera/image/left_trajectory",
                                rr.LineStrips2D(
                                    strips=[np.array(valid_left_points)],
                                    colors=[[0, 0, 255]],  # 蓝色
                                    radii=2.0
                                )
                            )
                    
                    # 投影并记录右手轨迹
                    if len(right_ee_trajectory) > 1:
                        right_traj_2d, right_depths = project_3d_to_2d(
                            right_ee_trajectory, 
                            camera_intrinsics, 
                            main_cam_transformation
                        )
                        
                        # 在第一次成功投影时打印调试信息
                        if frame_idx == 1:
                            print(f"右手轨迹点数: {len(right_ee_trajectory)}")
                            print(f"右手投影点数: {len(right_traj_2d)}")
                            if len(right_traj_2d) > 0:
                                print(f"右手投影示例点: {right_traj_2d[0]}, 深度: {right_depths[0]}")
                        
                        # 过滤有效点
                        valid_right_points = []
                        for pt, depth in zip(right_traj_2d, right_depths):
                            if depth > 0 and 0 <= pt[0] < w and 0 <= pt[1] < h:
                                valid_right_points.append(pt)
                        
                        # 使用Rerun的LineStrips2D在图像空间中显示轨迹
                        if len(valid_right_points) > 1:
                            rr.log(
                                "world/camera/image/right_trajectory",
                                rr.LineStrips2D(
                                    strips=[np.array(valid_right_points)],
                                    colors=[[0, 255, 0]],  # 绿色
                                    radii=2.0
                                )
                            )
            
            # 记录轨迹
            if show_trajectory and len(left_ee_trajectory) > 1:
                rr.log(
                    "world/robot/left_trajectory",
                    rr.LineStrips3D(
                        strips=[np.array(left_ee_trajectory)],
                        colors=[[0, 0, 255]]
                    )
                )
            
            if show_trajectory and len(right_ee_trajectory) > 1:
                rr.log(
                    "world/robot/right_trajectory",
                    rr.LineStrips3D(
                        strips=[np.array(right_ee_trajectory)],
                        colors=[[0, 255, 0]]
                    )
                )
            
            # 记录qpos（作为标量）
            if qpos is not None:
                rr.log("robot/joints", rr.Scalars(qpos[frame_idx]))
                # for i, q in enumerate(qpos[frame_idx]):
                #     rr.log(f"robot/joint_{i}", rr.Scalars(q))
            
            # 记录动作
            if actions is not None:
                rr.log("action/dims", rr.Scalars(actions[frame_idx]))
                # for i, a in enumerate(actions[frame_idx]):
                #     rr.log(f"action/dim_{i}", rr.Scalars(a))
    
    # 数据记录完成后，如果指定了输出路径则保存到文件
    if output_rrd_path:
        rr.save(output_rrd_path)
        print(f"RRD文件已保存到: {output_rrd_path}")
    
    print("转换完成!")
    print("Rerun查看器已启动")


def batch_convert_directory(
    input_dir,
    output_dir,
    pattern="*.hdf5",
    **kwargs
):
    """
    批量转换目录中的HDF5文件
    
    参数:
        input_dir: 输入目录
        output_dir: 输出目录
        pattern: 文件匹配模式
        **kwargs: 传递给convert_hdf5_to_rrd的其他参数
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    hdf5_files = list(input_path.rglob(pattern))
    print(f"找到 {len(hdf5_files)} 个HDF5文件")
    
    for hdf5_file in tqdm(hdf5_files, desc="批量转换"):
        # 保持相对路径结构
        relative_path = hdf5_file.relative_to(input_path)
        output_file = output_path / relative_path.with_suffix('.rrd')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            convert_hdf5_to_rrd(
                str(hdf5_file),
                str(output_file),
                **kwargs
            )
        except Exception as e:
            print(f"转换失败 {hdf5_file}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='将HDF5机器人数据转换为Rerun RRD格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换单个文件并在查看器中显示
  python hdf5_to_rrd.py episode_0.hdf5
  
  # 转换并保存为RRD文件
  python hdf5_to_rrd.py episode_0.hdf5 -o output.rrd
  
  # 跳帧转换以加快速度
  python hdf5_to_rrd.py episode_0.hdf5 --frame-skip 5
  
  # 只转换前100帧
  python hdf5_to_rrd.py episode_0.hdf5 --max-frames 100
  
  # 批量转换目录
  python hdf5_to_rrd.py --batch-dir /path/to/hdf5/dir --output-dir /path/to/rrd/dir
  
  # 裁剪起始和结束帧
  python hdf5_to_rrd.py episode_0.hdf5 --clip-start 20 --clip-end 200
        """
    )
    
    parser.add_argument('input', nargs='?', type=str, 
                        help='输入HDF5文件路径')
    parser.add_argument('-o', '--output', type=str,
                        help='输出RRD文件路径（不指定则在查看器中显示）')
    parser.add_argument('-s', '--save-auto', action='store_true',
                    help='自动保存为同名RRD文件')
    parser.add_argument('--frame-skip', type=int, default=1,
                        help='帧跳过间隔（默认: 1）')
    parser.add_argument('--max-frames', type=int,
                        help='最大帧数')
    parser.add_argument('--clip-start', type=int, default=0,
                        help='起始帧索引（默认: 0）')
    parser.add_argument('--clip-end', type=int,
                        help='结束帧索引')
    parser.add_argument('--no-hands', action='store_true',
                        help='不显示手部姿态')
    parser.add_argument('--no-ee-pose', action='store_true',
                        help='不显示末端执行器姿态')
    parser.add_argument('--no-trajectory', action='store_true',
                        help='不显示轨迹')
    
    # 批量处理选项
    parser.add_argument('--batch-dir', type=str,
                        help='批量处理目录中的所有HDF5文件')
    parser.add_argument('--output-dir', type=str,
                        help='批量处理的输出目录')
    parser.add_argument('--pattern', type=str, default='*.hdf5',
                        help='批量处理的文件匹配模式（默认: *.hdf5）')
    
    args = parser.parse_args()
    
    if args.batch_dir:
        # 批量处理模式
        if not args.output_dir:
            print("错误: 批量处理需要指定 --output-dir")
            return
        
        batch_convert_directory(
            args.batch_dir,
            args.output_dir,
            pattern=args.pattern,
            frame_skip=args.frame_skip,
            max_frames=args.max_frames,
            clip_start=args.clip_start,
            clip_end=args.clip_end,
            show_hands=not args.no_hands,
            show_ee_pose=not args.no_ee_pose,
            show_trajectory=not args.no_trajectory,
        )
    else:
        # 单文件处理模式
        if not args.input:
            print("错误: 需要指定输入文件或使用 --batch-dir")
            parser.print_help()
            return
        
        if not Path(args.input).exists():
            print(f"错误: 文件不存在: {args.input}")
            return
        
        # 处理自动保存选项
        output_path = args.output
        if args.save_auto and not output_path:
            output_path = str(Path(args.input).with_suffix('.rrd'))
        
        convert_hdf5_to_rrd(
            args.input,
            output_path,
            frame_skip=args.frame_skip,
            max_frames=args.max_frames,
            clip_start=args.clip_start,
            clip_end=args.clip_end,
            show_hands=not args.no_hands,
            show_ee_pose=not args.no_ee_pose,
            show_trajectory=not args.no_trajectory,
        )


if __name__ == "__main__":
    main()

    '''
    # 1. 安装rerun (如果还没安装)
    pip install rerun-sdk

    # 2. 转换单个文件并在查看器中显示
    python hdf5_to_rrd.py /path/to/episode_0.hdf5

    # 3. 转换并保存为RRD文件
    python hdf5_to_rrd.py /path/to/episode_0.hdf5 -o output.rrd

    # 4. 跳帧转换（例如每5帧取1帧，加快处理速度）
    python hdf5_to_rrd.py episode_0.hdf5 --frame-skip 5

    # 5. 裁剪数据（从第20帧到第200帧）
    python hdf5_to_rrd.py episode_0.hdf5 --clip-start 20 --clip-end 200

    # 6. 批量转换整个目录
    python hdf5_to_rrd.py --batch-dir /data/episodes/ --output-dir /data/rrd_files/

    # 7. 批量转换特定任务
    python hdf5_to_rrd.py --batch-dir /data/Pour-Balls/ --output-dir /output/ --pattern "episode_*.hdf5"
    '''