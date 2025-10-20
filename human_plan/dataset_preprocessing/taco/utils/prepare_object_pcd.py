import os
from os.path import join
import sys
sys.path.append("..")
import numpy as np
from utils.parse_NOKOV import get_obj_model_paths
from utils.parse_object import load_obj_mesh
from utils.mesh import downsample_points_from_mesh


def downsample_all_objects(obj_model_dir, save_dir, N_point=10000, max_idx=300):
    os.makedirs(save_dir, exist_ok=True)
    for idx in range(max_idx):
        obj_name = str(idx).zfill(3)
        print("processing {} ...".format(obj_name))
        obj_model_path = get_obj_model_paths(obj_model_dir, [obj_name])[0]
        if obj_model_path is None:
            print("[error] no obj_model_path for object {} !!!".format(obj_name))
            continue
        obj_mesh = load_obj_mesh(obj_model_path, unit=0.01)
        obj_points = downsample_points_from_mesh(obj_mesh, N_point=N_point)
        np.save(join(save_dir, obj_name + ".npy"), obj_points)


if __name__ == "__main__":
    
    #######################################################################
    N_point = 10000
    obj_model_dir = "/share/datasets/HOI-mocap/object_models_final"
    save_dir = "/share/datasets/HOI-mocap/object_model_points/{}".format(str(N_point))
    #######################################################################
    
    downsample_all_objects(obj_model_dir, save_dir, N_point=N_point)
