import os
from os.path import join, isfile
import sys
sys.path.append("..")
import numpy as np
import json
from utils.parse_NOKOV import parse_trc, parse_xrs


def load_sequence_names(file_path):
    sequence_names = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            sequence_names.append(line)
    return sequence_names


def load_Luster_timestamps(file_path):
    ts = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            ts.append(int(line.split(" ")[-1]))
    return ts


def txt_to_timestamps(txt_path):
    ts = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            ts.append(int(line))  # ms
    return ts


def trc_to_timestamps(trc_data):
    NOKOV_ts = trc_data[0]["timestamps"]
    day_after_20230829 = (NOKOV_ts.min() - 1693238400000) // 86400000
    NOKOV_ts = list((NOKOV_ts - 1693238400000 - day_after_20230829 * 86400000).astype(np.int64))
    return NOKOV_ts


def txt_to_rgbd_timestamps(ts_path):
    video_ts = {"rgb": [], "depth": []}
    with open(ts_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            ts = [int(float(x)) for x in line.split(" ")]
            video_ts["rgb"].append(ts[0])
            video_ts["depth"].append(ts[1])
    
    assert len(video_ts["rgb"]) > 0
    day_after_20230829 = (video_ts["rgb"][0] - 1693238400000) // 86400000
    video_ts["rgb"] = list(np.int64(video_ts["rgb"]) - 1693238400000 - day_after_20230829 * 86400000)
    video_ts["depth"] = list(np.int64(video_ts["depth"]) - 1693238400000 - day_after_20230829 * 86400000)
    return video_ts


def align_frames(sequence_dir, video_dir, SNs, trc_data, SNs_supp=[], threshould=40):
    assert len(SNs) > 0
    # video timestamps
    video_ts = {}
    for SN in SNs:
        video_ts[SN] = load_Luster_timestamps(join(video_dir, SN + "_FrameTimeStamp.txt"))
        
    NOKOV_ts = trc_to_timestamps(trc_data)  # NOKOV timestamps
    
    # supp video timestamps
    if len(SNs_supp) > 0:
        assert len(SNs) > 0
        for SN in SNs_supp:
            video_ts[SN] = load_Luster_timestamps(join(video_dir, SN + "_FrameTimeStamp.txt"))
        dt_supp2main = video_ts[SNs_supp[0]][0] - video_ts[SNs[0]][0]  # 主电脑和副电脑强行首帧同步
        for SN in SNs_supp:
            for i in range(len(video_ts[SN])):
                video_ts[SN][i] -= dt_supp2main
    
    overall_SNs = SNs + SNs_supp
    
    ps = {}
    for SN in overall_SNs:
        ps[SN] = 0
    p_NOKOV = 0
    
    aligned_frames = []
    
    for idx in range(len(video_ts[overall_SNs[0]])):
        camera1_t = video_ts[overall_SNs[0]][idx]
        for SN in overall_SNs[1:]:
            p = ps[SN]
            while (p + 1 < len(video_ts[SN])) and (abs(video_ts[SN][p] - camera1_t) >= abs(video_ts[SN][p+1] - camera1_t)):
                p += 1
            ps[SN] = p
        while (p_NOKOV + 1 < len(NOKOV_ts)) and (abs(NOKOV_ts[p_NOKOV] - camera1_t) >= abs(NOKOV_ts[p_NOKOV+1] - camera1_t)):
            p_NOKOV += 1
        
        flag = abs(NOKOV_ts[p_NOKOV] - camera1_t) < threshould
        for SN in overall_SNs[1:]:
            flag &= (abs(video_ts[SN][ps[SN]] - camera1_t) < threshould)
        if flag:
            paired_frames = [idx] + [ps[SN] for SN in overall_SNs[1:]] + [p_NOKOV]
            aligned_frames.append(paired_frames)

    with open(join(sequence_dir, "aligned_frames.txt"), "w") as f:
        f.write(",".join(overall_SNs + ["NOKOV"]) + "\n")
        for paired_frames in aligned_frames:
            f.write(",".join([str(x) for x in paired_frames]) + "\n")


def align_frames_ego(sequence_dir, video_dir, trc_data, threshould=40):
    video_ts = txt_to_rgbd_timestamps(join(video_dir, "timestamp.txt"))  # rgb & depth timestamps
    NOKOV_ts = trc_to_timestamps(trc_data)  # NOKOV timestamps
    print(video_ts["rgb"][0], video_ts["depth"][0], NOKOV_ts[0])
    
    p_depth, p_NOKOV = 0, 0
    aligned_frames = []
    
    for idx in range(len(video_ts["rgb"])):
        t_rgb = video_ts["rgb"][idx]
        while (p_depth + 1 < len(video_ts["depth"])) and (abs(video_ts["depth"][p_depth] - t_rgb) >= abs(video_ts["depth"][p_depth+1] - t_rgb)):
            p_depth += 1
        while (p_NOKOV + 1 < len(NOKOV_ts)) and (abs(NOKOV_ts[p_NOKOV] - t_rgb) >= abs(NOKOV_ts[p_NOKOV+1] - t_rgb)):
            p_NOKOV += 1
        # print(t_rgb, p_depth, p_NOKOV)
        
        flag = (abs(NOKOV_ts[p_NOKOV] - t_rgb) < threshould) & (abs(video_ts["depth"][p_depth] - t_rgb) < threshould)
        if flag:
            paired_frames = [idx, p_depth, p_NOKOV]
            aligned_frames.append(paired_frames)

    with open(join(sequence_dir, "aligned_frames_ego.txt"), "w") as f:
        f.write("ego_rgb,ego_depth,NOKOV\n")
        for paired_frames in aligned_frames:
            f.write(",".join([str(x) for x in paired_frames]) + "\n")


def align_frames_with_given_timestamps(sequence_dir, timestamp_path, rgb_video_dir, ego_video_dir, SNs=None, SNs_supp=None, align_ego=False, trc_data=None, xrs_data=None, save=True, threshould=40):
    """
    SNs: 主电脑
    SNs_supp: 副电脑
    """

    # timestamps
    common_ts = txt_to_timestamps(timestamp_path)
    video_ts = {}
    
    assert not SNs is None
    
    print(SNs)
    print(SNs_supp)

    for SN in SNs:
        video_ts[SN] = load_Luster_timestamps(join(rgb_video_dir, SN + "_FrameTimeStamp.txt"))
    if not SNs_supp is None:
        for SN in SNs_supp:
            timestamps = load_Luster_timestamps(join(rgb_video_dir, SN + "_FrameTimeStamp.txt"))
            if len(timestamps) == len(video_ts[SNs[0]]):
                delta_t = video_ts[SNs[0]][0] - timestamps[0]
                timestamps_aligned = [t + delta_t for t in timestamps]
                video_ts[SN] = timestamps_aligned
    for k in video_ts:
        print(k, len(video_ts[k]))
    
    if not isfile(join(ego_video_dir, "timestamp.txt")):
        align_ego = False
    if align_ego:
        ego_ts = txt_to_rgbd_timestamps(join(ego_video_dir, "timestamp.txt"))  # rgb & depth timestamps
    NOKOV_ts = trc_to_timestamps(trc_data)  # NOKOV timestamps
    
    # if not xrs_data is None:  # 滤掉没捕捉到pose的帧
    #     assert len(NOKOV_ts) == xrs_data["poses"][0].shape[0]
    #     NOKOV_ts_ori = NOKOV_ts.copy()

    SNs_overall = SNs.copy()
    if not SNs_supp is None:
        SNs_overall += SNs_supp
    
    ps = {}
    Ns = {}
    for SN in SNs_overall:
        ps[SN] = 0
        Ns[SN] = len(video_ts[SN])
    if align_ego:
        ps["ego_rgb"], ps["ego_depth"] = 0, 0
        Ns["ego_rgb"], Ns["ego_depth"] = len(ego_ts["rgb"]), len(ego_ts["depth"])
    ps["NOKOV"] = 0
    Ns["NOKOV"] = len(NOKOV_ts)
    
    aligned_frames = []
    for t in common_ts:
        for SN in SNs_overall:
            p = ps[SN]
            while (p + 1 < Ns[SN]) and (abs(video_ts[SN][p] - t) >= abs(video_ts[SN][p+1] - t)):
                p += 1
            ps[SN] = p
        if align_ego:
            for (signal_name, ts_name) in zip(["rgb", "depth"], ["ego_rgb", "ego_depth"]):
                p = ps[ts_name]
                while (p + 1 < Ns[ts_name]) and (abs(ego_ts[signal_name][p] - t) >= abs(ego_ts[signal_name][p+1] - t)):
                    p += 1
                ps[ts_name] = p
        p = ps["NOKOV"]
        while (p + 1 < Ns["NOKOV"]) and (abs(NOKOV_ts[p] - t) >= abs(NOKOV_ts[p+1] - t)):
            p += 1
        ps["NOKOV"] = p
        
        paired_frames = {}
        for SN in SNs_overall:
            paired_frames[SN] = ps[SN] if abs(video_ts[SN][ps[SN]] - t) < threshould else None
        if align_ego:
            for (signal_name, ts_name) in zip(["rgb", "depth"], ["ego_rgb", "ego_depth"]):
                paired_frames[ts_name] = ps[ts_name] if abs(ego_ts[signal_name][ps[ts_name]] - t) < threshould else None
        paired_frames["NOKOV"] = ps["NOKOV"] if abs(NOKOV_ts[ps["NOKOV"]] - t) < threshould else None
        aligned_frames.append(paired_frames)
    
    print("len(aligned_frames) =", len(aligned_frames))
    N_None = 0
    for pf in aligned_frames:
        for key in pf:
            if pf[key] is None:
                N_None += 1
    print("Number of \"None\" in aligned_frames =", N_None)
    if save:
        os.makedirs(sequence_dir, exist_ok=True)
        json.dump(aligned_frames, open(join(sequence_dir, "aligned_frames_with_common_timestamps.json"), "w"))
    
    return aligned_frames


def load_aligned_frames(file_path):
    aligned_frames = []
    signals = None
    
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == 0:
                continue
            
            if signals is None:  # the first line
                signals = line.split(",")
                continue
            
            paired_frames = {}
            ids = line.split(",")
            assert len(signals) == len(ids)
            for signal, idx in zip(signals, ids):
                paired_frames[signal] = int(idx)
            
            aligned_frames.append(paired_frames)

    return aligned_frames
