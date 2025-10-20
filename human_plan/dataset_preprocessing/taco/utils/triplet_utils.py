import os
import sys
sys.path.append("..")
from dataset_statistics.info import ACTION_MAP


def txt2triplets(file_path):
    triplets = []
    video_idx = 0
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            if "tool_id" in line:
                continue
            values = line.split(" ")
            if (len(values) == 5) or (len(values) == 6):
                video_idx += 1
                triplets.append({"video_idx": int(values[0]), "tool_idx": int(values[1]), "target_idx": int(values[2]), "action_label": ACTION_MAP[values[4]]})
            elif (len(values) == 3) or (len(values) == 4):
                video_idx += 1
                triplets.append({"video_idx": video_idx, "tool_idx": int(values[0]), "target_idx": int(values[1]), "action_label": ACTION_MAP[values[2]]})
            else:
                raise NotImplementedError
    return triplets

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

def get_organized_date_list(path: str):
    organized_date_list = []
    
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 0:
            organized_date_list.append(parts[0][:8])
    
    organized_date_list = list(set(organized_date_list))
    organized_date_list.sort()
    
    return organized_date_list
    
