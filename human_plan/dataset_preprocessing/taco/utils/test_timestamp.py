import os
from os.path import join


if __name__ == "__main__":
    date = "20231005"
    date_dir = join("/share/datasets/HOI-mocap", date)
    
    fns = []
    for fn in os.listdir(date_dir):
        if date in fn:
            fns.append(fn)
    fns.sort()
    
    last1, last2 = -1, -1
    for fn in fns:
        t_last1 = -1
        with open(join(date_dir, fn, "rgb", "22070938_FrameTimeStamp.txt"), "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                t1 = int(line.split(" ")[-1])
                # if last1 >= t1:
                #     print("error", fn, last1, t1)
                t_last1 = t1
        with open(join(date_dir, fn, "rgb", "22139914_FrameTimeStamp.txt"), "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                t2 = int(line.split(" ")[-1])
                # if last >= t:
                #     print("error", fn, last, t)
                t_last2 = t2
        print(fn, t_last1, t_last2, t_last1-last1, t_last2-last2)
        last1 = t_last1
        last2 = t_last2
