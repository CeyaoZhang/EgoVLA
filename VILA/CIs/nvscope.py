import os
import os.path as osp
import re
import subprocess
import sys

r_on = re.compile(r"""[\t\s]*#[\t\s]*nvcode[\t\s]*:[\t\s]*on[\t\s]*""")
r_off = re.compile(r"""[\t\s]*#[\t\s]*nvcode[\t\s]*:[\t\s]*off[\t\s]*""")

"""
find all files btw #nvcode:on and #nvcode:off
"""


def filter_nvcode(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    new_lines = []
    new_idx = []
    skip_flag = False
    for i, line in enumerate(lines):
        if r_off.findall(line):
            skip_flag = False
            continue

        if skip_flag:
            continue

        if r_on.findall(line):
            skip_flag = True
            continue
        new_lines.append(line)
        new_idx.append(i)

    return lines, new_idx, new_lines


def iterate_py_files(directory):
    py_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith((".py", ".sh")):
                py_files.append(os.path.join(root, file))
    return py_files


def get_authors(file_path):
    cmd = f"git log --format='%aN' '{file_path}' | sort -u"
    output = subprocess.check_output(cmd, shell=True, stderr=None).decode("utf-8")
    authors = output.strip().split("\n")
    return authors


def check_rule(line):
    if "/lustre/fsw" in line or "/home" in line:
        return True

    if "nvr_elm_llm" in line or "llmservice" in line or "cosmos_misc":
        return True

    return False


def check_file_confidential_info(fpath):
    lines, new_idx, new_lines = filter_nvcode(fpath)

    pass_flag = True
    for idx, (_idx, line) in enumerate(zip(new_idx, new_lines)):
        if "/lustre/fsw" in line or "/home" in line:
            authors = get_authors(fpath)
            print(f"{fpath} --- Line: {_idx} --- authors: {authors}")
            print("\t", line.strip())
            pass_flag = False

    return pass_flag


def main(fpath="llava/data/datasets_mixture.py"):
    if osp.isdir(fpath):
        py_files = iterate_py_files(fpath)
        for fpath in py_files:
            res = check_file_confidential_info(fpath)
    elif osp.isfile(fpath):
        res = check_file_confidential_info(fpath)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
# print("".join(res))
