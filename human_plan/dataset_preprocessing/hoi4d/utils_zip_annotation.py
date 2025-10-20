import zipfile
import os

# def read_zip_file(zip_path, extract_to=None):
#     # Check if the file is a valid ZIP file
#     if not zipfile.is_zipfile(zip_path):
#         print(f"{zip_path} is not a valid zip file.")
#         return

#     # Open the zip file in read mode
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         # List all the files in the zip file
#         zip_ref.printdir()

#         # If extract_to is specified, extract the contents
#         if extract_to:
#             # Ensure the output directory exists
#             os.makedirs(extract_to, exist_ok=True)
            
#             # Extract all files to the specified directory
#             zip_ref.extractall(extract_to)
#             print(f"Files extracted to {extract_to}")

# # Example usage:
zip_file_path = '/mnt/data3/data/HOI4D/HOI4D_annotations.zip'
# # output_directory = 'extracted_files'  # Optional: specify where to extract files
# read_zip_file(zip_file_path)

# import zipfile

# def read_zip_file_in_memory(zip_path):
#     # Check if the file is a valid ZIP file
#     if not zipfile.is_zipfile(zip_path):
#         print(f"{zip_path} is not a valid zip file.")
#         return

#     # Open the zip file in read mode
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         # Iterate through each file in the zip
#         for file_info in zip_ref.infolist():
#             print(f"Reading file: {file_info.filename}")
            
#             # Read the file content as bytes
#             with zip_ref.open(file_info.filename) as file:
#                 content = file.read()
#                 # Do something with the content (e.g., decode if it's text)
#                 print(f"File content (first 100 bytes): {content[:100]}")  # Example: print first 100 bytes

# # # Example usage:
# # zip_file_path = 'your_zip_file.zip'
# read_zip_file_in_memory(zip_file_path)

import zipfile
import pickle
import numpy as np

def obtain_pose(
  full_sequence,
):
  
  poses = []
  
  num_pose = len(full_sequence) // 5
  # print(num_pose)
  for i in range(num_pose):
    pose_data = full_sequence[i * 5: (i + 1) * 5]
    # print(pose_data)
    pose_data = [line.split() for line in pose_data[1:]]
    # print(pose_data)
    pose_data = np.array(pose_data, dtype=float)
    # print(pose_data)
    poses.append(pose_data)
  return poses

# Convert the values into a NumPy array of floats
# numpy_array = np.array(values, dtype=float)]
    

def read_specific_file_from_zip(zip_path, file_name):
    # Check if the file is a valid ZIP file
    if not zipfile.is_zipfile(zip_path):
        print(f"{zip_path} is not a valid zip file.")
        return

    # Open the zip file in read mode
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Check if the file exists in the zip archive
        file_name = f'HOI4D_annotations/ZY20210800004/H4/C1/N35/S244/s04/T4/3Dseg/output.log'  # Specify the file you want to read from the zip
        if file_name in zip_ref.namelist():
            print(f"Reading file: {file_name}")
            
            with zip_ref.open(file_name) as file:
                # Load the file content using pickle
                data = file.readlines()
        else:
            print(f"File '{file_name}' not found in the zip archive.")
        # print)
        data = [line.decode('utf-8').strip() for line in data]
        print(data[:10])
        pose_datas = obtain_pose(data)
        for pose in pose_datas[:10]:
          print(pose)
        # print(data)
        # for key in data.keys():
        #   print(key)
        #   print(data[key].shape)
# Example usage:
# zip_file_path = 'your_zip_file.zip'


specific_file_name = 'handpose/refinehandpose_right/ZY20210800002/H2/C6/N10/S42/s05/T1/294.pickle'  # Specify the file you want to read from the zip
read_specific_file_from_zip(zip_file_path, specific_file_name)



# def list_files_in_subdirectory(zip_path, subdirectory):
#     # Ensure the subdirectory path ends with a slash (to match directories)
#     if not subdirectory.endswith('/'):
#         subdirectory += '/'

#     # Check if the file is a valid ZIP file
#     if not zipfile.is_zipfile(zip_path):
#         print(f"{zip_path} is not a valid zip file.")
#         return []

#     # Open the zip file in read mode
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         p = zipfile.Path(zip_ref, subdirectory)
#         # files_in_subdirectory = list(p.iterdir())
#         # List all files in the zip and filter those in the specified subdirectory
#         files_in_subdirectory = [file for file in zip_ref.namelist() if file.startswith(subdirectory) and not file.endswith('/')]

#         return files_in_subdirectory

# # Example usage:
# # zip_file_path = 'your_zip_file.zip'
# import natsort
# import time
# t = time.time()
# # for i in range(10):
# for i in range(1):
#   subdirectory_name = 'handpose/refinehandpose_right/ZY20210800002/H2/C6/N10/S42/s05/T1/'  # Specify the subdirectory you want to list files from
#   files = list_files_in_subdirectory(zip_file_path, subdirectory_name)
#   print("Num of files:", len(files))
#   # print(f"Files in '{subdirectory_name}':")
#   # for file in natsort.natsorted(files):
#   #     print(file)

# c_t = time.time() - t

# print( c_t / 10)
    