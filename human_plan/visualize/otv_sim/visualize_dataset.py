import argparse
import numpy as np
import cv2
import os
from sklearn.neighbors import NearestNeighbors
import tqdm
import sys
import shutil
from ffprobe import FFProbe

axis_transform = np.linalg.inv(
    np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))


class IndexSearch():
  def __init__(self, time_array):
    self.time_array = time_array
    self.prev = 0
    self.index = 0
    self.len = len(time_array)

  def nearest_neighbor(self, target_time):
    while (target_time > self.time_array[self.index]):
      if self.len - 1 <= self.index:
        return self.index
      self.index += 1
      self.prev = self.time_array[self.index]

    if (abs(self.time_array[self.index] - target_time) > abs(self.time_array[self.index - 1] - target_time)) and (self.index != 0):
      ret_index = self.index - 1
    else:
      ret_index = self.index
    return ret_index


def get_handpose_connectivity():
  # Hand joint information is in https://github.com/microsoft/psi/tree/master/Sources/MixedReality/HoloLensCapture/HoloLensCaptureExporter
  return [
      [0, 1],

      # Thumb
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],

      # Index
      [1, 6],
      [6, 7],
      [7, 8],
      [8, 9],
      [9, 10],

      # Middle
      [1, 11],
      [11, 12],
      [12, 13],
      [13, 14],
      [14, 15],

      # Ring
      [1, 16],
      [16, 17],
      [17, 18],
      [18, 19],
      [19, 20],

      # Pinky
      [1, 21],
      [21, 22],
      [22, 23],
      [23, 24],
      [24, 25]
  ]


def read_intrinsics_txt(img_instrics_path, resized_width, resized_height):
  with open(img_instrics_path) as f:
    data = list(map(float, f.read().split('\t')))
    intrinsics = np.array(data[:9]).reshape(3, 3)
    width = data[-2]
    height = data[-1]

  scale_x = resized_width / width
  scale_y = resized_height / width
  intrinsics[0] = intrinsics[0] * scale_x
  intrinsics[1] = intrinsics[1] * scale_y
  # print(data)
  return intrinsics, width, height


def project_one_hand(hand_points, img, color, img_intrinsics):

    # Put an empty camera pose for image.
  rvec = np.array([[0.0, 0.0, 0.0]])
  tvec = np.array([0.0, 0.0, 0.0])

  points, _ = cv2.projectPoints(
      # hand_points[:3], rvec, tvec, img_intrinsics, np.array([]))
      hand_points, rvec, tvec, img_intrinsics, np.array([]))

  connectivity = get_handpose_connectivity()
  radius = 5
  thickness = 2

  # print("points",points)
  if not (np.isnan(points).any()):
    for limb in connectivity:
      cv2.line(img, (int(points[limb[0]][0][0]), int(points[limb[0]][0][1])),
               (int(points[limb[1]][0][0]), int(points[limb[1]][0][1])), color, thickness)

  return img


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_path', type=str, default='/media/taein/8tb_sdd/datasets/HoloAssist/HoloAssist',
                      help='Directory of untar data')
  parser.add_argument('--processed_data_path', type=str, default='/media/taein/8tb_sdd/datasets/HoloAssist/HoloAssist',
                      help='Directory of untar data')
  parser.add_argument('--video_name', type=str,
                      default='z013-june-16-22-gopro', help="Directory sequence")
  parser.add_argument('--frame_num', type=int, default=None,
                      help='Specific number of frame')
  # parser.add_argument('--eye', action='store_true',
  #                     help='Proejct eye gaze on images')
  # parser.add_argument('--save_eyeproj', action='store_true',
  #                     help='Save eyeproj.txt file.')
  parser.add_argument('--save_video', action='store_true',
                      help='Save hand_project_mpeg file.')
  # parser.add_argument('--eye_dist', type=float, default=0.5,
  #                     help='Eyegaze projection dist is 50cm by default')
  parser.add_argument('--offset', type=int, default=0,
                      help='Temporal offset for image and hand poses (ms), hand pose processing time is approximately 50ms')
  args = parser.parse_args()

  base_path = os.path.join(
      args.processed_data_path,
      args.video_name,
  )

  if not os.path.exists(base_path):
    # Exit if the path does not exist
    sys.exit('{} does not exist'.format(base_path))

  mpeg_img_path = os.path.join(base_path, "frames")
  projected_img_path = os.path.join(base_path, "projected_frames")

  # mpeg_aligned_img_path = os.path.join(base_path, "Video", "images_aligned")
  # hands_path = os.path.join(base_path, 'Hands')
  # img_path = os.path.join(base_path, 'Video')

  # Project into the image
  projected_path = os.path.join(base_path, "projected_frames")
  if not os.path.exists(projected_path):
    # Create a new directory because it does not exist
    os.makedirs(projected_path)

  file_list = os.listdir(mpeg_img_path)
  image_list = [f for f in file_list if f.endswith(".jpg")]
  npy_list = [f for f in file_list if f.endswith(
      ".npy") and f.startswith('label')]

  num_frames = len(image_list)

  # Read cam instrics
  img_instrics_path = os.path.join(
      args.dataset_path, args.video_name,
      'Export_py', 'Video/Intrinsics.txt'
  )
  img_intrinsics, width, height = read_intrinsics_txt(
      img_instrics_path, 454, 256)

  import re
  for image_name in tqdm.tqdm(image_list):
    match = re.search(r'frame_(\d+)\.jpg', image_name)
    if match:
      numeric_part = match.group(1)
    else:
      continue
    data_file = f"label_{numeric_part}.npy"
    data_label = np.load(os.path.join(
        mpeg_img_path, data_file
    ), allow_pickle=True)

    left_hand_trans = data_label[
        "current_left_hand_pose"]["hand_trans_cam_frame"].reshape(26, 3)
    left_hand_trans = left_hand_trans.T

    right_hand_trans = data_label[
        "current_right_hand_pose"]["hand_trans_cam_frame"].reshape(26, 3)
    right_hand_trans = right_hand_trans.T

    img = cv2.imread(
        os.path.join(mpeg_img_path, image_name)
    )

    print("*" * 100)
    # print(left_hand_trans.T)

    img = project_one_hand(left_hand_trans, img, (255, 0, 0), img_intrinsics)
    img = project_one_hand(right_hand_trans, img, (0, 255, 0), img_intrinsics)



    future_idx = 30

    left_hand_trans_future = data_label[
        "future_left_hand_pose"]["hand_trans_cam_frame"].reshape(-1, 26, 3)[future_idx]
    left_hand_trans_future = left_hand_trans_future.T

    right_hand_trans_future = data_label[
        "future_right_hand_pose"]["hand_trans_cam_frame"].reshape(-1, 26, 3)[future_idx]
    right_hand_trans_future = right_hand_trans_future.T

    # print(left_hand_trans_future.T)
    print(left_hand_trans_future.T - left_hand_trans.T)

    img = project_one_hand(left_hand_trans_future, img, (255, 0, 255), img_intrinsics)
    img = project_one_hand(right_hand_trans_future, img, (0, 255, 255), img_intrinsics)


    cv2.imwrite(
        os.path.join(projected_path, image_name), img
    )

    # video_out.write(img)


if __name__ == '__main__':
  main()
