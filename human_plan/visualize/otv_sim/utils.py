import argparse
import numpy as np
import cv2
import os
import tqdm
import sys
import shutil


def get_handpose_connectivity():
  # Hand joint information is in https://github.com/microsoft/psi/tree/master/Sources/MixedReality/HoloLensCapture/HoloLensCaptureExporter
  return [
      [0, 1],# Thumb
      [0, 2],# Index
      [0, 3],# Middle
      [0, 4],# Ring
      [0, 5],# Pinky
  ]


def read_intrinsics_txt(
    img_instrics_path,
      # resized_width, resized_height
):
  with open(img_instrics_path) as f:
    data = list(map(float, f.read().split('\t')))
    intrinsics = np.array(data[:9]).reshape(3, 3)
    width = int(data[-2])
    height = int(data[-1])

  # scale_x = resized_width / width
  # scale_y = resized_height / width
  # intrinsics[0] = intrinsics[0] * scale_x
  # intrinsics[1] = intrinsics[1] * scale_y
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
      # print(
      if  np.abs(int(points[limb[0]][0][0])) > 5000 or \
        np.abs(int(points[limb[0]][0][1])) > 5000 or \
        np.abs(int(points[limb[1]][0][0])) > 5000 or \
        np.abs(int(points[limb[1]][0][1])) > 5000:
            continue
      # )
      print((int(points[limb[0]][0][0]), int(points[limb[0]][0][1])),
        (int(points[limb[1]][0][0]), int(points[limb[1]][0][1])))
      cv2.line(
        img,
        (int(points[limb[0]][0][0]), int(points[limb[0]][0][1])),
        (int(points[limb[1]][0][0]), int(points[limb[1]][0][1])),
        color, thickness
      )

  return img
