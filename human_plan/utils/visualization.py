import cv2
import numpy as np

def project_points(
  points, img_intrinsics
):
  # Put an empty camera pose for image.
  rvec = np.array([0.0, 0.0, 0.0])
  tvec = np.array([0.0, 0.0, 0.0])
  points = points.astype(np.float32)

  points_2d, _ = cv2.projectPoints(
    points, rvec, tvec, img_intrinsics, np.array([])
  )
  points_2d = np.array(points_2d).reshape(-1, 2)
  
  return points_2d
    # 尝试直接调用并捕获更详细的错误
  # try:
  #   projected_points, _ = cv2.projectPoints(
  #     points, rvec, tvec, img_intrinsics, None  # 改用 None 而不是 np.array([])
  #   )
  #   return projected_points.reshape(-1, 2)
  # except Exception as e:
  #   print(f"错误详情: {e}")
  #   print(f"尝试检查 cv2 版本: {cv2.__version__}")
  #   # 尝试备用方法：手动投影
  #   print("尝试手动投影...")
  #   # 手动实现投影
  #   fx, fy = img_intrinsics[0, 0], img_intrinsics[1, 1]
  #   cx, cy = img_intrinsics[0, 2], img_intrinsics[1, 2]
  #   projected = np.zeros((points.shape[0], 2))
  #   projected[:, 0] = (points[:, 0] / points[:, 2]) * fx + cx
  #   projected[:, 1] = (points[:, 1] / points[:, 2]) * fy + cy
  #   return projected


def plot_points(points, img, color):
  points = points.reshape(-1, 2)
  # img_list = []
  for idx, point in enumerate(points):
    
    img = cv2.circle(
      img, (int(round(point[0])), int(round(point[1]))),
      radius=10, color=color,
      thickness=-1
    )
  return img


def plot_hand(points, img, color, get_handpose_connectivity_func):
  points = points.reshape(-1, 2)

  connectivity = get_handpose_connectivity_func()
  line_thickness = 3
  circle_radius = 5

  if not (np.isnan(points).any()):
    # 画线条连接
    for limb in connectivity:
      pt1 = (int(points[limb[0]][0]), int(points[limb[0]][1]))
      pt2 = (int(points[limb[1]][0]), int(points[limb[1]][1]))
      cv2.line(img, pt1, pt2, color, line_thickness)
    
    # 画关键点（圆圈）
    for point in points:
      pt = (int(point[0]), int(point[1]))
      cv2.circle(img, pt, circle_radius, color, -1)  # 实心圆

  return img

def get_handpose_connectivity_sim():
  return [
      [0, 1],
      [0, 2],
      [0, 3],
      [0, 4],
      [0, 5],
  ]

def plot_hand_sim(points, img, color):
  return plot_hand(points, img, color, get_handpose_connectivity_sim)

def get_handpose_connectivity_mano():
  return [
      # Thumb
      [0, 1], [1, 2], [2, 3], [3, 4],
      # Index
      [0, 5], [5, 6], [6, 7], [7, 8],
      # Middle
      [0, 9], [9, 10], [10, 11], [11, 12],
      # Ring
      [0, 13], [13, 14], [14, 15], [15, 16],
      # Pinky
      [0, 17], [17, 18], [18, 19], [19, 20],
  ]

def plot_hand_mano(points, img, color):
  return plot_hand(points, img, color, get_handpose_connectivity_mano)
