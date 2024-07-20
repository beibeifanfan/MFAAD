
# https://github.com/lmas/opensimplex

from ctypes import c_int64
from math import floor
import scipy
import numpy as np
from scipy.interpolate import BSpline, BarycentricInterpolator
import cv2
import random

height = 256
width = 608

def generate_hair_curve(image_shape, num_points=100, hair_length=50, hair_width=2):
    # Generate random root position for the hair
    root_position = np.random.uniform(0, 1, size=(1, 2))
    root_position[:, 0] *= image_shape[1]  # Scale x coordinate to image width
    root_position[:, 1] *= image_shape[0]  # Scale y coordinate to image height

    # Generate control points for the hair curve
    x_root, y_root = root_position[0]
    x = np.linspace(x_root, x_root + hair_length, num=num_points)
    y = np.random.normal(y_root, scale=hair_width, size=num_points)

    # Add random perturbation to control points to make the curve more natural
    y += np.random.normal(scale=0.8, size=num_points)

    # Generate B-spline curve for the hair
    spline = BSpline(np.arange(num_points), np.column_stack((x, y)), 3)
    t = np.linspace(0, num_points - 1, num_points * 10)
    x_curve, y_curve = spline(t).T
    x_curve = np.clip(x_curve, 0, image_shape[1] - 1).astype(int)
    y_curve = np.clip(y_curve, 0, image_shape[0] - 1).astype(int)

    return (x_curve, y_curve)

def add_single_bspline_noise(image, num_hairs=1 ,color=(128, 128, 128)):
    # noisy_image = np.copy(image)
    noisy_image = np.zeros((height, width, 3))
    for j in range(num_hairs):
        hair = generate_hair_curve(image.shape[:2], num_points=15, hair_length=150, hair_width=1)
        x_curve, y_curve = hair
        for i in range(len(x_curve) - 1):
            cv2.line(noisy_image, (x_curve[i], y_curve[i]), (x_curve[i + 1], y_curve[i + 1]), color, thickness=2)

    return noisy_image



class HairNoiseGenerator:
    def __init__(self, image_shape):
        self.image_shape = image_shape

    def generate_b_spline_noise(self, num_hairs=1,  hair_thickness_range=(1, 3),
                                degree=2):
        hair_noise = np.zeros(self.image_shape, dtype=np.float32)

        for _ in range(num_hairs):
            while True:
                try:
                    start_point = (np.random.randint(20, self.image_shape[0]-20), np.random.randint(20, self.image_shape[1]-20))
                    end_point = (np.random.randint(20, self.image_shape[0]-20), np.random.randint(20, self.image_shape[1]-20))

                    # print(start_point,end_point,image_shape[0])
                    min_distance = min(self.image_shape[0], self.image_shape[1]) * 0.4

                    control_point1 = (
                        np.clip(np.random.randint(start_point[0] + min_distance, end_point[0] - min_distance), 0,
                                self.image_shape[1] - 1),
                        np.random.randint(0, self.image_shape[0])
                    )
                    control_point2 = (
                        np.clip(np.random.randint(start_point[0] + min_distance, end_point[0] - min_distance), 0,
                                self.image_shape[1] - 1),
                        np.random.randint(0, self.image_shape[0])
                    )


                    num_knots = len([start_point, control_point1, control_point2, end_point]) + degree + 1
                    knots = np.linspace(0, 1, num_knots)

                    spline = BSpline(knots, [start_point, control_point1, control_point2, end_point], degree)

                    t = np.linspace(0, 1, 100)
                    curve_points = np.array([spline(ti) for ti in t])

                    color =(50,50,50)
                    thickness = 1.5

                    for i in range(len(curve_points) - 1):
                        x1, y1 = np.clip(curve_points[i].astype(int), (0, 0), (self.image_shape[0] - 1, self.image_shape[1] - 1))
                        x2, y2 = np.clip(curve_points[i + 1].astype(int), (0, 0),
                                         (self.image_shape[0] - 1, self.image_shape[1] - 1))
                        hair_noise = cv2.line(hair_noise, (x1, y1), (x2, y2), color, int(thickness))

                    break
                except ValueError:
                    # Handle ValueError by retrying with new random values
                    pass
        hair_noise = hair_noise.astype(np.uint8)
        return hair_noise
