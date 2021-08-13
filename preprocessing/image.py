import cv2
import numpy as np
import torchvision
import matplotlib.pyplot as plt


def resize_filling(image, new_size, color=None):
    n_width, n_height = new_size
    height, width = image.shape[:2]
    if width > n_width:
        ratio = n_width / width
        image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    blank_image = np.zeros((n_height, n_width, 3), np.uint8)
    if color is None:
        color = bincount_app(image)
    lower = np.array([color[0] - 20, color[1] - 20, color[2] - 20])
    upper = np.array([color[0] + 20, color[1] + 20, color[2] + 20])
    mask = cv2.inRange(image, lower, upper)
    masked_image = np.copy(image)
    masked_image[mask != 0] = color

    # img_bw = 255 * (cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY) > 10).astype('uint8')
    #
    # se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    #
    # mask = np.dstack([mask, mask, mask]) / 255
    # out = masked_image * mask
    #
    blank_image[:] = color

    x_offset, y_offset = int((n_width - width) / 2), 10
    # Here, y_offset+height <= blank_image.shape[0] and x_offset+width <= blank_image.shape[1]
    blank_image[y_offset:y_offset + height, x_offset:x_offset + width] = masked_image.copy()
    # plt.figure()
    # plt.imshow(blank_image)
    #
    # plt.axis('off')
    # plt.ioff()
    # # plt.pause(0.05)
    # # plt.clf()
    # plt.show()
    return blank_image


def generate_background(image):
    dominant_color = bincount_app(image)
    lower = np.array([dominant_color[0] - 20, dominant_color[1] - 20, dominant_color[2] - 20])
    upper = np.array([dominant_color[0] + 20, dominant_color[1] + 20, dominant_color[2] + 20])
    mask = cv2.inRange(image, lower, upper)
    masked_image = np.copy(image)
    masked_image[mask != 0] = dominant_color

    return masked_image


def bincount_app(a):
    a2D = a.reshape(-1,a.shape[-1])
    col_range = (256, 256, 256) # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)


class ImageNormalization:
    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, img):
        img = img / 128.0 - 1.0
        return self.to_tensor(img)
