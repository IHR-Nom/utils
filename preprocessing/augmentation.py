import cv2
import numpy as np

from preprocessing import grid_distortion


def tensmeyer_brightness(img, foreground=0, background=0):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    th = (th.astype(np.float32) / 255)[..., None]

    img = img.astype(np.float32)
    img = img + (1.0 - th) * foreground
    img = img + th * background

    img[img > 255] = 255
    img[img < 0] = 0

    return img.astype(np.uint8)


class TensmeyerBrightness:

    def __init__(self, sigma=30, prob=0.3):
        self.sigma = sigma
        self.prob = prob

    def __call__(self, img):
        should_transform = np.random.choice(np.arange(0, 2), p=[1 - self.prob, self.prob])
        if not should_transform:
            return img
        random_state = np.random.RandomState(None)
        foreground = random_state.normal(0, self.sigma)
        background = random_state.normal(0, self.sigma)

        img = tensmeyer_brightness(img, foreground, background)

        return img


class GridDistortion:

    def __init__(self, prob=0.3):
        self.prob = prob

    def __call__(self, img):
        should_transform = np.random.choice(np.arange(0, 2), p=[1 - self.prob, self.prob])
        if should_transform:
            return grid_distortion.warp_image(img)
        return img


class RandomColorRotation:

    def __init__(self, prob=0.3):
        self.prob = prob

    def __call__(self, img):
        should_transform = np.random.choice(np.arange(0, 2), p=[1 - self.prob, self.prob])
        if not should_transform:
            return img
        random_state = np.random.RandomState(None)
        shift = random_state.randint(0, 255)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = hsv[..., 0] + shift
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return img

