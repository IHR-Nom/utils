import glob
import os

import cv2

from utils.constants import LUC_VAN_TIEN_BOOK_PATH

dataset_path = LUC_VAN_TIEN_BOOK_PATH
patch_pattern = os.path.join(dataset_path, 'patches', '**', '*.jpg')

max_width, max_height = 0, 0

for img_path in glob.glob(patch_pattern, recursive=True):
    img = cv2.imread(img_path)
    h, w, c = img.shape
    if h > max_height:
        max_height = h
    if w > max_width:
        max_width = w

print(f'Max width: {max_width}, Max height: {max_height}')