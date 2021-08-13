import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.image import resize_filling
from utils import string_utils, debug_utils
from utils.constants import LUC_VAN_TIEN_BOOK_PATH

MAX_PIXEL_HEIGHT = 292
MAX_PIXEL_WIDTH = 48
ERROR_PAGE = [34, 103, 11, 14]


def get_word_map(dataset_path=LUC_VAN_TIEN_BOOK_PATH):
    annotation_path = os.path.join(dataset_path, 'annotation.json')
    assert os.path.isfile(annotation_path)
    with open(annotation_path) as fd:
        annotations = json.load(fd)

    for page in sorted(ERROR_PAGE, reverse=True):
        del annotations[page]

    words = set([])
    for page in annotations:
        for column in page['annotations']:
            for word in column['hn_text']:
                words.add(word)
    words = [' '] + sorted(list(words))
    idx_to_word, word_to_idx = {}, {}
    for idx, item in enumerate(words):
        idx_to_word[idx] = item
        word_to_idx[item] = idx

    return idx_to_word, word_to_idx


class HWLVT(Dataset):
    def __init__(self, word_map, patches_file, dataset_path=LUC_VAN_TIEN_BOOK_PATH, transforms=None, img_width=32):

        self.transforms = transforms
        self.idx_to_word, self.word_to_idx = word_map
        with open(patches_file) as f:
            patches = json.load(f)
        self.patches = [dataset_path + x for x in patches]
        self.img_width = img_width

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        with open(patch) as f:
            gt = json.load(f)
        image_path = patch.replace('.json', '.jpg')
        org_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)

        image = resize_filling(image, (MAX_PIXEL_WIDTH, MAX_PIXEL_HEIGHT))
        percent = float(self.img_width) / image.shape[1]
        image = cv2.resize(image, (0, 0), fx=percent, fy=percent, interpolation=cv2.INTER_CUBIC)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # debug_utils.show_image(image)

        if self.transforms:
            image = self.transforms(image)

        hannom = [x for sub in gt for x in sub['hn_text']]
        label = string_utils.str2label(hannom, self.word_to_idx)
        res = {
            'image': image,
            'id': patch,
            'gt': torch.from_numpy(label.astype(np.int32)),
            'gt_length': len(label)
        }
        return res


def get_all_pages():
    return [x for x in range(104)]
