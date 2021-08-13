import json
import os

import cv2

from databases.hw_lvt import MAX_PIXEL_WIDTH, MAX_PIXEL_HEIGHT
from preprocessing.image import resize_filling
from utils import resource_utils
from utils.constants import LUC_VAN_TIEN_BOOK_PATH

val_path = resource_utils.get_resource('books/val.json')
train_path = resource_utils.get_resource('books/train.json')
all_chars = set([])
min_w, max_w = 9990, 0


def gen_data(conf, new_db_path):
    global min_w, max_w
    os.makedirs(new_db_path, exist_ok=True)
    lines = []
    for idx, path in enumerate(conf):
        gt_path = path
        img_path = gt_path.replace('.json', '.jpg')
        with open(gt_path, encoding='utf-8') as f:
            gt = json.load(f)
        hannom = [x for sub in gt for x in sub['hn_text']]
        for item in hannom:
            all_chars.add(item)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        h, w, c = image.shape
        if w > max_w:
            max_w = w
        if w < min_w:
            min_w = w
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = resize_filling(image, (MAX_PIXEL_WIDTH, 432))
        # percent = float(32) / image.shape[1]
        # image = cv2.resize(image, (0, 0), fx=percent, fy=percent, interpolation=cv2.INTER_AREA)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_name = os.path.basename(img_path)
        file_path = os.path.join(new_db_path, img_name)
        cv2.imwrite(file_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        lines.append(file_path)
        lines.append(''.join(hannom))
    return lines


out_path = resource_utils.get_resource('newdb')
with open(val_path) as f:
    lines = gen_data(json.load(f), os.path.join(out_path, 'val'))
    with open(os.path.join(out_path, 'val.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

with open(train_path) as f:
    lines = gen_data(json.load(f), os.path.join(out_path, 'train'))
    with open(os.path.join(out_path, 'train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

print('All characters: ')
# print(''.join(all_chars))
