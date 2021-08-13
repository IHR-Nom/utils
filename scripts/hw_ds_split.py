import glob
import json
import os

from databases.hw_lvt import ERROR_PAGE
from utils import resource_utils
from utils.constants import LUC_VAN_TIEN_BOOK_PATH, TALE_OF_KIEU_PATH


def splitting(dataset_path, ignore_pages):
    annotation_path = os.path.join(dataset_path, 'annotation.json')
    assert os.path.isfile(annotation_path)
    with open(annotation_path) as fd:
        annotations = json.load(fd)

    patch_pattern = os.path.join(dataset_path, 'patches', '**', '*.json')
    patches = glob.glob(patch_pattern, recursive=True)

    patch_to_remove = set([])
    for page in sorted(ignore_pages, reverse=True):
        remove_annotation = annotations[page]
        img_name = os.path.basename(remove_annotation['img']).split('.jpg')[0]
        for idx, patch in enumerate(patches):
            if img_name in patch:
                patch_to_remove.add(idx)

    for idx, patch in enumerate(patches):
        with open(patch, encoding='utf-8') as f:
            sample = json.load(f)
            hannom = [x for sub in sample for x in sub['hn_text']]
            if '[' in hannom:
                patch_to_remove.add(idx)
            if '-' in hannom:
                patch_to_remove.add(idx)

    for idx in sorted(list(patch_to_remove), reverse=True):
        del patches[idx]

    n_val = int(len(patches) * 0.2)

    word_map = {}
    for patch in patches:
        with open(patch) as f:
            gt = json.load(f)
        hannom = [x for sub in gt for x in sub['hn_text']]
        word_map[patch] = []
        for word in hannom:
            if word not in word_map[patch]:
                word_map[patch].append(word)

    ranking = []
    for patch in word_map:
        curr = set(word_map[patch])
        other_words = [x for sub in word_map for x in word_map[sub] if sub != patch]
        appearance = []
        for x in other_words:
            if x in curr:
                appearance.append(x)
        intersect = curr & set(other_words)
        ranking.append({'patch': patch, 'appearance': len(appearance), 'intersect': len(intersect)})

    ranking = sorted(ranking, key=lambda x: (x['intersect'], x['appearance']), reverse=True)
    val = [x['patch'] for x in ranking[:n_val]]
    train = [x['patch'] for x in ranking[n_val:]]
    return train, val


if __name__ == '__main__':
    tk_train, tk_val = splitting(TALE_OF_KIEU_PATH, [])
    lvt_train, lvt_val = splitting(LUC_VAN_TIEN_BOOK_PATH, ERROR_PAGE)

    val_path = resource_utils.get_resource('books/val.json', create_parent_dir=True)
    train_path = resource_utils.get_resource('books/train.json', create_parent_dir=True)

    with open(train_path, 'w') as f:
        json.dump(lvt_train + tk_train, f)

    with open(val_path, 'w') as f:
        json.dump(lvt_val + tk_val, f)



