import json

from utils import resource_utils

val_path = resource_utils.get_resource('books/val.json')
train_path = resource_utils.get_resource('books/train.json')


def read_data(file):
    sentences = []
    char_map = {}
    with open(file) as f:
        for patch in json.load(f):
            with open(patch) as fp:
                gt = json.load(fp)
            han_nom = [x for sub in gt for x in sub['hn_text']]
            sentences.append(han_nom)
            for char in han_nom:
                if char not in char_map:
                    char_map[char] = 0
                char_map[char] += 1

    return sentences, char_map


val_sentences, val_char_map = read_data(val_path)
train_sentences, train_char_map = read_data(train_path)

intersection = len(set(val_char_map.keys()).intersection(set(train_char_map.keys())))
print(intersection / len(val_char_map.keys()))
print(intersection / len(train_char_map.keys()))






