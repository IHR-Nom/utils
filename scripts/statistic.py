import csv
import json
import sys

from utils import resource_utils

val_path = resource_utils.get_resource('books/val.json')
train_path = resource_utils.get_resource('books/train.json')

patches = []
with open(train_path) as f:
    patches += json.load(f)

with open(val_path) as f:
    patches += json.load(f)

char_count_map = {}
length_count = {}
for patch in patches:
    with open(patch, encoding='utf-8') as f:
        gt = json.load(f)

    hannom = [x for sub in gt for x in sub['hn_text']]

    if len(hannom) not in length_count:
        length_count[len(hannom)] = 0
    length_count[len(hannom)] += 1

    for char in hannom:
        if char not in char_count_map:
            char_count_map[char] = 0
        char_count_map[char] += 1


def group_sum(groups, count_map):
    result = {}
    for character in count_map:
        for group in groups:
            if isinstance(group, int):
                _min, _max = group, group
            else:
                _min, _max = group
            if group not in result:
                result[group] = 0
            val = count_map[character]
            if _min <= val <= _max:
                result[group] += val
    return result


items = group_sum([1, (2, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, sys.maxsize)], char_count_map)

with open('char_count.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=['range', 'count'])
    writer.writeheader()
    for item in items:
        writer.writerow({'range': item, 'count': items[item]})
