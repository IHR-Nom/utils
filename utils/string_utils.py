import numpy as np


def str2label(value, character_to_index):
    label = []
    for v in value:
        label.append(character_to_index[v])
    return np.array(label, np.uint32)


def label2str_single(label, index_to_word, as_raw, space_char ="~"):
    string = u""
    for i in range(len(label)):
        if label[i] == 0:
            if as_raw:
                string += space_char
            else:
                break
        else:
            val = label[i]
            string += index_to_word[val]
    return string


def naive_decode(output):
    raw_pred_data = np.argmax(output, axis=1)
    pred_data = []
    for i in range(len(output)):
        if raw_pred_data[i] != 0 and not (i > 0 and raw_pred_data[i] == raw_pred_data[i - 1]):
            pred_data.append(raw_pred_data[i])
    return pred_data, list(raw_pred_data)
