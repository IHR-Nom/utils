from torch.utils.data.dataloader import default_collate


def crnn_collate(batch):
    ids = []
    gts = []
    for _batch in batch:
        ids.append(_batch['id'])
        gts.append(_batch['gt'])
        del _batch['gt']
    result = default_collate(batch)
    result['id'] = ids
    result['gt'] = gts
    return result
