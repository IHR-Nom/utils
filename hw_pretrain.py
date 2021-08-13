import copy
import logging
import random

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from databases import hw_lvt
from databases.hw_lvt import HWLVT
from network.cnn_lstm import CRNN
from preprocessing.augmentation import RandomColorRotation, TensmeyerBrightness, GridDistortion
from preprocessing.image import ImageNormalization
from utils import string_utils, error_rates, resource_utils
from utils.cnn_utils import crnn_collate
from utils.constants import device

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

pre_processing = transforms.Compose([
    TensmeyerBrightness(), GridDistortion(), ImageNormalization()
])

lr, batch_size, n_epochs = 0.0002, 8, 1000
snapshot_path = resource_utils.get_resource('checkpoints/hw.pt', create_parent_dir=True)
val_path = resource_utils.get_resource('books/Luc-Van-Tien/hw/val.json')
train_path = resource_utils.get_resource('books/Luc-Van-Tien/hw/train.json')
idx_to_word, word_to_idx = hw_lvt.get_word_map()
train_data = HWLVT((idx_to_word, word_to_idx), train_path, transforms=pre_processing)
val_data = HWLVT((idx_to_word, word_to_idx), val_path, transforms=pre_processing)


dl_model = CRNN(1024, 3, len(word_to_idx), 1024).to(device)
optimizer = torch.optim.Adam(dl_model.parameters(), lr=lr)


train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=10, shuffle=True, pin_memory=True,
                          drop_last=True, collate_fn=crnn_collate)
val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=10, pin_memory=True, collate_fn=crnn_collate)

best_loss = 999999
criteria = torch.nn.CTCLoss()

for epoch in range(n_epochs):
    steps, total_loss = 0.0, 0.0
    dl_model.train()
    for batch_data in train_loader:
        inputs = batch_data['image'].to(device, dtype=torch.float, non_blocking=True)
        labels = torch.cat(batch_data['gt']).to(device, dtype=torch.long, non_blocking=True)
        labels_length = batch_data['gt_length'].to(device, dtype=torch.long, non_blocking=True)
        with torch.set_grad_enabled(True):
            preds = dl_model(inputs)
            preds_length = torch.LongTensor([preds.size(0)] * batch_size).to(device)
            log_preds = preds.log_softmax(2)
            loss = criteria(log_preds, labels, preds_length, labels_length)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        output_batch = preds.detach().permute(1, 0, 2)
        for i, gt_line in enumerate(batch_data['gt']):
            logits = output_batch[i].cpu().numpy()
            pred, raw_pred = string_utils.naive_decode(logits)
            pred_str = string_utils.label2str_single(pred, idx_to_word, False)
            actual_str = string_utils.label2str_single(gt_line.numpy(), idx_to_word, False)
            total_loss += error_rates.cer(actual_str, pred_str)
            steps += 1

    train_loss = total_loss / steps
    steps, total_loss = 0.0, 0.0
    dl_model.eval()
    for batch_data in val_loader:
        inputs = batch_data['image'].to(device, dtype=torch.float, non_blocking=True)
        with torch.set_grad_enabled(False):
            preds = dl_model(inputs)

        output_batch = preds.permute(1, 0, 2)
        for i, gt_line in enumerate(batch_data['gt']):
            logits = output_batch[i].cpu().numpy()
            pred, raw_pred = string_utils.naive_decode(logits)
            pred_str = string_utils.label2str_single(pred, idx_to_word, False)
            actual_str = string_utils.label2str_single(gt_line.numpy(), idx_to_word, False)
            total_loss += error_rates.cer(actual_str, pred_str)
            steps += 1

    val_loss = total_loss / steps
    if best_loss > val_loss:
        torch.save(dl_model.state_dict(), snapshot_path)
    logging.info('Epoch %d: Train loss: %.2f, Val loss: %.2f' % (epoch, train_loss, val_loss))


