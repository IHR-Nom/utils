import torch

from utils import resource_utils

LUC_VAN_TIEN_BOOK_PATH = resource_utils.get_resource('books/Luc-Van-Tien')
TALE_OF_KIEU_PATH = resource_utils.get_resource('books/tale-of-kieu')

DEVICE_ID = "cuda:0" if torch.cuda.is_available() else "cpu"

# Detect if we have a GPU available
device = torch.device(DEVICE_ID)

