import torch

import helper

from config import DEVICE
from model import SegmentationModel
from utils import get_dataset

model = SegmentationModel()
model.to(DEVICE)

trainset, validset = get_dataset()
print('validset', validset)

idx = 6

model.load_state_dict(torch.load('best_model.pt')) #best model

image, mask = validset[idx]

logits_mask = model(image.to(DEVICE).unsqueeze(0)) #(c, h, w) => (1, c, h, w)
pred_mask = torch.sigmoid(logits_mask)
pred_mask = (pred_mask > 0.5).float()

helper.show_image(image, mask, pred_mask.detach().cpu().squeeze(0))