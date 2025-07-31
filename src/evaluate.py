import torch
import pandas as pd

from sklearn.model_selection import train_test_split

from config import DEVICE, CSV_FILE
from helper import show_image
from model import SegmentationModel
from dataset import SegmentationDataset
from augmentation import get_train_augs, get_valid_augs

model = SegmentationModel()
model.to(DEVICE)

df = pd.read_csv(CSV_FILE)
df.head()

train_df, valid_df = train_test_split(df, test_size = 0.2, random_state = 42)

trainset = SegmentationDataset(train_df, get_train_augs())
validset = SegmentationDataset(valid_df, get_valid_augs())

idx = 50

model.load_state_dict(torch.load('best_model.pt')) #best model

image, mask = validset[idx]
logits_mask = model(image.to(DEVICE).unsqueeze(0)) #(c, h, w) => (1, c, h, w)
pred_mask = torch.sigmoid(logits_mask)
pred_mask = (pred_mask > 0.5).float()

show_image(image, mask, pred_mask.detach().cpu().squeeze(0))