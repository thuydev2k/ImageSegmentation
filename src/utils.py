import torch
import cv2

import pandas as pd
import matplotlib.pyplot as plt
import helper

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import DEVICE, CSV_FILE
from dataset import SegmentationDataset
from augmentation import get_train_augs, get_valid_augs

def train_fn(data_loader, model, optimizer):
  model.train()
  total_loss = 0.0

  for images, masks in tqdm(data_loader):
    images = images.to(DEVICE)
    masks = masks.to(DEVICE)

    optimizer.zero_grad()
    loss = model(images, masks)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

    return total_loss / len(data_loader)
  
def eval_fn(data_loader, model):
  model.eval()
  total_loss = 0.0

  with torch.no_grad():
    for images, masks in tqdm(data_loader):
      images = images.to(DEVICE)
      masks = masks.to(DEVICE)

      loss = model(images, masks)

      total_loss += loss.item()

  return total_loss / len(data_loader)

def show_original_image_and_mask ():
    df = pd.read_csv(CSV_FILE)
    df.head()

    row = df.iloc[5]
    image_path = row.images
    mask_path = row.masks

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0

    (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    ax1.set_title('IMAGE')
    ax1.imshow(image)

    ax2.set_title('GROUND TRUTH')
    ax2.imshow(mask,cmap = 'gray')

def get_dataset ():
    df = pd.read_csv(CSV_FILE)
    df.head()

    train_df, valid_df = train_test_split(df, test_size = 0.2, random_state = 42)

    trainset = SegmentationDataset(train_df, get_train_augs())
    validset = SegmentationDataset(valid_df, get_valid_augs())

    print(f"Size of Trainset : {len(trainset)}")
    print(f"Size of Validset : {len(validset)}")

    return trainset, validset

def show_image_from_trainset (trainset):
    idx = 32

    image, mask = trainset[idx]
    helper.show_image(image, mask)
