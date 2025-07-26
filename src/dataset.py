import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from config import IMAGE_SIZE

class SegmentationDataset(Dataset):

  def __init__(self, df, augmentations):

    self.df = df
    self.augmentations = augmentations

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]

    image_path = row.images
    mask_path = row.masks

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) #(h, w, c)
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
    mask = np.expand_dims(mask, axis= -1)  # => (H, W, 1)



    if self.augmentations:
      data = self.augmentations(image = image, mask = mask)
      image = data['image']
      mask = data['mask']

    #(h, w, c) -> (c, h, w)

    image = np.transpose(image, (2,0,1)).astype(np.float32)
    mask = np.transpose(mask, (2,0,1)).astype(np.float32)

    image = torch.round(torch.Tensor(image) / 255.0)
    mask = torch.round(torch.Tensor(mask) / 255.0)

    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)

    return image, mask