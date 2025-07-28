import torch

import numpy as np

from torch.utils.data import DataLoader

from config import EPOCHS, DEVICE, LR, BATCH_SIZE
from model import SegmentationModel
from utils import train_fn, eval_fn, get_dataset, show_original_image_and_mask, show_image_from_trainset

show_original_image_and_mask()

trainset, validset = get_dataset()

show_image_from_trainset(trainset)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
validloader = DataLoader(validset, batch_size=BATCH_SIZE)

model = SegmentationModel()
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr = LR)

best_valid_loss = np.inf

for i in range(EPOCHS):

  train_loss = train_fn(trainloader, model, optimizer)
  valid_loss = eval_fn(validloader, model, optimizer)

  if valid_loss < best_valid_loss:
    torch.save(model.state_dict(), 'best_model.pt')
    print("SAVED MODEL")
    best_valid_loss = valid_loss

  print(f"Epoch: {i+1} Train_loss: {train_loss} Valid_loss: {valid_loss}")