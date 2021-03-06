# author: Jintao Huang
# date: 2020-5-26

from models.darknet53 import darknet53, preprocess
import torch
import torch.nn as nn
from PIL import Image

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

image_fname = "images/1.jpg"
with Image.open(image_fname) as x:
    x = preprocess([x], 224).to(device)
y_true = torch.randint(0, 1000, (1,)).to(device)

model = darknet53(pretrained=True).to(device)
loss_func = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), 5e-4)

for i in range(20):
    pred = model(x)
    loss = loss_func(pred, y_true)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print("loss: %f" % loss.item())
