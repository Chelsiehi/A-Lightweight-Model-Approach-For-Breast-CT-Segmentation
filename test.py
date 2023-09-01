import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from base_model import UNet

transform_rgb = transforms.Compose([
    transforms.Resize((256, 256), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

data = transform_rgb(Image.open('Dataset_BUSI_with_GT/malignant/malignant (3).png')).unsqueeze(0)
breast_type_dict = {0: "benign", 1: "malignant", 2: "normal"}
model = UNet(3, 2)
model.load_state_dict(torch.load('u-net.pth'))
model.eval()
with torch.no_grad():
    mask = model(data)
    mask = torch.argmax(mask, dim=1)[0] * 255
    mask = mask.unsqueeze(0)
    for m in mask:
        print(m)
    print(mask.shape)

    transform = transforms.ToPILImage()
    mask = mask.to(torch.uint8)
    img = transform(mask)

    
    img.save('output.png')
