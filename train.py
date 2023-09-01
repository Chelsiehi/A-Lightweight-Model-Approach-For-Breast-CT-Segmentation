import torch
import torch.nn as nn
from dataset import get_data_loader
from losses import unetLoss
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
from base_model import UNet
from lightweight_model import LightWeightUNet

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)


def accuracy(output, target):
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(target)
    return acc


def sample_images(batches_done, model, model_name, dataloader):
    model.eval()
    # Saves a generated sample from the validation set
    batch = next(iter(dataloader))
    source = Variable(batch["source"].type(Tensor))
    mask = Variable(batch["mask"].type(Tensor))
    mask_pred = model(source)
    mask_pred = torch.argmax(mask_pred, dim=1).unsqueeze(1)
    img_sample = torch.cat((mask.data, mask_pred.data), -2)
    save_image(img_sample, "result/%s/train_%s.png" % (model_name, batches_done), nrow=4, normalize=True)


def train(model, f_name='u_net_train.txt', model_name='u-net', n_classes=2):
    f = open(f_name, 'w')
    epochs = 200
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    train_loader = get_data_loader()
    
    for epoch in range(epochs):
        model.train()
        loss1_total, loss2_total, loss4_total, acc, pix_acc, times = 0, 0, 0, 0, 0, 0
        for i, data in enumerate(train_loader, 0):
            source, true_masks, label, id = data["source"].to(device), data["mask"].to(device), data["label"], data[
                "id"].to(
                device)
            optimizer.zero_grad()
            masks_pred = model(source)

            if model.n_classes == 1:
                loss1, pix_ac = unetLoss(masks_pred, true_masks)
            else:
                loss1, pix_ac = unetLoss(masks_pred, true_masks)
            loss = loss1
            loss1_total += loss1.item()
            pix_acc += pix_ac
            times += 1
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0: sample_images(epoch, model, model_name, train_loader)
        f.write(
            "epoch:{}, dice_loss:{}, pix acc:{}\n".format(epoch, loss1_total / times, pix_acc / times))
        print("epoch:{}, dice_loss:{}, pix acc:{}".format(epoch, loss1_total / times, pix_acc / times))
    torch.save(model.state_dict(), "{}.pth".format(model_name))
    print('Finished Training')
    f.flush()
    f.close()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=3, n_classes=2).to(device)
model.apply(weight_init)
train(model, 'u_net_train.txt', 'u-net', 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LightWeightUNet(n_channels=3, n_classes=2).to(device)
model.apply(weight_init)
train(model, 'LightWeightUNet_train.txt', 'LightWeightUNet', 2)
