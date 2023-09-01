from base_model import UNet
from lightweight_model import LightWeightUNet
from torchstat import stat

unet = UNet(3, 2)
light_unet = LightWeightUNet(3, 2)
unet_report = stat(unet, (3, 256, 256))
stat(light_unet, (3, 256, 256))
