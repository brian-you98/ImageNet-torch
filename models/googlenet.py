import torch
import torch.nn as nn
from torchsummary import summary


class GoogleNET(nn.Module):
    def __init__(self, img_size=224):
        super(GoogleNET, self).__init__()

    def forward(self, x):
        return x


if __name__ == "__main__":
    device = torch.device('cpu')
    vgg_model = GoogleNET(img_size=224).to(device)
    summary(vgg_model, (3, 224, 224), 1, 'cpu')
