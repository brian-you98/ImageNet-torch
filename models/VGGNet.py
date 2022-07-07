import torch
import torch.nn as nn
from torchsummary import summary


class VGG11(nn.Module):
    def __init__(self, img_size=224):
        super(VGG11, self).__init__()
        self.maxpool1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.maxpool2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.maxpool3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.maxpool4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.maxpool5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        nw = int(img_size/32)

        self.dense = nn.Sequential(
            nn.Linear(512 * nw * nw, 4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        pool1 = self.maxpool1(x)
        pool2 = self.maxpool2(pool1)
        pool3 = self.maxpool3(pool2)
        pool4 = self.maxpool4(pool3)
        pool5 = self.maxpool5(pool4)
        flatten = nn.Flatten()(pool5)
        out = self.dense(flatten)
        return out


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg_model = VGG11(img_size=224).to(device)
    summary(vgg_model, (3, 224, 224))  # 打印网络结构
