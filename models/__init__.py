import torch.nn as nn
from .vggnet import VGG11
from .vision_transformer import ViT
from .swin_transfrmer import SwinTransformer

model_exist = ["VGG11", "ViT", "Swin"]


def creat_model(name: str) -> nn.Module:
    assert name in model_exist, "模型名称使用错误"
    if name == "VGG11":
        model = VGG11()
    elif name == "ViT":
        model = ViT(num_classes=1, emb_dropout=0.1)
    elif name == "Swin":
        model = SwinTransformer(num_classes=1)
    return model