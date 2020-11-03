import torch
import torch.nn as nn
from .segmentation import deeplabv3_resnet50, deeplabv3_resnet101

__ALL__ = ["get_model"]
BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01


class Transform(nn.Module):
    def forward(self, input):
        return 2 * input / 255 - 1


def load_pretrain_model(model, pretrain: str, city):
    if pretrain != "":
        if pretrain.find("{city}") != -1:
            pretrain = pretrain.format(city=city)
        pretrain = torch.load(pretrain, map_location="cpu")
        weight = model.state_dict()
        for k, v in pretrain.items():
            if k in weight:
                if v.shape == weight[k].shape:
                    weight[k] = v
        model.load_state_dict(weight)


def get_hrnet(
    input_channels, num_classes, model_version, pos_weight=None, cfg_path="", **kwargs
):
    from .hrnet import get_cfg, get_seg_model

    model_cfg = get_cfg()
    if cfg_path != "":
        model_cfg.merge_from_file(cfg_path)
    model_cfg.NUM_CLASSES = num_classes
    model = get_seg_model(
        model_cfg,
        model_version=model_version,
        pos_weight=pos_weight,
        all_cfg=kwargs["cfg"],
    )

    if model_version in ["HighResolutionNetGaussRank"]:
        input_transform = nn.Identity()
    else:
        input_transform = Transform()

    if "frozen_layers" in kwargs:
        if kwargs["frozen_layers"]:
            for param in model.parameters():
                param.requires_grad = False

    model.conv1 = nn.Sequential(
        input_transform,
        nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
    )

    last_inp_channels = model.last_layer[0].in_channels

    # redefine last layer
    model.last_layer = nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=last_inp_channels,
            out_channels=last_inp_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
        ),
        BatchNorm2d(last_inp_channels // 2, momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(
            in_channels=last_inp_channels // 2,
            out_channels=last_inp_channels // 2 // 2,  # config.NUM_CLASSES,
            kernel_size=3,
            stride=2,
            padding=0,
            output_padding=(0, 1),
        ),
        BatchNorm2d(last_inp_channels // 2 // 2, momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            in_channels=last_inp_channels // 2 // 2,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )
    return model


def get_unet(input_channels, num_classes, model_version, **kwargs):
    from .unet import get_unet

    model = get_unet(
        model_version=model_version, in_channels=input_channels, classes=num_classes
    )
    return model


# useless
def get_deeplabv3(
    input_channels, num_classes, model_version="deeplabv3_resnet50", **kwargs
):
    model_map = {
        "deeplabv3_resnet50": deeplabv3_resnet50,
        "deeplabv3_resnet101": deeplabv3_resnet101,
    }
    model = model_map[model_version](
        pretrained=False, progress=False, num_classes=num_classes, aux_loss=None
    )
    model.backbone.conv1 = nn.Conv2d(
        input_channels,
        64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False,
    )
    return model


def replace_relu(model, activation="ReLU"):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            if activation in ["GELU", "CELU"]:
                setattr(model, child_name, eval(f"nn.{activation}()"))
            else:
                setattr(model, child_name, eval(f"nn.{activation}(inplace=True)"))
        else:
            replace_relu(child, activation)


MODEL_MAPS = {"hrnet": get_hrnet, "deeplabv3": get_deeplabv3, "unet": get_unet}


def get_model(
    cfg, city=None,
):
    model_name = cfg.MODEL.NAME.lower()
    model_version = cfg.MODEL.MODEL_VERSION
    num_classes = cfg.DATASET.OUTPUT_CHANNELS  # .lower()
    input_channels = cfg.DATASET.INPUT_CHANNELS
    assert model_name in MODEL_MAPS, "model is not allowed"
    pos_weight = cfg.MODEL.POS_WEIGHT if cfg.MODEL.USE_POS_WEIGHT else None

    model = MODEL_MAPS[model_name](
        input_channels,
        num_classes,
        model_version,
        pos_weight=pos_weight,
        cfg_path=cfg.MODEL.MODEL_CONFIG_FILE,
        cfg=cfg,
        frozen_layers=cfg.MODEL.FROZEN_LAYERS,
    )
    load_pretrain_model(model, cfg.DIST.PRETRAIN_MODEL, city=city)
    if cfg.MODEL.HIDDEN_ACTIVATION != "default":
        replace_relu(model, cfg.MODEL.HIDDEN_ACTIVATION)
    return model
