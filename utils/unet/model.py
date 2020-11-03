# Created by fw at 10/9/20
# modify from https://github.com/qubvel/segmentation_models.pytorch

from typing import Optional, Union, List
from .decoder import UnetDecoder, SCSEModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder


class ArgMax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Activation(nn.Module):
    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "argmax":
            self.activation = ArgMax(**params)
        elif name == "argmax2d":
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                "Activation should be callable/sigmoid/softmax/logsoftmax/None; got {}".format(
                    name
                )
            )

    def forward(self, x):
        return self.activation(x)


class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class SegmentationHead(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1
    ):
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class ClassificationHead(nn.Sequential):
    def __init__(
        self, in_channels, classes, pooling="avg", dropout=0.2, activation=None
    ):
        if pooling not in ("max", "avg"):
            raise ValueError(
                "Pooling should be one of ('max', 'avg'), got {}.".format(pooling)
            )
        pool = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else nn.AdaptiveMaxPool2d(1)
        flatten = Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)


def initialize_decoder(module):
    for m in module.modules():

        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Unet(nn.Module):
    """Unet_ is a fully convolution neural network for image semantic segmentation

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_depth (int): number of stages used in decoder, larger depth - more features are generated.
            e.g. for depth=3 encoder will generate list of features with following spatial shapes
            [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature tensor will have
            spatial resolution (H/(2^depth), W/(2^depth)]
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used. If 'inplace' InplaceABN will be used, allows to decrease memory consumption.
            One of [True, False, 'inplace']
        decoder_attention_type: attention module used in decoder of the model
            One of [``None``, ``scse``]
        in_channels: number of input channels for model, default is 3.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function to apply after final convolution;
            One of [``sigmoid``, ``softmax``, ``logsoftmax``, ``identity``, callable, None]
        aux_params: if specified model will have additional classification auxiliary output
            build on top of encoder, supported params:
                - classes (int): number of classes
                - pooling (str): one of 'max', 'avg'. Default is 'avg'.
                - dropout (float): dropout factor in [0, 1)
                - activation (str): activation function to apply "sigmoid"/"softmax" (could be None to return logits)

    Returns:
        ``torch.nn.Module``: **Unet**

    .. _Unet:
        https://arxiv.org/pdf/1505.04597

    """

    def __init__(
        self,
        encoder_name: str = "efficientnet-b3",
        encoder_depth: int = 5,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.encoder = get_encoder(
            encoder_name, in_channels=in_channels, depth=encoder_depth, weights=None,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            initialize_head(self.classification_head)

    def logits(self, **kargs):
        x = kargs["x"].float()
        if "extra" in kargs:
            extra = kargs["extra"].float()
            x = torch.cat([x, extra], axis=1)
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels
        return masks

    def forward(self, **kargs):
        x = self.logits(**kargs)
        x = (x + 1) / 2
        y = kargs["y"].float() / 255
        if "mask" in kargs:
            x = x * kargs["mask"]
        loss = F.mse_loss(x, y)
        return loss


class UnetV1(Unet):
    def logits(self, **kargs):
        x = kargs["x"].float()
        if "extra" in kargs:
            extra = kargs["extra"].float()
            x = torch.cat([x, extra], axis=1)
        x = 2 * x / 255 - 1
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels
        return masks

    def forward(self, **kargs):
        x = self.logits(**kargs)
        x = (x + 1) / 2
        y = kargs["y"].float() / 255
        if "mask" in kargs:
            x = x * kargs["mask"]
        loss = F.mse_loss(x, y)
        return loss


class UnetGaussRank(Unet):
    def logits(self, **kargs):
        x = kargs["x"].float()
        if "extra" in kargs:
            extra = kargs["extra"].float()
            extra = extra / 255
            x = torch.cat([x, extra], axis=1)
        x = 2 * x - 1
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels
        return masks

    def forward(self, **kargs):
        x = self.logits(**kargs)
        x = (x + 1) / 2
        y = kargs["y"].float() / 255
        if "mask" in kargs:
            x = x * kargs["mask"]
        loss = F.mse_loss(x, y)
        return loss


def get_unet(**kwargs):
    model = globals()[kwargs["model_version"]](**kwargs)
    return model
