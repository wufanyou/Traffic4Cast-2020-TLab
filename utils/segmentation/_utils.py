from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F


class _SimpleSegmentationModel(nn.Module):
    __constants__ = ["aux_classifier"]

    def __init__(self, backbone, classifier, aux_classifier=None):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def logits(self, **kargs):
        x = kargs["x"].float()

        if "extra" in kargs:
            extra = kargs["extra"].float()
            x = torch.cat([x, extra], axis=1)
        x = 2 * x / 255 - 1

        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = x
            return result
        else:
            return x

    def forward(self, **kargs):
        x = self.logits(**kargs)
        x = (x + 1) / 2
        y = kargs["y"].float() / 255
        loss = F.mse_loss(x, y)
        return loss
