"""StencilHead: prototype-based instance segmentation branch.

LOFOP's original mask design: a small tower on the finest pyramid level (P3)
produces a shared bank of prototype masks at stride 4, and a per-location
coefficient tower on every pyramid level predicts how to linearly combine
them. A detection's mask is ``sigmoid(coefficients . prototypes)`` cropped to
its box -- one prototype bank per image, so mask cost grows with detections,
not with locations. Used by :class:`~lofop.models.LofopSegment`
(``lofop-detect-*-seg`` variants).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lofop.core.exceptions import ModelError
from lofop.registries import HEADS


def _group_count(width: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if width % groups == 0:
            return groups
    return 1


def _conv_block(in_channels: int, out_channels: int) -> list[nn.Module]:
    return [
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(_group_count(out_channels), out_channels),
        nn.SiLU(inplace=True),
    ]


@HEADS.register()
class StencilHead(nn.Module):
    """Prototype + coefficient branch over a feature pyramid.

    Args:
        width: Channel width of the incoming pyramid maps.
        num_prototypes: Size of the shared prototype bank.
        num_convs: Depth of the prototype and coefficient towers.

    Forward takes ``[P3, P4, P5]`` and returns ``(prototypes, coefficients)``:
    prototypes (B, K, 2*H3, 2*W3) at stride 4, and one (B, K, H, W)
    coefficient map per level (tanh, so combinations can subtract prototypes).
    """

    def __init__(self, width: int = 96, num_prototypes: int = 16, num_convs: int = 2) -> None:
        super().__init__()
        if num_prototypes < 1:
            raise ModelError(
                "num_prototypes must be >= 1", context={"num_prototypes": num_prototypes}
            )
        self.num_prototypes = num_prototypes
        proto_layers: list[nn.Module] = []
        for _ in range(num_convs):
            proto_layers += _conv_block(width, width)
        self.proto_tower = nn.Sequential(*proto_layers)
        self.proto_pred = nn.Conv2d(width, num_prototypes, 1)
        coeff_layers: list[nn.Module] = []
        for _ in range(num_convs):
            coeff_layers += _conv_block(width, width)
        self.coeff_tower = nn.Sequential(*coeff_layers)
        self.coeff_pred = nn.Conv2d(width, num_prototypes, 3, padding=1)

    def forward(self, features: list[Tensor]) -> tuple[Tensor, list[Tensor]]:
        finest = features[0]
        proto = self.proto_tower(finest)
        proto = F.interpolate(proto, scale_factor=2.0, mode="bilinear", align_corners=False)
        prototypes = self.proto_pred(proto)
        coefficients = [
            torch.tanh(self.coeff_pred(self.coeff_tower(level))) for level in features
        ]
        return prototypes, coefficients
