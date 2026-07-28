"""VertexHead: dense keypoint (pose) branch.

LOFOP's original pose design follows the same anchor-free philosophy as
ApexHead: every pyramid location regresses, for each keypoint, an (x, y)
offset from the location center in stride units plus a visibility logit. A
detection's skeleton is read off at its winning location -- no heatmaps, no
second-stage crop, so pose costs one extra tower. Used by
:class:`~lofop.models.LofopPose` (``lofop-detect-*-pose`` variants).
"""

from __future__ import annotations

from torch import Tensor, nn

from lofop.core.exceptions import ModelError
from lofop.registries import HEADS


def _group_count(width: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if width % groups == 0:
            return groups
    return 1


@HEADS.register()
class VertexHead(nn.Module):
    """Per-location keypoint offsets and visibilities over a pyramid.

    Args:
        width: Channel width of the incoming pyramid maps.
        num_keypoints: Keypoints per instance (COCO person = 17).
        num_convs: Depth of the shared keypoint tower.

    Forward takes ``[P3, P4, P5]`` and returns two lists (one entry per
    level): offsets (B, 2*K, H, W) in stride units, interleaved
    ``(dx1, dy1, dx2, dy2, ...)``, and visibility logits (B, K, H, W).
    """

    def __init__(self, width: int = 96, num_keypoints: int = 17, num_convs: int = 2) -> None:
        super().__init__()
        if num_keypoints < 1:
            raise ModelError(
                "num_keypoints must be >= 1", context={"num_keypoints": num_keypoints}
            )
        self.num_keypoints = num_keypoints
        layers: list[nn.Module] = []
        for _ in range(num_convs):
            layers += [
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.GroupNorm(_group_count(width), width),
                nn.SiLU(inplace=True),
            ]
        self.tower = nn.Sequential(*layers)
        self.offset_pred = nn.Conv2d(width, 2 * num_keypoints, 3, padding=1)
        self.visibility_pred = nn.Conv2d(width, num_keypoints, 3, padding=1)

    def forward(self, features: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        offsets, visibilities = [], []
        for level in features:
            feat = self.tower(level)
            offsets.append(self.offset_pred(feat))
            visibilities.append(self.visibility_pred(feat))
        return offsets, visibilities
