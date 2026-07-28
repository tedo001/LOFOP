"""LofopSegment: instance segmentation on top of LOFOP-Detect.

Extends :class:`LofopDetect` with a :class:`~lofop.models.seg_head.StencilHead`
prototype branch. Detection (assignment, box/cls/quality losses, class-aware
NMS) is inherited unchanged; the mask branch is supervised on the same
dynamic assignment, and inference attaches a cropped, thresholded mask to
every detection. Variants: ``lofop-detect-{n,s,ex}-seg`` configs.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lofop.core.exceptions import ModelError
from lofop.models.detector import LofopDetect, flatten_levels
from lofop.models.head import ApexHead
from lofop.models.seg_head import StencilHead
from lofop.registries import MODELS


def _box_region(boxes: Tensor, height: int, width: int, scale_x: float,
                scale_y: float) -> Tensor:
    """(P, height, width) float mask of each box's interior in map coords."""
    xs = torch.arange(width, device=boxes.device, dtype=boxes.dtype) + 0.5
    ys = torch.arange(height, device=boxes.device, dtype=boxes.dtype) + 0.5
    in_x = (xs >= (boxes[:, 0] * scale_x).unsqueeze(1)) & (
        xs <= (boxes[:, 2] * scale_x).unsqueeze(1))
    in_y = (ys >= (boxes[:, 1] * scale_y).unsqueeze(1)) & (
        ys <= (boxes[:, 3] * scale_y).unsqueeze(1))
    return (in_y.unsqueeze(2) & in_x.unsqueeze(1)).to(boxes.dtype)


@MODELS.register()
class LofopSegment(LofopDetect):
    """Anchor-free instance segmentation model.

    Args:
        mask_head: A :class:`StencilHead` (prototypes + coefficients).
        mask_weight: Weight of the mask loss term.
        mask_threshold: Probability above which an output mask pixel is on.
        mask_sample_size: Max positive locations supervised for masks per
            image (bounds mask-loss cost on crowded images).

    Remaining arguments as :class:`LofopDetect`. Training targets may carry
    ``"masks"`` -- (M, Hm, Wm) per-instance binary masks aligned with
    ``"boxes"`` (any resolution; they are resampled to the prototype grid).
    Images without masks contribute detection losses only. ``predict`` adds
    ``"masks"``: a (K, H, W) bool tensor at input resolution per image.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: ApexHead,
        mask_head: StencilHead,
        mask_weight: float = 2.0,
        mask_threshold: float = 0.5,
        mask_sample_size: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(backbone, neck, head, **kwargs)
        if not isinstance(mask_head, StencilHead):
            got = type(mask_head).__name__
            raise ModelError("LofopSegment requires a StencilHead", context={"got": got})
        self.mask_head = mask_head
        self.mask_weight = mask_weight
        self.mask_threshold = mask_threshold
        self.mask_sample_size = mask_sample_size

    def compute_losses(self, images: Tensor, targets: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        features = self._features(images)
        losses, aux = self._detection_losses(self.head(features), images, targets)
        prototypes, coefficients = self.mask_head(features)
        coeff = flatten_levels(coefficients, self.mask_head.num_prototypes)
        proto_h, proto_w = prototypes.shape[-2:]
        scale_x = proto_w / images.shape[-1]
        scale_y = proto_h / images.shape[-2]

        total_mask = images.new_zeros(())
        instances = 0
        for image_index, (target, info) in enumerate(zip(targets, aux)):
            gt_masks = target.get("masks")
            if gt_masks is None or gt_masks.shape[0] == 0:
                continue
            pos_idx = info["positive"].nonzero(as_tuple=False).squeeze(1)
            if pos_idx.numel() == 0:
                continue
            if pos_idx.numel() > self.mask_sample_size:
                pick = torch.randperm(pos_idx.numel(), device=pos_idx.device)
                pos_idx = pos_idx[pick[: self.mask_sample_size]]
            gt_index = info["assigned"][pos_idx]
            if gt_masks.shape[-2:] != (proto_h, proto_w):
                gt_masks = F.interpolate(
                    gt_masks.unsqueeze(1).float(), size=(proto_h, proto_w), mode="nearest"
                ).squeeze(1)
            mask_targets = gt_masks[gt_index]
            logits = torch.einsum(
                "pk,khw->phw", coeff[image_index][pos_idx], prototypes[image_index]
            )
            # Supervise inside the assigned GT box only (inference crops to
            # the detected box anyway), normalized per instance by box area
            # so large objects don't dominate.
            region = _box_region(
                target["boxes"][gt_index], proto_h, proto_w, scale_x, scale_y
            )
            bce = F.binary_cross_entropy_with_logits(logits, mask_targets, reduction="none")
            per_instance = (bce * region).sum(dim=(1, 2)) / region.sum(dim=(1, 2)).clamp(min=1.0)
            total_mask = total_mask + per_instance.sum()
            instances += int(pos_idx.numel())

        losses["mask"] = self.mask_weight * total_mask / max(instances, 1)
        losses["total"] = losses["total"] + losses["mask"]
        return losses

    @torch.no_grad()
    def predict(self, images: Tensor) -> list[dict[str, Any]]:
        if self._channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        features = self._features(images)
        results = self._decode_batch(self.head(features), images)
        prototypes, coefficients = self.mask_head(features)
        coeff = flatten_levels(coefficients, self.mask_head.num_prototypes)
        height, width = images.shape[-2:]
        proto_h, proto_w = prototypes.shape[-2:]
        for image_index, result in enumerate(results):
            locations = result.pop("locations")
            if locations.numel() == 0:
                result["masks"] = torch.zeros(
                    (0, height, width), dtype=torch.bool, device=images.device
                )
                continue
            logits = torch.einsum(
                "pk,khw->phw", coeff[image_index][locations], prototypes[image_index]
            )
            masks = logits.sigmoid() * _box_region(
                result["boxes"], proto_h, proto_w, proto_w / width, proto_h / height
            )
            masks = F.interpolate(
                masks.unsqueeze(1), size=(height, width), mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            result["masks"] = masks > self.mask_threshold
        return results
