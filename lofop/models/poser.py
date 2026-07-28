"""LofopPose: keypoint (pose) estimation on top of LOFOP-Detect.

Extends :class:`LofopDetect` with a :class:`~lofop.models.pose_head.VertexHead`
branch. Detection is inherited unchanged; each positive location additionally
regresses its instance's keypoints as stride-relative offsets, and inference
reads the skeleton off the winning location of every detection -- no
heatmaps, no second stage. Variants: ``lofop-detect-{n,s,ex}-pose`` configs.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lofop.core.exceptions import ModelError
from lofop.models.detector import LofopDetect, flatten_levels
from lofop.models.head import ApexHead
from lofop.models.pose_head import VertexHead
from lofop.registries import MODELS


@MODELS.register()
class LofopPose(LofopDetect):
    """Anchor-free pose estimation model.

    Args:
        keypoint_head: A :class:`VertexHead`.
        keypoint_weight: Weight of the keypoint loss term.

    Remaining arguments as :class:`LofopDetect`. Training targets may carry
    ``"keypoints"`` -- (M, K, 3) per-instance ``(x, y, visibility)`` rows in
    input pixels, aligned with ``"boxes"`` (COCO visibility: 0 unlabeled,
    1 labeled-occluded, 2 visible). Offsets are supervised where visibility
    is > 0; the visibility logit is supervised as "labeled". Images without
    keypoints contribute detection losses only. ``predict`` adds
    ``"keypoints"``: a (K_det, K, 3) tensor per image of ``(x, y,
    visibility probability)`` in input pixels.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: ApexHead,
        keypoint_head: VertexHead,
        keypoint_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(backbone, neck, head, **kwargs)
        if not isinstance(keypoint_head, VertexHead):
            got = type(keypoint_head).__name__
            raise ModelError("LofopPose requires a VertexHead", context={"got": got})
        self.keypoint_head = keypoint_head
        self.keypoint_weight = keypoint_weight

    def compute_losses(self, images: Tensor, targets: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        features = self._features(images)
        outputs = self.head(features)
        losses, aux = self._detection_losses(outputs, images, targets)
        points, strides = self.head.level_points(outputs[0])
        offset_maps, visibility_maps = self.keypoint_head(features)
        num_kp = self.keypoint_head.num_keypoints
        offsets = flatten_levels(offset_maps, 2 * num_kp)
        visibilities = flatten_levels(visibility_maps, num_kp)

        total_offset = images.new_zeros(())
        total_visibility = images.new_zeros(())
        visible_count = 0
        location_count = 0
        for image_index, (target, info) in enumerate(zip(targets, aux)):
            gt_kp = target.get("keypoints")
            if gt_kp is None or gt_kp.shape[0] == 0:
                continue
            pos_idx = info["positive"].nonzero(as_tuple=False).squeeze(1)
            if pos_idx.numel() == 0:
                continue
            instance_kp = gt_kp[info["assigned"][pos_idx]]  # (P, K, 3)
            labeled = instance_kp[..., 2] > 0
            pred_off = offsets[image_index][pos_idx].reshape(-1, num_kp, 2)
            target_off = (instance_kp[..., :2] - points[pos_idx].unsqueeze(1)) / strides[
                pos_idx].reshape(-1, 1, 1)
            if labeled.any():
                total_offset = total_offset + F.l1_loss(
                    pred_off[labeled], target_off[labeled], reduction="sum"
                )
                visible_count += int(labeled.sum())
            total_visibility = total_visibility + F.binary_cross_entropy_with_logits(
                visibilities[image_index][pos_idx], labeled.to(images.dtype), reduction="sum"
            )
            location_count += int(pos_idx.numel()) * num_kp

        losses["kpt"] = self.keypoint_weight * (
            total_offset / max(visible_count, 1) + total_visibility / max(location_count, 1)
        )
        losses["total"] = losses["total"] + losses["kpt"]
        return losses

    @torch.no_grad()
    def predict(self, images: Tensor) -> list[dict[str, Any]]:
        if self._channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        features = self._features(images)
        outputs = self.head(features)
        results = self._decode_batch(outputs, images)
        points, strides = self.head.level_points(outputs[0])
        offset_maps, visibility_maps = self.keypoint_head(features)
        num_kp = self.keypoint_head.num_keypoints
        offsets = flatten_levels(offset_maps, 2 * num_kp)
        visibilities = flatten_levels(visibility_maps, num_kp)
        for image_index, result in enumerate(results):
            locations = result.pop("locations")
            if locations.numel() == 0:
                result["keypoints"] = torch.zeros((0, num_kp, 3), device=images.device)
                continue
            pred_off = offsets[image_index][locations].reshape(-1, num_kp, 2)
            xy = points[locations].unsqueeze(1) + pred_off * strides[locations].reshape(-1, 1, 1)
            visibility = visibilities[image_index][locations].sigmoid().unsqueeze(-1)
            result["keypoints"] = torch.cat([xy, visibility], dim=-1)
        return results
