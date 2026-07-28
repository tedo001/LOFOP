"""Bridge from the canonical LOFOP dataset model to PyTorch training data.

`DetectionTorchDataset` wraps a :class:`lofop.data.Dataset`: it loads images
with Pillow, resizes to a square training resolution (aspect distortion is
accepted in v1 -- letterboxing is a planned refinement), scales boxes to
match, and optionally applies horizontal-flip augmentation. Tensors only;
no numpy dependency.
"""

from __future__ import annotations

import random

import torch
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

from lofop.core.exceptions import DataError
from lofop.data.dataset import Dataset, Sample
from lofop.training.augment import color_jitter, horizontal_flip, mosaic


def image_to_tensor(image: Image.Image) -> Tensor:
    """Convert an RGB PIL image to a float CHW tensor in [0, 1]."""
    rgb = image.convert("RGB")
    data = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
    return data.reshape(rgb.height, rgb.width, 3).permute(2, 0, 1).float() / 255.0


class DetectionTorchDataset(TorchDataset):
    """Torch-facing view of a canonical detection dataset.

    Args:
        dataset: Canonical dataset (its ``image_root`` must resolve files).
        image_size: Square side length images are resized to.
        augment: Apply random horizontal flip (training) or nothing (eval).
        flip_probability: Chance of the horizontal flip when augmenting.
        strong_augment: Additionally apply mosaic (probability
            ``mosaic_probability``, combining 4 random samples) and color
            jitter -- the richer recipe for real-data training. Off by
            default so existing runs and benchmarks are unchanged.
        mosaic_probability: Chance of building a mosaic per fetched item
            when ``strong_augment`` is enabled.
        include_masks: Add ``"masks"`` targets -- (M, S/4, S/4) binary
            instance masks rasterized from each annotation's polygons (or its
            box when polygons are absent). For ``LofopSegment`` training.
        include_keypoints: Add ``"keypoints"`` targets -- (M, K, 3)
            ``(x, y, visibility)`` in resized pixels (annotations without
            keypoints get all-zero rows). For ``LofopPose`` training.
            Disables the horizontal flip, which would need left/right
            keypoint-pair swapping.

    ``__getitem__`` returns ``(image, target)`` where ``image`` is (3, S, S)
    and ``target`` is ``{"boxes": (M, 4) xyxy in resized pixels, "labels":
    (M,) contiguous class indices}`` plus any requested extras.
    """

    def __init__(
        self,
        dataset: Dataset,
        image_size: int = 640,
        augment: bool = False,
        flip_probability: float = 0.5,
        strong_augment: bool = False,
        mosaic_probability: float = 0.5,
        include_masks: bool = False,
        include_keypoints: bool = False,
    ) -> None:
        if strong_augment and (include_masks or include_keypoints):
            raise DataError(
                "strong_augment does not support mask/keypoint targets yet; "
                "train segmentation/pose with the default augmentation"
            )
        self.dataset = dataset
        self.image_size = image_size
        self.augment = augment or strong_augment
        self.flip_probability = flip_probability
        self.strong_augment = strong_augment
        self.mosaic_probability = mosaic_probability
        self.include_masks = include_masks
        self.include_keypoints = include_keypoints
        self.class_index = dataset.category_index()

    def __len__(self) -> int:
        return len(self.dataset.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        image, boxes, labels = self._load_scaled(index)
        if self.include_masks or self.include_keypoints:
            sample = self.dataset.samples[index]
            target: dict[str, Tensor] = {"boxes": boxes, "labels": labels}
            if self.include_masks:
                target["masks"] = self._instance_masks(sample)
            if self.include_keypoints:
                target["keypoints"] = self._scaled_keypoints(sample)
            flip = (
                self.augment and not self.include_keypoints
                and random.random() < self.flip_probability
            )
            if flip:
                image, target["boxes"] = horizontal_flip(image, boxes, self.image_size)
                if self.include_masks:
                    target["masks"] = torch.flip(target["masks"], dims=[-1])
            return image, target
        if self.strong_augment:
            if random.random() < self.mosaic_probability:
                others = [random.randrange(len(self)) for _ in range(3)]
                items = [(image, boxes, labels)]
                items += [self._load_scaled(i) for i in others]
                image, boxes, labels = mosaic(items, self.image_size)
            image = color_jitter(image)
        if self.augment and random.random() < self.flip_probability:
            image, boxes = horizontal_flip(image, boxes, self.image_size)
        return image, {"boxes": boxes, "labels": labels}

    def _load_scaled(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        sample = self.dataset.samples[index]
        path = self.dataset.image_path(sample)
        try:
            with Image.open(path) as img:
                image = image_to_tensor(
                    img.resize((self.image_size, self.image_size), Image.BILINEAR)
                )
        except OSError as exc:
            raise DataError(f"Cannot load image: {exc}", context={"path": str(path)}) from exc
        boxes, labels = self._scaled_targets(sample)
        return image, boxes, labels

    def _scaled_targets(self, sample: Sample) -> tuple[Tensor, Tensor]:
        if not sample.annotations:
            return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long)
        scale_x = self.image_size / sample.width
        scale_y = self.image_size / sample.height
        boxes = torch.tensor(
            [[b.bbox[0] * scale_x, b.bbox[1] * scale_y, b.bbox[2] * scale_x, b.bbox[3] * scale_y]
             for b in sample.annotations],
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [self.class_index[b.category_id] for b in sample.annotations], dtype=torch.long
        )
        return boxes, labels

    def _instance_masks(self, sample: Sample) -> Tensor:
        # Rasterized at the model's mask (prototype) resolution, stride 4.
        mask_size = self.image_size // 4
        scale_x = mask_size / sample.width
        scale_y = mask_size / sample.height
        masks = []
        for ann in sample.annotations:
            canvas = Image.new("L", (mask_size, mask_size), 0)
            draw = ImageDraw.Draw(canvas)
            if ann.segmentation:
                for polygon in ann.segmentation:
                    points = [
                        (polygon[i] * scale_x, polygon[i + 1] * scale_y)
                        for i in range(0, len(polygon) - 1, 2)
                    ]
                    if len(points) >= 3:
                        draw.polygon(points, fill=1)
            else:  # box-only annotation: the box itself is the best mask known
                x1, y1, x2, y2 = ann.bbox
                draw.rectangle(
                    [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y], fill=1
                )
            data = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8)
            masks.append(data.reshape(mask_size, mask_size).float())
        if not masks:
            return torch.zeros((0, mask_size, mask_size))
        return torch.stack(masks)

    def _scaled_keypoints(self, sample: Sample) -> Tensor:
        counts = {len(a.keypoints) for a in sample.annotations if a.keypoints}
        if len(counts) > 1:
            raise DataError(
                "Inconsistent keypoint counts within one image",
                context={"image": sample.image, "counts": sorted(counts)},
            )
        num_kp = counts.pop() if counts else 0
        if num_kp == 0:
            return torch.zeros((len(sample.annotations), 0, 3))
        scale_x = self.image_size / sample.width
        scale_y = self.image_size / sample.height
        rows = []
        for ann in sample.annotations:
            if ann.keypoints:
                rows.append([
                    [x * scale_x, y * scale_y, float(v)] for x, y, v in ann.keypoints
                ])
            else:
                rows.append([[0.0, 0.0, 0.0]] * num_kp)
        return torch.tensor(rows, dtype=torch.float32)


def detection_collate(
    batch: list[tuple[Tensor, dict[str, Tensor]]],
) -> tuple[Tensor, list[dict[str, Tensor]]]:
    """Stack images; keep per-image target dicts (variable object counts)."""
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets
