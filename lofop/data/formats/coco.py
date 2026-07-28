"""COCO object-detection format adapter.

Layout: a single JSON file (``instances_*.json`` style) with ``images``,
``annotations``, and ``categories`` arrays. COCO boxes are ``[x, y, w, h]``
absolute pixels and convert losslessly to the canonical xyxy form. COCO
category ids, image ids, and ``iscrowd`` flags survive a round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path

from lofop.core.exceptions import DataError
from lofop.core.logging import get_logger
from lofop.data.dataset import BoxAnnotation, Category, Dataset, Sample
from lofop.data.formats.base import FORMATS, DatasetAdapter

logger = get_logger(__name__)


@FORMATS.register(name="coco")
class CocoAdapter(DatasetAdapter):
    """Adapter for COCO detection JSON files.

    ``source`` is the annotation JSON file. ``target`` is the JSON file to
    write. Pass ``image_root`` to :meth:`load` to enable image file checks
    downstream.
    """

    def load(
        self,
        source: str | Path,
        *,
        name: str | None = None,
        image_root: str | Path | None = None,
    ) -> Dataset:
        path = Path(source)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise DataError(f"Cannot read COCO file: {exc}", context={"source": str(path)}) from exc
        except json.JSONDecodeError as exc:
            raise DataError(f"Invalid COCO JSON: {exc}", context={"source": str(path)}) from exc
        for key in ("images", "annotations", "categories"):
            if not isinstance(payload.get(key), list):
                raise DataError(
                    f"COCO file is missing the {key!r} array", context={"source": str(path)}
                )

        categories = [Category(id=int(c["id"]), name=str(c["name"])) for c in payload["categories"]]
        dataset = Dataset(
            name=name or path.stem,
            categories=categories,
            image_root=Path(image_root) if image_root is not None else None,
        )
        samples: dict[int, Sample] = {}
        for image in payload["images"]:
            sample = Sample(
                image=str(image["file_name"]),
                width=int(image["width"]),
                height=int(image["height"]),
                id=int(image["id"]),
            )
            samples[sample.id] = sample
            dataset.add_sample(sample)

        known_categories = {c.id for c in categories}
        skipped = 0
        for ann in payload["annotations"]:
            image_id = int(ann["image_id"])
            sample = samples.get(image_id)
            if sample is None:
                skipped += 1
                continue
            x, y, w, h = (float(v) for v in ann["bbox"])
            attributes = {}
            if ann.get("iscrowd"):
                attributes["iscrowd"] = int(ann["iscrowd"])
            category_id = int(ann["category_id"])
            if category_id not in known_categories:
                raise DataError(
                    "Annotation references unknown category",
                    context={"source": str(path), "category_id": category_id, "image_id": image_id},
                )
            segmentation = None
            raw_seg = ann.get("segmentation")
            if isinstance(raw_seg, list) and raw_seg and isinstance(raw_seg[0], list):
                segmentation = [[float(v) for v in poly] for poly in raw_seg]
            keypoints = None
            raw_kp = ann.get("keypoints")
            if isinstance(raw_kp, list) and raw_kp and len(raw_kp) % 3 == 0:
                keypoints = [
                    (float(raw_kp[i]), float(raw_kp[i + 1]), int(raw_kp[i + 2]))
                    for i in range(0, len(raw_kp), 3)
                ]
            sample.annotations.append(
                BoxAnnotation(bbox=(x, y, x + w, y + h), category_id=category_id,
                              attributes=attributes, segmentation=segmentation,
                              keypoints=keypoints)
            )
        if skipped:
            logger.warning("Skipped %d annotations referencing unknown images", skipped)
        return dataset

    def save(self, dataset: Dataset, target: str | Path) -> None:
        path = Path(target)
        images = [
            {"id": s.id, "file_name": s.image, "width": s.width, "height": s.height}
            for s in dataset.samples
        ]
        annotations = []
        ann_id = 1
        for sample in dataset.samples:
            for box in sample.annotations:
                x1, y1, x2, y2 = box.bbox
                record = {
                    "id": ann_id,
                    "image_id": sample.id,
                    "category_id": box.category_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": box.area,
                    "iscrowd": int(box.attributes.get("iscrowd", 0)),
                }
                if box.segmentation is not None:
                    record["segmentation"] = box.segmentation
                if box.keypoints is not None:
                    record["keypoints"] = [v for kp in box.keypoints for v in kp]
                    record["num_keypoints"] = sum(1 for kp in box.keypoints if kp[2] > 0)
                annotations.append(record)
                ann_id += 1
        payload = {
            "info": {"description": dataset.name},
            "images": images,
            "annotations": annotations,
            "categories": [{"id": c.id, "name": c.name} for c in dataset.categories],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        logger.info("Wrote COCO dataset %r to %s", dataset.name, path)
