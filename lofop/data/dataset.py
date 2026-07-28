"""In-memory dataset model.

LOFOP normalizes every annotation format into one canonical representation:
:class:`Dataset` (categories + samples), :class:`Sample` (one image and its
annotations), and :class:`BoxAnnotation` (one axis-aligned box). Format
adapters translate to and from this model, so N formats need N adapters
instead of N^2 converters, and validators/statistics/loaders are written once.

Conventions:

* Boxes are ``(x1, y1, x2, y2)`` in **absolute pixel** coordinates. Absolute
  xyxy is the least ambiguous interchange form; normalized formats (YOLO)
  convert at the adapter boundary where image sizes are known.
* Category ids are arbitrary ints unique within a dataset (COCO ids survive a
  round-trip); adapters that need contiguous indices (YOLO) map at the edge.
* ``Sample.image`` is a path relative to ``Dataset.image_root`` when that is
  set, otherwise a path usable as-is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lofop.core.exceptions import DataError


@dataclass(frozen=True)
class Category:
    """One object category.

    Attributes:
        id: Dataset-unique integer id.
        name: Human-readable class name.
    """

    id: int
    name: str


@dataclass
class BoxAnnotation:
    """One axis-aligned bounding box, optionally with a mask and keypoints.

    Attributes:
        bbox: ``(x1, y1, x2, y2)`` in absolute pixels.
        category_id: Id of the category this box belongs to.
        attributes: Format-specific extras (e.g. COCO ``iscrowd``) preserved
            through conversions when the target format understands them.
        segmentation: Optional instance mask as polygons -- each polygon a
            flat ``[x1, y1, x2, y2, ...]`` list in absolute pixels (COCO
            polygon convention). ``None`` for box-only annotations.
        keypoints: Optional ``[(x, y, visibility), ...]`` in absolute pixels;
            visibility follows COCO (0 = not labeled, 1 = labeled but not
            visible, 2 = visible). ``None`` for annotations without keypoints.
    """

    bbox: tuple[float, float, float, float]
    category_id: int
    attributes: dict[str, Any] = field(default_factory=dict)
    segmentation: list[list[float]] | None = None
    keypoints: list[tuple[float, float, int]] | None = None

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)


@dataclass
class Sample:
    """One image and its annotations.

    Attributes:
        image: Image path, relative to the dataset's ``image_root`` when set.
        width: Image width in pixels.
        height: Image height in pixels.
        annotations: Boxes on this image.
        id: Stable sample id (defaults assigned by the dataset when omitted).
    """

    image: str
    width: int
    height: int
    annotations: list[BoxAnnotation] = field(default_factory=list)
    id: int = -1


class Dataset:
    """A named collection of categories and samples.

    Args:
        name: Dataset name (used in exports and reports).
        categories: Object categories; ids and names must be unique.
        samples: Samples; ids are assigned sequentially where missing.
        image_root: Directory that ``Sample.image`` paths are relative to,
            when known. Enables image existence checks and size reading.
    """

    def __init__(
        self,
        name: str,
        categories: list[Category],
        samples: list[Sample] | None = None,
        *,
        image_root: Path | None = None,
    ) -> None:
        ids = [c.id for c in categories]
        names = [c.name for c in categories]
        if len(set(ids)) != len(ids) or len(set(names)) != len(names):
            raise DataError(
                "Category ids and names must be unique",
                context={"dataset": name, "ids": ids, "names": names},
            )
        self.name = name
        self.categories = list(categories)
        self.image_root = Path(image_root) if image_root is not None else None
        self.samples: list[Sample] = []
        for sample in samples or []:
            self.add_sample(sample)

    def add_sample(self, sample: Sample) -> Sample:
        """Append a sample, assigning the next sequential id when unset."""
        if sample.id < 0:
            sample.id = self.samples[-1].id + 1 if self.samples else 0
        self.samples.append(sample)
        return sample

    def category(self, category_id: int) -> Category:
        """Return the category with ``category_id``.

        Raises:
            DataError: If no such category exists.
        """
        for cat in self.categories:
            if cat.id == category_id:
                return cat
        raise DataError(
            "Unknown category id",
            context={"dataset": self.name, "category_id": category_id},
        )

    def category_index(self) -> dict[int, int]:
        """Map category id -> contiguous index in ``categories`` order."""
        return {cat.id: idx for idx, cat in enumerate(self.categories)}

    def image_path(self, sample: Sample) -> Path:
        """Absolute-ish path to a sample's image, honoring ``image_root``."""
        path = Path(sample.image)
        if self.image_root is not None and not path.is_absolute():
            return self.image_root / path
        return path

    @property
    def num_annotations(self) -> int:
        """Total box count across all samples."""
        return sum(len(s.annotations) for s in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        return (
            f"Dataset(name={self.name!r}, categories={len(self.categories)}, "
            f"samples={len(self.samples)}, annotations={self.num_annotations})"
        )
