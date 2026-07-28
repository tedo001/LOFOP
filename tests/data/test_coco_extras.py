"""COCO adapter round-trip of segmentation polygons and keypoints."""

import json

from lofop.data import load_dataset, save_dataset


def _payload():
    return {
        "images": [{"id": 1, "file_name": "a.png", "width": 64, "height": 64}],
        "annotations": [
            {
                "id": 1, "image_id": 1, "category_id": 7,
                "bbox": [4, 4, 20, 20], "area": 400, "iscrowd": 0,
                "segmentation": [[4, 4, 24, 4, 24, 24]],
                "keypoints": [8, 8, 2, 0, 0, 0],
                "num_keypoints": 1,
            },
            {
                "id": 2, "image_id": 1, "category_id": 7,
                "bbox": [30, 30, 10, 10], "area": 100, "iscrowd": 0,
                "segmentation": {"counts": "rle-not-supported", "size": [64, 64]},
            },
        ],
        "categories": [{"id": 7, "name": "thing"}],
    }


class TestCocoExtras:
    def test_polygons_and_keypoints_roundtrip(self, tmp_path):
        source = tmp_path / "in.json"
        source.write_text(json.dumps(_payload()), encoding="utf-8")
        dataset = load_dataset("coco", source)
        first, second = dataset.samples[0].annotations
        assert first.segmentation == [[4.0, 4.0, 24.0, 4.0, 24.0, 24.0]]
        assert first.keypoints == [(8.0, 8.0, 2), (0.0, 0.0, 0)]
        assert second.segmentation is None  # RLE masks are not parsed
        assert second.keypoints is None

        target = tmp_path / "out.json"
        save_dataset(dataset, "coco", target)
        saved = json.loads(target.read_text(encoding="utf-8"))
        ann = saved["annotations"][0]
        assert ann["segmentation"] == [[4.0, 4.0, 24.0, 4.0, 24.0, 24.0]]
        assert ann["keypoints"] == [8.0, 8.0, 2, 0.0, 0.0, 0]
        assert ann["num_keypoints"] == 1
        assert "segmentation" not in saved["annotations"][1]
