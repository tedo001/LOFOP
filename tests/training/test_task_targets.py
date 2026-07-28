"""Tests for mask/keypoint targets in the torch data bridge and SDK training."""

import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from lofop.core.exceptions import DataError  # noqa: E402
from lofop.data.dataset import BoxAnnotation, Category, Dataset, Sample  # noqa: E402
from lofop.training.torch_data import DetectionTorchDataset  # noqa: E402


def tiny_dataset(tmp_path, *, with_polygons=False, with_keypoints=False, images=2):
    root = tmp_path / "images"
    root.mkdir(exist_ok=True)
    dataset = Dataset("tiny", [Category(1, "thing")], image_root=root)
    for i in range(images):
        Image.new("RGB", (32, 32), (i * 40 % 255, 100, 50)).save(root / f"{i}.png")
        ann = BoxAnnotation(bbox=(4.0, 4.0, 24.0, 24.0), category_id=1)
        if with_polygons:
            ann.segmentation = [[4.0, 4.0, 24.0, 4.0, 24.0, 24.0, 4.0, 24.0]]
        if with_keypoints:
            ann.keypoints = [(8.0, 8.0, 2), (20.0, 20.0, 2), (0.0, 0.0, 0)]
        dataset.add_sample(Sample(image=f"{i}.png", width=32, height=32, annotations=[ann]))
    return dataset


class TestMaskTargets:
    def test_polygon_rasterized_at_quarter_resolution(self, tmp_path):
        ds = DetectionTorchDataset(
            tiny_dataset(tmp_path, with_polygons=True), image_size=64, include_masks=True
        )
        _, target = ds[0]
        masks = target["masks"]
        assert masks.shape == (1, 16, 16)
        # Polygon covers pixels 4..24 of 32 -> mask cells ~2..12 of 16.
        assert masks[0, 8, 8] == 1.0 and masks[0, 0, 0] == 0.0

    def test_box_fallback_when_no_polygons(self, tmp_path):
        ds = DetectionTorchDataset(
            tiny_dataset(tmp_path), image_size=64, include_masks=True
        )
        _, target = ds[0]
        assert target["masks"].shape == (1, 16, 16)
        assert target["masks"].sum() > 0

    def test_flip_flips_masks_with_boxes(self, tmp_path):
        ds = DetectionTorchDataset(
            tiny_dataset(tmp_path, with_polygons=True), image_size=64,
            augment=True, flip_probability=1.0, include_masks=True,
        )
        _, target = ds[0]
        # Square polygon centered symmetrically: flipped box must still match
        # the flipped mask's occupied columns.
        masks = target["masks"]
        occupied = masks[0].sum(dim=0).nonzero().squeeze(1)
        x1, x2 = target["boxes"][0][0] / 4, target["boxes"][0][2] / 4
        assert x1 - 1.5 <= occupied.min() <= x1 + 1.5
        assert x2 - 1.5 <= occupied.max() <= x2 + 1.5


class TestKeypointTargets:
    def test_scaled_to_image_size(self, tmp_path):
        ds = DetectionTorchDataset(
            tiny_dataset(tmp_path, with_keypoints=True), image_size=64,
            include_keypoints=True,
        )
        _, target = ds[0]
        kp = target["keypoints"]
        assert kp.shape == (1, 3, 3)
        assert kp[0, 0].tolist() == [16.0, 16.0, 2.0]  # 8 * (64/32)
        assert kp[0, 2, 2] == 0.0  # unlabeled keypoint preserved

    def test_flip_disabled_for_keypoints(self, tmp_path):
        ds = DetectionTorchDataset(
            tiny_dataset(tmp_path, with_keypoints=True), image_size=64,
            augment=True, flip_probability=1.0, include_keypoints=True,
        )
        _, target = ds[0]
        assert target["boxes"][0][0] == pytest.approx(8.0)  # unflipped

    def test_strong_augment_rejected_with_extras(self, tmp_path):
        with pytest.raises(DataError):
            DetectionTorchDataset(
                tiny_dataset(tmp_path), strong_augment=True, include_masks=True
            )

    def test_default_path_unchanged(self, tmp_path):
        _, target = DetectionTorchDataset(tiny_dataset(tmp_path), image_size=64)[0]
        assert sorted(target) == ["boxes", "labels"]


class TestSdkTraining:
    @pytest.mark.parametrize("variant,key", [("n-seg", "masks"), ("n-pose", "keypoints")])
    def test_train_and_predict_task_variant(self, tmp_path, variant, key):
        from lofop import Detector

        torch.manual_seed(0)
        data = tiny_dataset(
            tmp_path, with_polygons=True, with_keypoints=True, images=4
        )
        detector = Detector(
            variant, num_classes=1, image_size=64, device="cpu", num_keypoints=3
        )
        detector.train(
            train_data=data, epochs=1, batch_size=2, workers=0,
            checkpoint_dir=tmp_path / "ck",
        )
        [result] = detector.predict(
            data.image_path(data.samples[0]), score_threshold=0.0
        )
        assert getattr(result, key) is not None
