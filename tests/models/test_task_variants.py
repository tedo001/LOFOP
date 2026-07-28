"""Tests for the segmentation (LofopSegment) and pose (LofopPose) variants."""

import pytest

torch = pytest.importorskip("torch")

import lofop.models  # noqa: E402, F401  (registers model components)
from lofop.core.config import Config  # noqa: E402
from lofop.core.exceptions import ModelError  # noqa: E402
from lofop.models import LofopPose, LofopSegment  # noqa: E402
from lofop.registries import HUB  # noqa: E402

CONFIG_DIR = "lofop/configs/lofop-detect"


def build_variant(name, num_classes=2, **overrides):
    cfg = Config.load(f"{CONFIG_DIR}/{name}.yaml", resolve=False)
    cfg.num_classes = num_classes
    for key, value in overrides.items():
        cfg[key] = value
    cfg.resolve()
    return HUB.build(cfg.model)


def targets_with_extras(with_masks=False, with_keypoints=False, num_kp=17):
    full = {"boxes": torch.tensor([[8.0, 8.0, 40.0, 40.0]]), "labels": torch.tensor([0])}
    if with_masks:
        mask = torch.zeros(1, 16, 16)
        mask[0, 2:10, 2:10] = 1.0
        full["masks"] = mask
    if with_keypoints:
        kp = torch.rand(1, num_kp, 3) * 30
        kp[..., 2] = 2.0
        full["keypoints"] = kp
    empty = {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}
    return [full, empty]


class TestSegment:
    def test_all_variants_build(self):
        for name in ("n-seg", "s-seg", "ex-seg"):
            model = build_variant(name)
            assert isinstance(model, LofopSegment)

    def test_losses_include_mask_and_backward(self):
        model = build_variant("n-seg")
        losses = model.compute_losses(
            torch.rand(2, 3, 64, 64), targets_with_extras(with_masks=True)
        )
        assert set(losses) == {"cls", "box", "quality", "mask", "total"}
        assert losses["mask"].item() > 0
        losses["total"].backward()
        grads = [p.grad for p in model.mask_head.parameters() if p.grad is not None]
        assert grads  # the mask branch actually trains

    def test_box_only_targets_still_train(self):
        # Datasets without polygons must not break segmentation training.
        model = build_variant("n-seg")
        losses = model.compute_losses(torch.rand(2, 3, 64, 64), targets_with_extras())
        assert losses["mask"].item() == 0.0
        losses["total"].backward()

    def test_predict_attaches_masks(self):
        model = build_variant("n-seg").eval()
        model.score_threshold = 0.0
        [result] = model.predict(torch.rand(1, 3, 64, 64))
        assert result["masks"].dtype == torch.bool
        assert result["masks"].shape == (result["boxes"].shape[0], 64, 64)

    def test_wrong_head_type_rejected(self):
        det = build_variant("n")
        with pytest.raises(ModelError):
            LofopSegment(det.backbone, det.neck, det.head, mask_head=det.head)


class TestPose:
    def test_all_variants_build(self):
        for name in ("n-pose", "s-pose", "ex-pose"):
            model = build_variant(name)
            assert isinstance(model, LofopPose)
            assert model.keypoint_head.num_keypoints == 17

    def test_num_keypoints_override(self):
        model = build_variant("n-pose", num_keypoints=5)
        assert model.keypoint_head.num_keypoints == 5

    def test_losses_include_kpt_and_backward(self):
        model = build_variant("n-pose")
        losses = model.compute_losses(
            torch.rand(2, 3, 64, 64), targets_with_extras(with_keypoints=True)
        )
        assert set(losses) == {"cls", "box", "quality", "kpt", "total"}
        assert losses["kpt"].item() > 0
        losses["total"].backward()
        grads = [p.grad for p in model.keypoint_head.parameters() if p.grad is not None]
        assert grads

    def test_predict_attaches_keypoints(self):
        model = build_variant("n-pose").eval()
        model.score_threshold = 0.0
        [result] = model.predict(torch.rand(1, 3, 64, 64))
        kp = result["keypoints"]
        assert kp.shape == (result["boxes"].shape[0], 17, 3)
        assert ((kp[..., 2] >= 0) & (kp[..., 2] <= 1)).all()  # visibility prob


class TestDetectionUnchanged:
    def test_base_predict_has_no_extra_keys(self):
        model = build_variant("n").eval()
        model.score_threshold = 0.0
        [result] = model.predict(torch.rand(1, 3, 64, 64))
        assert sorted(result) == ["boxes", "labels", "scores"]
