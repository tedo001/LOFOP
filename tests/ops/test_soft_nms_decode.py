"""Tests for soft_nms and decode_dense: Python reference and native parity."""

import random
import shutil

import pytest

from lofop.core.exceptions import LofopError
from lofop.ops import decode_dense, soft_nms
from lofop.ops.native import build_native, load_native


def random_boxes(n, seed, span=1000.0):
    rng = random.Random(seed)
    boxes, scores = [], []
    for _ in range(n):
        x1 = rng.uniform(0, span)
        y1 = rng.uniform(0, span)
        boxes.append([x1, y1, x1 + rng.uniform(1, 120), y1 + rng.uniform(1, 120)])
        scores.append(rng.random())
    return boxes, scores

HAVE_COMPILER = shutil.which("g++") or shutil.which("clang++")

needs_compiler = pytest.mark.skipif(not HAVE_COMPILER, reason="no C++ compiler available")


@pytest.fixture(scope="module", autouse=True)
def fresh_native_lib():
    # The new kernels must exist in the compiled library; force a rebuild so
    # a stale pre-1.2 library on disk cannot skew the parity tests.
    if HAVE_COMPILER:
        build_native(force=True)
        assert hasattr(load_native(), "lofop_soft_nms")


class TestSoftNmsPython:
    def test_decays_instead_of_dropping(self):
        # Two heavily overlapping boxes: greedy NMS would keep one; gaussian
        # soft-NMS keeps both, the second with a decayed score.
        boxes = [[0, 0, 10, 10], [1, 1, 11, 11]]
        keep, scores = soft_nms(boxes, [0.9, 0.8], score_threshold=0.01, native=False)
        assert keep == [0, 1]
        assert scores[0] == pytest.approx(0.9)
        assert 0.0 < scores[1] < 0.8

    def test_linear_matches_greedy_below_threshold(self):
        # Disjoint boxes are untouched by either method.
        boxes = [[0, 0, 10, 10], [50, 50, 60, 60]]
        keep, scores = soft_nms(boxes, [0.9, 0.8], method="linear", native=False)
        assert keep == [0, 1]
        assert scores == pytest.approx([0.9, 0.8])

    def test_score_threshold_discards(self):
        boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
        keep, _ = soft_nms(boxes, [0.9, 0.8], score_threshold=0.5, native=False)
        assert keep == [0]  # identical box decays to ~0 and is discarded

    def test_max_keep(self):
        boxes, scores = random_boxes(50, 3)
        keep, _ = soft_nms(boxes, scores, max_keep=5, native=False)
        assert len(keep) == 5

    def test_empty_and_validation(self):
        assert soft_nms([], [], native=False) == ([], [])
        with pytest.raises(LofopError):
            soft_nms([[0, 0, 1, 1]], [], native=False)
        with pytest.raises(LofopError):
            soft_nms([[0, 0, 1, 1]], [0.5], method="nope", native=False)


@needs_compiler
class TestSoftNmsParity:
    @pytest.mark.parametrize("method", ["gaussian", "linear"])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_native_matches_python(self, method, seed):
        boxes, scores = random_boxes(200, seed)
        keep_py, scores_py = soft_nms(boxes, scores, method=method, native=False)
        keep_c, scores_c = soft_nms(boxes, scores, method=method, native=True)
        assert keep_c == keep_py
        assert scores_c == pytest.approx(scores_py, abs=1e-5)


class TestDecodeDensePython:
    def test_picks_best_class_over_threshold(self):
        scores = [[0.1, 0.9], [0.3, 0.2], [0.05, 0.04]]
        idx, labels, best = decode_dense(scores, score_threshold=0.25, native=False)
        assert idx == [0, 1]
        assert labels == [1, 0]
        assert best == pytest.approx([0.9, 0.3])

    def test_empty(self):
        assert decode_dense([], native=False) == ([], [], [])


@needs_compiler
class TestDecodeDenseParity:
    def test_native_matches_python(self):
        rng = random.Random(7)
        scores = [[rng.random() for _ in range(20)] for _ in range(500)]
        py = decode_dense(scores, score_threshold=0.9, native=False)
        c = decode_dense(scores, score_threshold=0.9, native=True)
        assert c[0] == py[0] and c[1] == py[1]
        assert c[2] == pytest.approx(py[2], abs=1e-6)

    def test_numpy_zero_copy_path(self):
        numpy = pytest.importorskip("numpy")
        scores = numpy.random.RandomState(0).rand(300, 8).astype(numpy.float32)
        py = decode_dense(scores.tolist(), score_threshold=0.85, native=False)
        c = decode_dense(scores, score_threshold=0.85, native=True)
        assert c[0] == py[0] and c[1] == py[1]
        assert c[2] == pytest.approx(py[2], abs=1e-6)
