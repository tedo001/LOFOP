"""LOFOP command-line interface.

Built on argparse rather than a third-party CLI framework so the ``lofop``
command works in every environment the core does (containers, edge devices,
CI) with zero extra dependencies. Commands map one-to-one onto public SDK
functions -- the CLI parses arguments and prints; logic stays in the library.

Current commands::

    lofop version
    lofop dataset convert  --from coco --source ann.json --to yolo --target out/
    lofop dataset validate --format yolo --source dataset_root/
    lofop dataset stats    --format coco --source ann.json [-o stats.md]
    lofop dataset show     --format coco --source ann.json -o vis/ [--limit 10]
    lofop train            --config configs/train_shapes.yaml
    lofop benchmark        --config lofop/configs/lofop-detect/n.yaml [...] [-o table.md]
    lofop predict          --config s --checkpoint best.pt --source a.jpg b.jpg
    lofop evaluate         --config s --checkpoint best.pt --format coco --source val.json
    lofop export           --config lofop/configs/lofop-detect/s.yaml -o model.onnx
    lofop doctor
    lofop runs {list, show, compare} [--root runs/registry]

``train``, ``benchmark``, ``predict``, and ``evaluate`` need the
``lofop[models]`` extra (PyTorch); torch imports happen inside those handlers
so every other command (including ``doctor``) works without it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lofop.core.exceptions import LofopError
from lofop.core.logging import configure_logging
from lofop.data import compute_stats, convert_dataset, load_dataset, validate_dataset
from lofop.version import __version__


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="lofop", description="LOFOP computer vision framework")
    parser.add_argument("--log-level", default=None, help="log level (default: env or INFO)")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("version", help="print the LOFOP version")

    dataset = commands.add_parser("dataset", help="dataset tools")
    actions = dataset.add_subparsers(dest="action", required=True)

    convert = actions.add_parser("convert", help="convert between annotation formats")
    convert.add_argument("--from", dest="source_format", required=True, help="source format name")
    convert.add_argument("--source", required=True, help="source file or directory")
    convert.add_argument("--to", dest="target_format", required=True, help="target format name")
    convert.add_argument("--target", required=True, help="target file or directory")
    convert.add_argument("--image-root", default=None, help="image directory (COCO sources)")

    validate = actions.add_parser("validate", help="validate a dataset")
    validate.add_argument("--format", dest="format_name", required=True, help="format name")
    validate.add_argument("--source", required=True, help="source file or directory")
    validate.add_argument("--image-root", default=None, help="image directory (COCO sources)")
    validate.add_argument(
        "--no-check-images", action="store_true", help="skip image file existence checks"
    )

    stats = actions.add_parser("stats", help="compute dataset statistics")
    stats.add_argument("--format", dest="format_name", required=True, help="format name")
    stats.add_argument("--source", required=True, help="source file or directory")
    stats.add_argument("--image-root", default=None, help="image directory (COCO sources)")
    stats.add_argument("-o", "--output", type=Path, default=None, help="write markdown report here")
    stats.add_argument("--json", action="store_true", help="print JSON instead of markdown")

    show = actions.add_parser("show", help="draw ground-truth boxes onto dataset images")
    show.add_argument("--format", dest="format_name", required=True, help="format name")
    show.add_argument("--source", required=True, help="source file or directory")
    show.add_argument("--image-root", default=None, help="image directory (COCO sources)")
    show.add_argument(
        "-o", "--output", type=Path, required=True, help="directory to write rendered images",
    )
    show.add_argument("--limit", type=int, default=None, help="render at most this many samples")
    show.add_argument("--width", type=int, default=3, help="box outline thickness in pixels")

    train = commands.add_parser("train", help="train a detector from a training config")
    train.add_argument("--config", required=True, help="training config YAML")
    train.add_argument("--resume", action="store_true", help="resume from last.pt")

    bench = commands.add_parser("benchmark", help="measure the LOFOP metric table for models")
    bench.add_argument(
        "--config", action="append", required=True,
        help="model config YAML; repeat for multiple columns",
    )
    bench.add_argument("--size", type=int, default=640, help="benchmark image resolution")
    bench.add_argument("--checkpoint", default=None, help="weights (best.pt/last.pt) to load")
    bench.add_argument("-o", "--output", type=Path, default=None, help="write the table here")
    bench.add_argument(
        "--results-dir", type=Path, default=None,
        help="write results.md, results.csv, and results.json to this directory",
    )

    predict = commands.add_parser("predict", help="run detection on one or more images")
    predict.add_argument("--config", required=True, help="model config YAML or variant name")
    predict.add_argument("--checkpoint", default=None, help="weights (best.pt/last.pt) to load")
    predict.add_argument("--num-classes", type=int, default=80, help="number of object classes")
    predict.add_argument("--source", nargs="+", required=True, help="image path(s) to run on")
    predict.add_argument("--size", type=int, default=640, help="inference resolution")
    predict.add_argument(
        "--score-threshold", type=float, default=None, help="confidence cut for this run",
    )
    predict.add_argument("--json", action="store_true", help="print JSON instead of text")
    predict.add_argument("-o", "--output", type=Path, default=None, help="write JSON results here")

    evaluate = commands.add_parser("evaluate", help="evaluate a detector on a dataset")
    evaluate.add_argument("--config", required=True, help="model config YAML or variant name")
    evaluate.add_argument("--checkpoint", default=None, help="weights (best.pt/last.pt) to load")
    evaluate.add_argument("--num-classes", type=int, default=80, help="number of object classes")
    evaluate.add_argument("--format", dest="format_name", required=True, help="dataset format name")
    evaluate.add_argument("--source", required=True, help="dataset file or directory")
    evaluate.add_argument("--image-root", default=None, help="image directory (COCO sources)")
    evaluate.add_argument("--size", type=int, default=640, help="evaluation resolution")
    evaluate.add_argument("--batch-size", type=int, default=8, help="evaluation batch size")
    evaluate.add_argument("--json", action="store_true", help="print JSON instead of text")
    evaluate.add_argument("-o", "--output", type=Path, default=None, help="write JSON metrics here")

    runs = commands.add_parser("runs", help="list and compare tracked training runs")
    runs_actions = runs.add_subparsers(dest="action", required=True)
    runs_list = runs_actions.add_parser("list", help="list runs in a registry")
    runs_list.add_argument("--root", type=Path, default=Path("runs/registry"))
    runs_show = runs_actions.add_parser("show", help="show one run's record and history")
    runs_show.add_argument("run_id")
    runs_show.add_argument("--root", type=Path, default=Path("runs/registry"))
    runs_compare = runs_actions.add_parser("compare", help="side-by-side metric table")
    runs_compare.add_argument("run_ids", nargs="+")
    runs_compare.add_argument("--root", type=Path, default=Path("runs/registry"))

    commands.add_parser("doctor", help="report the LOFOP environment and available backends")

    export = commands.add_parser("export", help="export a detector to ONNX or TensorRT")
    export.add_argument("--config", required=True, help="model config YAML")
    export.add_argument("--checkpoint", default=None, help="weights (best.pt/last.pt) to load")
    export.add_argument(
        "--format", choices=["onnx", "tensorrt"], default="onnx", help="export format",
    )
    export.add_argument("--size", type=int, default=640, help="input resolution for the graph")
    export.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    export.add_argument(
        "--dynamic", action="store_true",
        help="ONNX: symbolic batch/height/width axes for variable input sizes",
    )
    export.add_argument("--no-verify", action="store_true", help="skip onnxruntime verification")
    export.add_argument("--fp16", action="store_true", help="TensorRT: enable FP16 kernels")
    export.add_argument("-o", "--output", type=Path, required=True, help="output path")
    return parser


def _load_kwargs(args: argparse.Namespace) -> dict:
    return {"image_root": args.image_root} if getattr(args, "image_root", None) else {}


def _cmd_convert(args: argparse.Namespace) -> int:
    dataset = convert_dataset(
        args.source_format, args.source, args.target_format, args.target, **_load_kwargs(args)
    )
    print(
        f"Converted {len(dataset)} samples / {dataset.num_annotations} annotations: "
        f"{args.source_format} -> {args.target_format} ({args.target})"
    )
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.format_name, args.source, **_load_kwargs(args))
    report = validate_dataset(dataset, check_images=not args.no_check_images)
    for issue in report.issues:
        print(issue)
    print(report.summary())
    return 0 if report.ok else 1


def _cmd_stats(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.format_name, args.source, **_load_kwargs(args))
    result = compute_stats(dataset)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.to_markdown())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result.to_markdown(), encoding="utf-8")
        print(f"Report written to {args.output}", file=sys.stderr)
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    from lofop.data import visualize_dataset

    dataset = load_dataset(args.format_name, args.source, **_load_kwargs(args))
    paths = visualize_dataset(dataset, args.output, limit=args.limit, width=args.width)
    print(f"Rendered {len(paths)} image(s) to {args.output}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    import lofop.models  # noqa: F401  (registers model components)
    from lofop.core.config import Config
    from lofop.registries import HUB
    from lofop.training import DetectionTorchDataset, Trainer

    cfg = Config.load(args.config)
    model = HUB.build(cfg.model)
    data = cfg.data
    image_size = data.get("image_size", 640)
    load_kwargs = {"image_root": data.image_root} if "image_root" in data else {}
    train_ds = DetectionTorchDataset(
        load_dataset(data.format, data.train_source, **load_kwargs),
        image_size=image_size, augment=True,
        strong_augment=data.get("strong_augment", False),
        include_masks=hasattr(model, "mask_head"),
        include_keypoints=hasattr(model, "keypoint_head"),
    )
    val_ds = None
    if "val_source" in data:
        val_ds = DetectionTorchDataset(
            load_dataset(data.format, data.val_source, **load_kwargs), image_size=image_size,
        )
    trainer = Trainer(model, train_ds, val_ds, **cfg.get("training", Config()).to_dict())
    if args.resume:
        trainer.resume()
    metrics = trainer.fit()
    if metrics is not None:
        print(f"final: mAP50 {metrics.map50:.4f}, mAP50-95 {metrics.map50_95:.4f}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    import torch

    import lofop.models  # noqa: F401  (registers model components)
    from lofop.core.config import Config
    from lofop.registries import HUB
    from lofop.utils import benchmark_model, render_table, write_reports

    reports = []
    for config_path in args.config:
        cfg = Config.load(config_path)
        model = HUB.build(cfg.model)
        if args.checkpoint:
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            state = payload.get("ema", {}).get("module", payload.get("model", payload))
            model.load_state_dict(state)
        reports.append(benchmark_model(model, Path(config_path).stem, image_size=args.size))
    table = render_table(reports)
    print(table)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table, encoding="utf-8")
    if args.results_dir:
        written = write_reports(reports, args.results_dir)
        paths = ", ".join(str(path) for path in written.values())
        print(f"Wrote {paths}", file=sys.stderr)
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    import torch

    import lofop.models  # noqa: F401  (registers model components)
    from lofop.core.config import Config
    from lofop.registries import HUB

    cfg = Config.load(args.config)
    model = HUB.build(cfg.model)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        state = payload.get("ema", {}).get("module", payload.get("model", payload))
        model.load_state_dict(state)

    if args.format == "tensorrt":
        from lofop.deploy import export_tensorrt

        path = export_tensorrt(model, args.output, image_size=args.size, fp16=args.fp16)
        size_mb = path.stat().st_size / 1e6
        print(f"Exported TensorRT engine {path} ({size_mb:.1f} MB, fp16={args.fp16})")
        return 0

    from lofop.deploy import export_onnx

    path = export_onnx(
        model, args.output, image_size=args.size, opset=args.opset,
        dynamic=args.dynamic, verify=not args.no_verify,
    )
    print(
        f"Exported {path} ({path.stat().st_size / 1e6:.1f} MB, "
        f"dynamic={args.dynamic}, verified={not args.no_verify})"
    )
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from lofop import Detector

    detector = Detector(
        args.config, num_classes=args.num_classes, checkpoint=args.checkpoint,
        image_size=args.size,
    )
    results = detector.predict(args.source, score_threshold=args.score_threshold)
    payload = []
    for path, detections in zip(args.source, results):
        payload.append({
            "image": str(path),
            "boxes": detections.boxes,
            "scores": detections.scores,
            "labels": detections.labels,
        })
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for item in payload:
            print(f"{item['image']}: {len(item['boxes'])} detection(s)")
            for box, score, label in zip(item["boxes"], item["scores"], item["labels"]):
                name = detector.class_names[label]
                coords = ", ".join(f"{value:.1f}" for value in box)
                print(f"  {name} ({score:.3f}) [{coords}]")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Results written to {args.output}", file=sys.stderr)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from lofop import Detector

    detector = Detector(
        args.config, num_classes=args.num_classes, checkpoint=args.checkpoint,
        image_size=args.size,
    )
    dataset = load_dataset(args.format_name, args.source, **_load_kwargs(args))
    metrics = detector.evaluate(dataset, batch_size=args.batch_size)
    result = {
        "map50": metrics.map50,
        "map50_95": metrics.map50_95,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "per_class_ap50": metrics.per_class_ap50,
        "per_class_precision": metrics.per_class_precision,
        "per_class_recall": metrics.per_class_recall,
        "confusion_classes": metrics.confusion_classes,
        "confusion_matrix": metrics.confusion_matrix,
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"mAP@50    : {metrics.map50:.4f}")
        print(f"mAP@50:95 : {metrics.map50_95:.4f}")
        print(f"precision : {metrics.precision:.4f}")
        print(f"recall    : {metrics.recall:.4f}")
        print(f"F1        : {metrics.f1:.4f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Metrics written to {args.output}", file=sys.stderr)
    return 0


def _cmd_runs(args: argparse.Namespace) -> int:
    from lofop.mlops import compare_runs, list_runs, load_run

    if args.action == "list":
        records = list_runs(args.root)
        if not records:
            print(f"No runs under {args.root}")
            return 0
        print(f"{'RUN ID':<22} {'STATUS':<10} {'EPOCHS':<8} {'BEST mAP50':<11} NAME")
        for r in records:
            epochs = f"{r.get('epochs_completed', 0)}/{r.get('epochs_planned') or '?'}"
            best = r.get("best_map50")
            best_str = f"{best:.4f}" if isinstance(best, float) else "-"
            print(f"{r['run_id']:<22} {r.get('status', '?'):<10} {epochs:<8} "
                  f"{best_str:<11} {r.get('name', '')}")
        return 0
    if args.action == "show":
        record, history = load_run(args.root, args.run_id)
        print(json.dumps(record, indent=2))
        if history:
            last = history[-1]
            print(f"history: {len(history)} epochs recorded (last: {json.dumps(last)})")
        return 0
    print(compare_runs(args.root, args.run_ids))
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    import importlib.util
    import platform

    from lofop.ops.native import backend, find_library

    print(f"LOFOP {__version__}")
    print(f"Python {platform.python_version()} ({platform.system()} {platform.machine()})")
    library = find_library()
    print(f"native ops: {backend()}" + (f" ({library})" if library else " (not built)"))
    optional = ["torch", "onnx", "onnxruntime", "tensorrt", "PIL", "yaml", "rich"]
    print("optional dependencies:")
    for name in optional:
        found = importlib.util.find_spec(name) is not None
        print(f"  {name:<12}: {'available' if found else 'missing'}")
    if importlib.util.find_spec("torch") is not None:
        import torch

        cuda = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda else "cpu only"
        print(f"torch {torch.__version__}: CUDA {'yes' if cuda else 'no'} ({device})")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(level=args.log_level)
    if args.command == "version":
        print(__version__)
        return 0
    try:
        if args.command == "train":
            return _cmd_train(args)
        if args.command == "benchmark":
            return _cmd_benchmark(args)
        if args.command == "export":
            return _cmd_export(args)
        if args.command == "predict":
            return _cmd_predict(args)
        if args.command == "evaluate":
            return _cmd_evaluate(args)
        if args.command == "doctor":
            return _cmd_doctor(args)
        if args.command == "runs":
            return _cmd_runs(args)
        handlers = {
            "convert": _cmd_convert, "validate": _cmd_validate,
            "stats": _cmd_stats, "show": _cmd_show,
        }
        return handlers[args.action](args)
    except LofopError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
