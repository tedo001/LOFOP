# LOFOP Architecture

This document is the top-level map of the framework: the subsystems, the contracts between them,
and the build order. Module-level documents (e.g. [`core-engine.md`](core-engine.md)) go deeper.

## Principles

1. **The core is task-agnostic.** `lofop.core` knows nothing about vision. It provides the
   mechanisms every subsystem shares -- registries, configuration, events, plugins, logging,
   exceptions -- and stays importable in the leanest environments (an edge device running
   inference should not import training code).
2. **Components over inheritance.** Capabilities are added by registering components into named
   groups, not by subclassing framework internals. A new task (say, pose estimation) is a set of
   registrations plus configs -- no core changes.
3. **Configs are data.** YAML in, YAML out. Anything executable belongs in a registered component.
   This keeps experiments diffable, auditable, and safe to ship through CI/CD systems that treat
   configuration as artifacts.
4. **Explicit wiring beats magic.** Cross-group component references are visibly qualified
   (`type: loss/FocalCost`); nested dicts are never instantiated by surprise; event topics are
   plain strings owned by their emitters.
5. **Isolation is first-class.** The default hub/bus/plugin-manager are module-level conveniences;
   every piece of core machinery can be instantiated privately (tests, embedded apps, multi-tenant
   servers).

## Subsystem map and build order

| Phase | Package | Contents | Status |
|-------|---------|----------|--------|
| 1 | `lofop.core` | Registry, Config, EventBus, PluginManager, logging, exceptions | **Done** |
| 1 | `lofop.registries` | Default hub + standard component groups | **Done** |
| 2 | `lofop.data` | Canonical dataset model, COCO/YOLO/VOC adapters, converter, validator, statistics | **Done** |
| 2b | `lofop.data` | Torch data bridge + augmentation (flip; opt-in mosaic + color jitter); letterboxing/caching planned | **Done** (letterbox planned) |
| 3 | `lofop.models` | LOFOP-Detect: RidgeNet, DeltaFusion, ApexHead, losses, dynamic assignment | **Done** (see docs/lofop-detect.md) |
| 3b | `lofop.models` | Task variants on the same detector: LofopSegment (StencilHead prototype masks), LofopPose (VertexHead keypoints) | **Done** (ONNX export planned) |
| 4 | `lofop.training` | Trainer (AMP, EMA, cosine schedule, resume), COCO-protocol evaluator, checkpoints, torch data bridge | **Done** (DDP path present, not CI-exercised) |
| 5 | `lofop.inference` | Image/video/stream predictors, batching, RTSP/webcam sources | Planned |
| 6 | `lofop.deploy` | ONNX (fixed + dynamic shapes, verified) + TensorRT export, torch-free post-processing; OpenVINO/REST planned | **Done** (OpenVINO/REST planned) |
| 7 | `lofop.cli` | `lofop` CLI: dataset (convert/validate/stats/show), train, benchmark, predict, evaluate, export, doctor | **Done** |
| 8 | `lofop.utils` | Model benchmarking: metric table, FLOPs, FPS, size, CSV/JSON export; dataset visualization lives in `lofop.data` | **Done** |
| -- | CI / packaging | GitHub Actions CI (lint + tests + build) and PyPI/AUR packaging | **Done** |
| -- | `lofop.sdk` | High-level Python SDK: the `Detector` class (build/train/predict/export) | **Done** (see docs/sdk.md) |
| -- | `lofop.ops` | Three-tier native box ops (IoU, NMS, Soft-NMS, dense decode): Python fallback, C++ fast path (MSVC/MinGW/g++/clang), optional CUDA tier (nvcc) | **Done** |

Each phase lands with its own tests and docs before the next begins. Dependencies point downward
only: `data`/`models`/`training` depend on `core`, never on each other's internals; interaction
happens through registries, configs, and events.

## The core contracts

### Registries (`lofop.core.registry`)

- A `Registry` is a flat, string-keyed namespace of component factories for one group
  (`backbone`, `loss`, ...). A `RegistryHub` owns the groups.
- `Registry.build(spec)` instantiates `spec["type"]` with the remaining keys as kwargs.
- **Qualified names** (`"group/Name"`) resolve across groups through the hub, and nested specs
  are recursively built *only* when qualified. Unqualified nested mappings pass through as plain
  data. This rule is what keeps config semantics predictable: you can always tell from the YAML
  alone what will be instantiated.
- Trade-off vs. alternatives: a single global namespace collides at scale; deeply nested
  hierarchical scopes with inheritance make lookup rules hard to predict. Flat
  groups + explicit qualification is the middle: one extra token in configs buys unambiguous
  resolution.

### Configuration (`lofop.core.config`)

- `Config` wraps nested mappings with attribute access, dotted `select()`/`update_path()`,
  deep `merge()`, and recursive `freeze()`.
- `Config.load()` applies `extends:` inheritance (parents deep-merged in order, child on top) and
  then interpolation: `${a.b.c}` references config values (type-preserving when the reference is
  the whole string), `${env:VAR:default}` reads the environment.
- Trade-off: Python-file configs are more expressive but not auditable as data; pure YAML without
  inheritance/interpolation breeds copy-paste. This design keeps YAML as the source format while
  removing the duplication that makes YAML painful.

### Events (`lofop.core.events`)

- `EventBus` is synchronous, priority-ordered publish/subscribe over dotted string topics.
- Handler failures are isolated by default (logged + returned) so one bad subscriber cannot kill a
  training run; emitters that need strictness use `raise_errors=True`.
- This is the seam where training hooks, experiment trackers, and plugins attach without the
  trainer knowing about them.

### Plugins (`lofop.core.plugins`)

- A plugin is a callable receiving a `PluginContext(hub, events)`. Distribution is either via the
  `lofop.plugins` entry-point group (discovered from installed packages, loaded lazily) or
  programmatic `add()`.
- Failures wrap in `PluginError` with the plugin name and source attached; a failed activation
  never marks the plugin active.

### Errors and logging

- Every framework error derives from `LofopError` and can carry structured `context` data; each
  subsystem has its own subclass (`ConfigError`, `RegistryError`, `BuildError`, `PluginError`,
  `EventError`).
- Logging namespaces under the `"lofop"` root logger. `configure_logging()` is an opt-in
  convenience (console + optional file, Rich when available); embedding applications keep full
  control by simply not calling it.

## Model direction (Phase 3 preview)

The flagship detector (working name: **LOFOP detector**) will be assembled from the standard
groups rather than existing as a monolith:

- an anchor-free, multi-scale prediction scheme over a feature pyramid with dynamic fusion,
- lightweight attention in the neck where it pays for itself on edge hardware,
- dynamic label assignment during training,
- every block registered (`backbone/...`, `neck/...`, `head/...`, `loss/...`) so variants are
  config edits, not code forks.

Design rationale for each block lands with the Phase 3 documentation.
