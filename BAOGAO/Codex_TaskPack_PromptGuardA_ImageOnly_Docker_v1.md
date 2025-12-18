# Codex 生成任务包（最终版）— PromptGuard-A（图片-only + Docker + 小目标增强检测）

版本：v1.0（定稿）  
日期：2025-12-17（Asia/Tokyo）  
用途：把本文档直接交给 Codex / Kimi 高级模型，按 Task 顺序生成代码与测试；你负责集成与验收。  
交付：Docker（CPU 可运行）+ Web GUI（Gradio）+ 自建评估脚本（50 张图，含标注）

---

## 0. 决策锁定（不要再改，否则返工）

- 选题：**A：小目标/密集场景增强检测系统（Baseline vs Enhanced 对比）**
- 输入：**图片-only**
- 交付：**Docker**
- Baseline 模型：**YOLOv8n（COCO 预训练）**（ultralytics）
- 检测范围：**COCO 80 类全量推理**
- Enhanced 方法：**全图兜底 + 切片推理（tiling）+ WBF（Weighted Boxes Fusion）融合**
- Enhanced 默认参数：`tile_size=640`, `overlap=0.2`, `wbf_iou=0.55`
- 评估集：你自建 **50 张密集小目标图片**（必须提供 GT 标注）
- 评估类（仅这 6 类做硬评估）：  
  `person, car, bicycle, motorcycle, bus, traffic light`
- redlight：**不做检测类**，做为 `traffic light` 的派生属性 `state=red|green|unknown`（规则法，不参与硬评估）
- crosswalk：**明确不做**

---

## 1. 成功标准（验收口径）

### 1.1 功能（必须全部通过）
- Docker 启动后在浏览器打开 GUI（默认 `7860`）：
  - 上传图片
  - 一键输出 **Baseline 叠框图**、**Enhanced 叠框图**
  - 输出两者耗时（ms）与框数量
  - 支持下载：两张图片 + 两份 JSON + summary.json
- Enhanced 必须包含三步：**全图检测 + 切片检测 + WBF 融合**
- 输出必须写入：`outputs/<job_id>/...`（目录结构固定）

### 1.2 性能（最低要求）
- 在你的机器上给出两条数字（写进 summary + README）：
  - Baseline 单张耗时（ms）
  - Enhanced 单张耗时（ms）
- 评估脚本输出：
  - `baseline_recall_small@0.5`（6 类合并统计）
  - `enhanced_recall_small@0.5`
  - 提升量（enhanced - baseline）

---

## 2. 数据与标注格式（你必须按这个准备）

### 2.1 目录
```
data/custom/images/xxx.jpg
data/custom/labels/xxx.txt
```

### 2.2 标签格式（YOLO）
每行：`class_id cx cy w h`（归一化 0~1）

### 2.3 评估类及 COCO class_id（写死，避免标错）
- person = 0
- bicycle = 1
- car = 2
- motorcycle = 3
- bus = 5
- traffic light = 9

> 你可以只标这 6 类。其它 COCO 类不需要标，但推理仍是 COCO 80 类全量。

---

## 3. 项目结构（强制）

```
promptguard/
  app/
    ui.py
    service.py
  core/
    config.py
    io_image.py
    detector_yolo.py
    tiling.py
    postprocess.py
    wbf.py
    visualize.py
    export.py
    traffic_light_state.py
    eval.py
    utils.py
  scripts/
    benchmark_custom.py
  assets/
    coco80.names
    eval_classes.yaml
  docker/
    Dockerfile.cpu
  requirements.txt
  README.md
  tests/
    test_iou.py
    test_tiling.py
    test_wbf.py
    test_export.py
    test_tl_state.py
```

---

## 4. 输出格式（写死）

### 4.1 文件清单
```
outputs/<job_id>/
  config.json
  baseline_boxed.png
  enhanced_boxed.png
  baseline.json
  enhanced.json
  summary.json
  run.log
```

### 4.2 baseline.json / enhanced.json
```json
{
  "job_id": "20251217_101010_ab12cd",
  "mode": "image",
  "pipeline": "baseline|enhanced",
  "model": {"name": "yolov8n", "num_parameters": 3151904},
  "image": {"width": 1280, "height": 720},
  "config": {"conf_threshold": 0.25, "nms_iou": 0.5, "tile_size": 640, "overlap": 0.2, "wbf_iou": 0.55},
  "detections": [
    {"label": "person", "class_id": 0, "score": 0.92, "box_xyxy": [12, 34, 220, 480], "attrs": {}},
    {"label": "traffic light", "class_id": 9, "score": 0.81, "box_xyxy": [900, 40, 940, 120], "attrs": {"state": "red"}}
  ]
}
```

### 4.3 summary.json
```json
{
  "job_id": "...",
  "device": "cpu",
  "baseline_ms": 85.2,
  "enhanced_ms": 190.4,
  "baseline_num_boxes": 17,
  "enhanced_num_boxes": 23,
  "num_parameters_total": 3151904,
  "notes": {"tiling": true, "wbf": true, "global_fallback": true}
}
```

---

## 5. 核心算法定义（Enhanced 写死）

Enhanced(image)：
1) `dets_global = YOLO(image)`
2) `tiles = tile_image(image, tile_size=640, overlap=0.2)`
3) 对每个 tile：`dets_tile = YOLO(tile)`，然后 remap 回原图坐标
4) `dets_all = dets_global + dets_tiles_remap`
5) `dets_fused = WBF(dets_all, wbf_iou=0.55)`（按 label 分组融合）
6) 对 `traffic light` 的每个 bbox：`attrs.state = classify_traffic_light_state(image, bbox)`
7) 输出 `dets_fused`

Baseline(image)：
1) `dets = YOLO(image)`  
2) 同样为 `traffic light` 预测 `attrs.state`  
3) 输出 `dets`

---

# 6. 代码规范（AI 大量生成时必须硬控）

1) **禁止中文变量名/函数名/类名**。中文仅允许出现在：README、UI 文案、少量注释。  
2) 全项目 UTF-8。读写文本显式 `encoding="utf-8"`。  
3) 路径统一 `pathlib.Path`。  
4) **中文路径兼容必须实现**（Windows 常见坑）：
   - 读图：`np.fromfile(path, dtype=np.uint8)` + `cv2.imdecode(...)`
   - 写图：`cv2.imencode(...)[1].tofile(path)`
5) UI 层不得包含推理细节：UI -> service -> core。  
6) 日志用 `logging`，禁止满屏 `print`。  
7) 必须带 tests（至少 5 个），并在 Docker 内能跑 `pytest -q`。  
8) 格式化：`black` + `ruff`（可选 pre-commit，但不强制）。

---

# 7. Codex 执行方式（强制：按 Task 顺序逐个交付）

每个 Task：  
- 只修改/新增指定文件  
- 必须附带对应单测（如 Task 要求）  
- 必须给出最小运行示例命令  
- 不得引入训练代码、不做大规模数据下载

---

## Task 0：工程骨架 + 统一配置（core/config.py, core/utils.py）
**目标**：统一 config、job_id、输出目录与日志。

### 交付文件
- `core/config.py`：定义 `JobConfig`（dataclass 或 pydantic）
- `core/utils.py`：`new_job_id()`, `ensure_dir()`, `setup_logger()`

### JobConfig 字段（写死）
- `conf_threshold: float = 0.25`
- `nms_iou: float = 0.5`
- `tile_size: int = 640`
- `overlap: float = 0.2`
- `wbf_iou: float = 0.55`
- `max_det: int = 300`
- `img_max_side: int = 1280`

### 验收
- 运行：
  - `python -c "from core.config import JobConfig; from core.utils import new_job_id; print(new_job_id()); print(JobConfig())"`
- 不报错。

### Codex Prompt（复制即用）
> Implement `core/config.py` and `core/utils.py` exactly as specified. Use UTF-8, no Chinese identifiers. Provide minimal docstrings. Do not import Gradio. Add logging setup helper. No extra features.

---

## Task 1：中文路径稳健读写（core/io_image.py + tests/test_export.py）
**目标**：避免中文路径导致读图失败。

### 交付文件
- `core/io_image.py`：`imread_any(Path)->np.ndarray(BGR)`, `imwrite_any(Path, img_bgr)->None`

### 验收
- 单测：生成一个临时中文文件名图片，写入再读出，shape 相同。

### Codex Prompt
> Implement `core/io_image.py` with robust Unicode path support using np.fromfile + cv2.imdecode and cv2.imencode + tofile. Add tests in `tests/test_export.py` covering Chinese filenames.

---

## Task 2：YOLOv8n 检测器适配器（core/detector_yolo.py）
**目标**：封装 YOLO，输出统一 Detection。

### 交付文件
- `core/detector_yolo.py`

### 接口（写死）
```python
@dataclass(frozen=True)
class Detection:
    label: str
    class_id: int
    score: float
    box_xyxy: tuple[int, int, int, int]
    attrs: dict
```

`class YoloDetector` methods:
- `load(device: str="cpu") -> None`
- `detect(image_bgr, conf_threshold, nms_iou, max_det) -> list[Detection]`
- `num_parameters() -> int`

### 要求
- 默认模型：`yolov8n.pt`（ultralytics 自动下载到缓存即可）
- label 读取：`assets/coco80.names`
- 输出坐标像素 int，裁剪到图像范围内

### Codex Prompt
> Implement `core/detector_yolo.py` as a stable adapter around ultralytics YOLOv8n. Ensure deterministic outputs (sorted by score desc). Provide `num_parameters()` by summing torch parameters. Do not add training code.

---

## Task 3：IoU 与 NMS（core/postprocess.py + tests/test_iou.py）
**目标**：后续 WBF 和评估要依赖。

### 交付文件
- `core/postprocess.py`
- `tests/test_iou.py`

### 要求
- `iou_xyxy(a,b)`：返回 0~1
- `nms(dets, iou_thresh)`：同 label 之间做 NMS（不同 label 不互斥）

### Codex Prompt
> Implement `core/postprocess.py` with IoU and NMS for Detection lists. Add `tests/test_iou.py` including symmetry, bounds, and simple NMS cases.

---

## Task 4：切片（tiling）与回映射（core/tiling.py + tests/test_tiling.py）
**目标**：Enhanced 的基础。

### 交付文件
- `core/tiling.py`
- `tests/test_tiling.py`

### 接口（写死）
- `tile_image(image_bgr, tile_size:int, overlap:float) -> list[tuple[np.ndarray, int, int]]`
- `remap_box_xyxy(box_xyxy, x0, y0) -> box_xyxy_global`

### 边界
- 小图：只出 1 tile（全图）
- stride 至少 1
- tile 覆盖完整图像（右下边缘不能漏）

### Codex Prompt
> Implement tiling with overlap and complete coverage. Provide remapping helpers. Add tests verifying coverage and remapped boxes stay within bounds.

---

## Task 5：WBF 融合（core/wbf.py + tests/test_wbf.py）
**目标**：Enhanced 的“提升点”。

### 交付文件
- `core/wbf.py`
- `tests/test_wbf.py`

### 算法规则（写死）
- 仅同 label 融合
- 聚类：IoU >= `wbf_iou`
- 坐标：score 加权平均
- fused score：取 cluster 内 max
- attrs：融合后先 `{}`

### Codex Prompt
> Implement WBF for Detection. Cluster by IoU threshold per label. Weighted average coordinates by score. Use max score as fused score. Add tests with overlapping boxes and non-overlapping, and different labels not fusing.

---

## Task 6：交通灯红绿状态（core/traffic_light_state.py + tests/test_tl_state.py）
**目标**：实现 redlight 需求（作为 traffic light 的属性）。

### 交付文件
- `core/traffic_light_state.py`
- `tests/test_tl_state.py`

### 接口（写死）
- `classify_traffic_light_state(image_bgr, box_xyxy) -> str`  # "red"|"green"|"unknown"

### 规则法（写死）
- HSV 统计红/绿像素比例；红>阈值且红>绿 -> red；绿同理；否则 unknown
- bbox 必须裁剪到图像范围

### Codex Prompt
> Implement HSV-based traffic light state classifier. Provide unit tests using synthetic images (red/green rectangles). Ensure bbox clipping safety.

---

## Task 7：可视化叠框（core/visualize.py）
**目标**：生成 Baseline/Enhanced 两张叠框图。

### 交付文件
- `core/visualize.py`

### Codex Prompt
> Implement visualization using OpenCV. Draw boxes and "label score". Keep it robust and simple.

---

## Task 8：导出与 Summary（core/export.py）
**目标**：写出 baseline/enhanced JSON、两张图与 summary。

### 交付文件
- `core/export.py`

### Codex Prompt
> Implement export helpers writing JSON with UTF-8 and ensure directories exist. Use imwrite_any for images. Include config snapshot in json objects.

---

## Task 9：服务编排（app/service.py）
**目标**：整合 Baseline 与 Enhanced，计时并落盘输出。

### 交付文件
- `app/service.py`

### Codex Prompt
> Implement `app/service.py` orchestrating baseline and enhanced pipelines exactly as spec. Cache detector. Produce outputs under outputs/<job_id>. Return paths for UI.

---

## Task 10：Gradio UI（app/ui.py）
**目标**：图片-only GUI，可下载输出文件。

### 交付文件
- `app/ui.py`

### Codex Prompt
> Build a Gradio UI for image-only processing. Call service.run_job. Show baseline/enhanced images and timings. Provide file download components.

---

## Task 11：评估脚本（scripts/benchmark_custom.py + core/eval.py）
**目标**：跑出 small recall@0.5 的提升。

### 交付文件
- `core/eval.py`
- `scripts/benchmark_custom.py`

### Codex Prompt
> Implement YOLO-format loader, IoU matching, and recall@0.5 for small objects. Run baseline and enhanced on each image using the same core, without GUI. Output JSON report to reports/custom_eval.json.

---

## Task 12：Docker + README（docker/Dockerfile.cpu, requirements.txt, README.md）
**目标**：可交付。

### Codex Prompt
> Create docker/Dockerfile.cpu, requirements.txt, and README.md for CPU-only Docker deployment. Ensure the container starts the Gradio app on port 7860. Document evaluation steps.

---

# 8. 你必须手动做的事（不做就没有“性能提升”证据）

1) 收集 50 张密集小目标图片  
2) 只标 6 类（person/car/bicycle/motorcycle/bus/traffic light）  
3) 跑评估脚本输出 `reports/custom_eval.json` 并写进报告

---

（结束）
