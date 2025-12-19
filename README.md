## Quick Start（交付验证）
```
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m app.ui
.\.venv\Scripts\python scripts\benchmark_custom.py --device cpu --profile balanced
```
期望输出：
- `outputs/<job_id>/...`（baseline/enhanced/diff 图片、config.json、summary.json、run.log）
- `reports/custom_eval.json`

## PromptGuard（图像检测，YOLOv8n）
- Baseline：整图一次 YOLOv8n 推理
- Enhanced：整图 + 切片（带 padding）+ WBF 融合（含小框自适应阈值）+ NMS 清理

### 本地启动 UI（Gradio）
```
.\.venv\Scripts\python -m app.ui --host 127.0.0.1 --port 7860
```
访问：http://127.0.0.1:7860

### Enhanced 小目标增强（推理侧，无训练不换权重）
Enhanced 的改进点（均为推理流程改造）：
1) **Tile padding + 有效区域过滤**
   - 每个 tile 会向四周扩展 `tile_pad_px` 取更大的 crop，缓解 tile 边缘目标被截断导致的漏检。
   - 在 padded crop 上推理后，把框映射回原图，再按“有效区域”（原始 tile 区域）用中心点过滤，减少跨 tile 重复噪声。
2) **分离阈值：global vs tile**
   - `global_conf_threshold` 用于整图推理（相对保守）。
   - `tile_conf_threshold` 用于切片推理（更宽松，提升小目标召回）。
3) **Tile-only Upscale（仅切片放大推理）**
   - `tile_scale`>1 时，只对切片 crop 做放大后推理，再将框坐标除以 scale 映射回原图。
   - 用于提升远处密集小目标的可见性，避免整图放大带来的巨大延迟。
4) **小框自适应 WBF + NMS**
   - 小框（面积 < `small_area_thresh`）使用更低融合阈值 `wbf_iou_small`，其余使用 `wbf_iou_normal`。
   - WBF 后再做一次 label 内 NMS（`nms_iou`）压噪。

### UI 参数档位（Fast / Balanced / Quality）
UI 顶部 `Profile` 提供三个推荐档位（仍可在 Advanced 中微调）：
- **Fast**：更快，提升适中（tile_size=640，overlap=0.18，tile_pad_px=32，tile_conf_threshold=0.20，tile_scale=1.0）
- **Balanced**：默认推荐（tile_size=512，overlap=0.20，tile_pad_px=48，tile_conf_threshold=0.18，tile_scale=1.0）
- **Quality**：更强召回但更慢（tile_scale=1.5，tile_conf_threshold=0.16，tile_pad_px=64）

### 评估（custom 50 张，small recall@0.5）
运行：
```
.\.venv\Scripts\python scripts\benchmark_custom.py --device cpu --profile balanced
.\.venv\Scripts\python scripts\benchmark_custom.py --device cpu --profile fast
.\.venv\Scripts\python scripts\benchmark_custom.py --device cpu --profile quality
```
输出：
- `reports/custom_eval.json`
- 终端会打印 `baseline_recall_small@0.5 / enhanced_recall_small@0.5 / delta`

额外指标（用于证明“提升不是纯堆框”）：
- `baseline_boxes_avg` / `enhanced_boxes_avg` / `delta_boxes_avg`：每张图预测框数量的均值与差值，用于衡量 recall 提升是否伴随预测框数量暴增。

最近一次在本机数据集上的示例输出（仅供参考，数值会随数据集与机器变化）：
- baseline_recall_small@0.5=0.1650
- enhanced_recall_small@0.5=0.2665
- delta=+0.1015
- runtime：baseline_ms_avg≈55.50，enhanced_ms_avg≈158.79（倍率≈2.86，≤3.0）
