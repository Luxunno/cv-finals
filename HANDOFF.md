## 1. 项目一句话目标
面向密集小目标的图像检测：Baseline（整图 YOLOv8n）vs Enhanced（整图 + 切片 + WBF），提供 UI、评估脚本与可落盘复现证据链。

## 2. 当前已实现能力清单
- Baseline/Enhanced 推理：全图 + 切片/重叠 + WBF + NMS，CPU 可跑。
- UI：上传图片、对比 Baseline/Enhanced/Diff、下载 JSON/图片、参数一致性自检。
- Export：outputs/<job_id>/ 下落盘 config.json、summary.json、可视化图、run.log。
- Eval：scripts/benchmark_custom.py 输出 small recall@0.5 与 boxes_avg。

## 3. 一键命令（Windows PowerShell）
- 单测：`.\.venv\Scripts\python -m pytest -q`
- UI：`.\.venv\Scripts\python -m app.ui`
- 评估：`.\.venv\Scripts\python scripts\benchmark_custom.py --device cpu --profile balanced`
- 单张：`.\.venv\Scripts\python run_one.py --image <path> --device cpu`

## 4. 目录结构说明
- app/：service（推理编排）、ui（Gradio）
- core/：config、tiling、wbf、export、eval 等核心逻辑
- scripts/：benchmark_custom、smoke_twice
- tests/：单测
- data/：自建数据集（不入库）
- outputs/：推理输出（不入库）
- reports/：评估输出与状态快照

## 5. 配置字段口径
- effective_config（生效字段）：global_conf_threshold/tile_conf_threshold/tile_size/overlap/tile_pad_px/tile_scale/wbf_iou_normal/wbf_iou_small/small_area_thresh/nms_iou/max_det/device
- deprecated：conf_threshold、wbf_iou（必须与 global_conf_threshold/wbf_iou_normal 同步）
- config.json 结构：full_config / effective_config / deprecated_fields / deprecated_mismatch

## 6. 当前已知问题清单
- P1：bicycle/bus small GT 数偏少，per-class recall 易为 0（数据侧供给问题）。
- P1：local data/outputs/reports 不入库，复现依赖本地数据。
- P2：UI diff 的 Added/Missing 依赖 diff_iou_thresh，阈值选择影响解释性。

## 7. 下一步任务 Top5
1) 数据侧：扩充 bicycle/bus small GT（filter/patches 与选片脚本），验收 per-class recall > 0。
2) 评估可追溯：在 benchmark 输出每图 TP/FP 明细（scripts/benchmark_custom.py），验收 diff 可解释。
3) UI 辅助：增加 diff 过滤/排序（app/ui.py），验收 Added/Missing 更可读。
4) 文档：完善评估数据来源与标注流程（README/HANDOFF），验收新人可复现。
5) Docker：补 Dockerfile.cpu 与运行说明（docker/ + README），验收容器启动 UI。

## 8. 验收清单
- 运行 `.\.venv\Scripts\python -m pytest -q` 全绿。
- 运行 UI：`python -m app.ui`，outputs/<job_id>/ 生成 baseline/enhanced/diff + config/summary/run.log。
- 运行 benchmark：`python scripts\benchmark_custom.py --device cpu --profile balanced`。
  - reports/custom_eval.json 含 profile、effective_config、boxes_avg。
  - outputs/<job_id>/config.json 含 effective_config 且无 deprecated_mismatch。
