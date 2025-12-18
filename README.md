## 使用说明

### 运行评估脚本（custom 50 张小目标集）

```
.\.venv\Scripts\python scripts\benchmark_custom.py --device cpu
```

### 本地启动 UI（Gradio）

```
.\.venv\Scripts\python -m app.ui --host 127.0.0.1 --port 7860
```

访问：http://127.0.0.1:7860 ，上传图片，选择设备和运行模式（Baseline/Enhanced/Both），查看可视化与 JSON 下载。
