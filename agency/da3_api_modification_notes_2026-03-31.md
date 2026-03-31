# DA3 API 修改记录（2026-03-31）

## 1. 背景与目标
- 目标是在 IsaacLab 高并行视觉环境中使用 DA3，形成可直接接收 batched RGB tensor 的 GPU 路径。
- 保留原有 `inference()`（CPU/PIL/Numpy 路径）兼容性，同时新增 `inference_torch()` 给在线蒸馏/训练使用。

## 2. 本次改动范围
- `Depth-Anything-3-HAND`:
  - `/workspace/simulation/HAND-distill/dep/Depth-Anything-3-HAND/src/depth_anything_3/api.py`
- `HAND-distill` 接入层:
  - `/workspace/simulation/HAND-distill/source/HAND/HAND/tasks/direct/hand/da3_utils.py`

## 3. `api.py` 主要修改

### 3.1 新增 GPU-only 推理接口
- 在 `DepthAnything3` 中新增 `inference_torch(...)`。
- 输入:
  - `image`: `(N,H,W,3)` 或 `(N,3,H,W)`，支持 3/4 通道（4 通道会丢弃 alpha）。
  - `intrinsics`: 可选 `(N,3,3)`。
  - `extrinsics`: 参数保留。
- 输出:
  - `dict[str, torch.Tensor]`，至少包含 `depth`，并保留模型返回的其它张量键。

### 3.2 Torch 预处理路径
- 新增 `_preprocess_inputs_torch(...)`，对齐 CPU `InputProcessor` 的核心规则：
  - 边界缩放（upper/lower bound）。
  - patch=14 对齐（resize 或 crop 分支）。
  - intrinsics 同步变换（resize/crop 时更新 fx/fy/cx/cy）。
  - ImageNet 归一化。

### 3.3 归一化实现方式（最终版本）
- 当前采用与源码风格一致的 transform 对象：
  - 在 `__init__` 中声明 `self._normalize_torch = T.Normalize(...)`
  - 在 `_preprocess_inputs_torch` 中直接 `imgs = self._normalize_torch(imgs)`
- 说明：你要求“把 #2 改回去”，因此没有保留 `register_buffer` 缓存 mean/std 的版本。

### 3.4 `item()` 同步点移除
- 去掉了通过 `imgs.max().item()` 判断并归一化的逻辑，避免 GPU->CPU 同步开销。
- 现在规则是：
  - 非浮点输入按 `[0,255]` 处理并除以 255。
  - 浮点输入默认视作 `[0,1]`。

### 3.5 输出后处理改为“按键处理”
- `_postprocess_model_output_torch(...)` 改为 key-aware，而非对所有 tensor 广义 squeeze：
  - batch squeeze keys: `depth/depth_conf/sky/extrinsics/intrinsics`
  - map-like keys: `depth/depth_conf/sky`
- 目的：避免未来新增 tensor 键被误改形状。

### 3.6 extrinsics 策略（按当前需求）
- `inference_torch(...)` 中显式保留 API 参数，但当前训练路径忽略 extrinsics：
  - 通过 `extrinsics = None` 进入后续流程。
- 原因：当前目标是 Isaac 仿真高并行 RGB+可选 intrinsics 的在线路径。

## 4. `da3_utils.py` 主要修改

### 4.1 新增 batched torch 接口
- 在 `DA3Inference` 中新增 `infer_torch_batched(...)`：
  - 输入 `rgb` batched tensor（N,H,W,3 或 N,3,H,W）。
  - 调用 `self.model.inference_torch(...)`。
  - 按现有规则保留 metric 模型缩放：
    - 若模型名含 `metric` 且设定 `focal`，则 `depth = focal * depth / 300.0`。

### 4.2 intrinsics 张量缓存
- 在 `DA3Inference.__init__` 中构建并缓存 `self._intrinsics_torch`。
- `infer_torch_batched` 中按 batch 维使用：
  - 优先 `expand`（当 base batch=1）。
  - 不匹配时再 `repeat`。

### 4.3 `infer(...)` 增加 `process_res_method`
- 为了和 torch 路径在预处理策略上可对齐，`infer(...)` 新增 `process_res_method` 参数并传给 `model.inference(...)`。

### 4.4 `__main__` 中新增双路径对比模式（未自动执行）
- 新增参数：
  - `--process_res_method`
  - `--compare_paths`
  - `--compare_frames`
  - `--compare_torch_batch_size`
- 对比模式会对同一批帧计算：
  - MAE
  - RMSE
  - relative median error
  - Pearson 相关性
- 若 shape 不同，会先做中心裁剪到共同尺寸再比较。
- `--compare_paths` 开启后执行对比并退出，不进入原 FPS benchmark 流程。

## 5. 本轮对话中的关键决策记录
- 你要求未许可前不跑程序：后续改动均按“只改代码/文档，不主动跑测试”执行。
- 你确认 `register_buffer` 方案“没必要”：已回退到 transform 归一化方案。
- 你要求保留 extrinsics API 但可暂不使用：已按该约束实现。
- 你要求在 `da3_utils` 中加入 `infer` 与 `infer_torch_batched` 对比测试：已完成 CLI 模式接入。

## 6. 当前状态与后续建议
- 当前代码已具备：
  - 旧路径兼容（`inference` / `infer`）
  - 新路径可接 tiled camera 批量 tensor（`inference_torch` / `infer_torch_batched`）
  - 双路径一致性对比工具（`--compare_paths`）
- 下一步（手动许可后再执行）：
  - 在你的目标环境中跑 `--compare_paths`，确认误差统计满足预期；
  - 再用现有 batch benchmark 测吞吐变化。

## 7. 可复现实验手册（给另一台机器/另一个 agent）

### 7.1 前置条件
- 代码位置（默认）：
  - DA3 repo: `/workspace/simulation/HAND-distill/dep/Depth-Anything-3-HAND`
  - 接入 wrapper: `/workspace/simulation/HAND-distill/source/HAND/HAND/tasks/direct/hand/da3_utils.py`
- Python 环境：
  - 推荐使用 `env_isaaclab_da3`（或具备相同 torch/cuda 版本的环境）。
- GPU：
  - 需要 CUDA 可用（`torch.cuda.is_available()` 为 True）。
- 模型缓存目录：
  - `DA3Inference` 默认使用 `/workspace/simulation/HAND-distill/model/da3`。
- 视频输入（示例）：
  - `/workspace/simulation/HAND-policy/logs/rsl_rl/jar/2026-03-27_19-51-18_cap_side_points/videos/train/rl-video-step-120000.mp4`

### 7.2 快速检查（仅确认导入与接口）
```bash
conda activate env_isaaclab_da3
python -c "from depth_anything_3.api import DepthAnything3; print('DepthAnything3 import ok')"
python -c "from HAND.HAND.tasks.direct.hand.da3_utils import DA3Inference; print('DA3Inference import ok')"
```

### 7.3 一致性测试：`infer` vs `infer_torch_batched`
```bash
conda activate env_isaaclab_da3
python /workspace/simulation/HAND-distill/source/HAND/HAND/tasks/direct/hand/da3_utils.py \
  --model depth-anything/DA3-BASE \
  --device cuda \
  --process_res 336 \
  --process_res_method upper_bound_resize \
  --max_frames 100 \
  --compare_paths \
  --compare_frames 32 \
  --compare_torch_batch_size 8
```

输出里关注：
- `valid_pairs`
- `shape_mismatch_pairs`
- `mean_MAE`
- `mean_RMSE`
- `mean_rel_median_error`
- `mean_pearson`

建议验收（经验阈值，可按任务收紧）：
- `shape_mismatch_pairs` 尽量为 0（若不为 0，确认仅是 resize/crop 细节导致并已中心裁剪比较）。
- `mean_rel_median_error` 越低越好（建议 < 0.1）。
- `mean_pearson` 越高越好（建议 > 0.9）。

### 7.4 吞吐测试：批量路径 benchmark
```bash
conda activate env_isaaclab_da3
python /workspace/simulation/HAND-distill/source/HAND/HAND/tasks/direct/hand/da3_utils.py \
  --model depth-anything/DA3-BASE \
  --device cuda \
  --process_res 336 \
  --process_res_method upper_bound_resize \
  --video_path /workspace/simulation/HAND-policy/logs/rsl_rl/jar/2026-03-27_19-51-18_cap_side_points/videos/train/rl-video-step-120000.mp4 \
  --input_width 640 \
  --input_height 480 \
  --max_frames 100 \
  --batch_sizes 8,16,32,64 \
  --warmup_iters 3
```

输出里关注：
- 每个 batch 的 `mean(ms)`、`total_fps`、`per_env_fps`
- 表格汇总中的 `mean_ms/std_ms/min_ms/max_ms/total_fps/per_env_fps`

### 7.5 常见问题排查
- `Cannot find DA3 source ...`：
  - 设置环境变量 `DA3_SRC=/workspace/simulation/HAND-distill/dep/Depth-Anything-3-HAND/src`
- 导入 `moviepy.editor` 报错：
  - `da3_utils.py` 已有 mock；若自写脚本需复用同样处理或安装兼容版本。
- 吞吐显著下降：
  - 检查是否有后台训练占用 GPU。
  - 检查 `process_res`、`batch_sizes` 是否与基准一致。
- 结果偏差大：
  - 确认两条路径使用了相同 `process_res_method`、相同输入分辨率、相同模型权重。

## 8. 下一步建议（面向接手 agent）
- 把 `infer_torch_batched` 接到 env 的 tiled camera 观测流，确保输入直接是 batched RGB tensor。
- 在 env 中先做小规模并行（例如 8/16）联调，再扩大到目标并行数。
- 固定一版 compare+benchmark 结果到 `agency/`，作为后续改动的回归基线。

## 9. 备注
- 本文档整理实现、决策和复现步骤。
- 我在这轮文档更新时没有执行新的推理/benchmark，仅提供可复现命令。
