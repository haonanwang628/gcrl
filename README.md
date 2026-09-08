# Goal-Conditioned Any-Step Future Prediction — Phase 1–4

实现范围：OGBench 固定离线轨迹、TempDATA temporal representation、四种受控未来模型、
future-model-only evaluation。没有 hindsight relabeler、GCIQL、policy evaluation 或在线采样。
模型只接收 `(current, goal, k)`，不接收动作。完整来源与偏离见 [SOURCES.md](SOURCES.md)，
全部文件用途见 [FILES.md](FILES.md)，实际验收结果见 [VALIDATION.md](VALIDATION.md)。

## Ubuntu / WSL 环境

在 PowerShell 中进入用户指定的环境：

```powershell
wsl -d ubuntu-22.04
```

随后在 Ubuntu shell 中：

```bash
cd /mnt/e/Vscode/gcrl/any_step_mher_ogbench
source /home/junf/.venvs/any-step-mher-ogbench/bin/activate
```

本次已创建以上独立 venv，并复用 `rl-env` 的 CUDA PyTorch，不修改原 conda 环境。
如需重建同类环境（新路径）：

```bash
/home/junf/miniconda3/envs/rl-env/bin/python -m venv --system-site-packages ~/.venvs/any-step-mher-ogbench-new
source ~/.venvs/any-step-mher-ogbench-new/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -r requirements-wsl.txt
python -m pip install --no-deps -e .
```

通用新机器需先安装 Python>=3.9 和匹配设备的 PyTorch；这里的绝对环境路径仅对应本工作区。
WSL 启动显示的 localhost proxy 提示不影响本地计算；下载失败需单独检查源站。

## 测试

```bash
python -m pytest tests -q

# 无下载的数值流程验证，输出明确标记 fixture=true，不是 OGBench 实验结果。
asm smoke --fixture --device cpu --output outputs/fixture-cpu
asm smoke --fixture --device cuda --output outputs/fixture-cuda

# 使用官方完整 train/validation 文件的真实数据 smoke test。
asm smoke --env pointmaze-medium-v0 --device cuda --output outputs/pointmaze-medium-smoke
```

smoke 使用每阶段 3 次更新、小网络和每个 k 的 8 个固定 validation tuples。
它检验运行与契约，不检验研究效果或收敛。输出目录必须为空／不存在，程序不会覆盖已有结果。

如果官方自动下载不可用，可将原始文件放在本地目录后运行：

```bash
# 所需文件：
# /path/to/ogbench/pointmaze-medium-navigate-v0.npz
# /path/to/ogbench/pointmaze-medium-navigate-v0-val.npz
asm smoke --dataset-dir /path/to/ogbench --device cuda --output outputs/pointmaze-local-smoke
```

程序仍调用官方 loader，仅命中本地缓存，不生成替代训练数据。
`pointmaze-medium-v0` 是环境 ID；`pointmaze-medium-navigate-v0` 是完整 dataset ID，两者不能混用。

## PointMaze Medium 正式训练命令

下面命令依次执行 Phase 1–4，默认 horizon `[1,5,10,20]`，约束 `1<=k<H`。
正式训练次数是初始配置，不代表已经完成收敛实验。

```bash
# Phase 1：只加载数据；不创建／step 环境。
asm prepare --config configs/datasets/pointmaze-medium.yaml --output data/pointmaze-medium

# Phase 2：先训练 encoder，再冻结 encoder 训练 decoder。
asm train-representation --config configs/datasets/pointmaze-medium.yaml \
  --data data/pointmaze-medium --output outputs/pointmaze-medium/representation --device cuda

# Phase 3：四模型每次更新接收完全相同的 tuples。
asm train-future --config configs/datasets/pointmaze-medium.yaml \
  --data data/pointmaze-medium --representation outputs/pointmaze-medium/representation \
  --output outputs/pointmaze-medium/models --device cuda

# Phase 4：四模型使用 Phase 1 保存的同一张验证 tuple 表。
asm evaluate --config configs/datasets/pointmaze-medium.yaml \
  --data data/pointmaze-medium --representation outputs/pointmaze-medium/representation \
  --models outputs/pointmaze-medium/models --output outputs/pointmaze-medium/evaluation --device cuda

asm plot --evaluation outputs/pointmaze-medium/evaluation \
  --output outputs/pointmaze-medium/evaluation/endpoints-k20.png --horizon 20 --count 4
```

需要自定义步数、网络、seed 时创建 YAML，仅写覆盖字段即可：所有配置会与
`configs/default.yaml` 深度合并。B–D 等名称在这里指四种未来模型，不是后续 GCIQL 实验组。

| 内部模型名 | 表示 | horizon 的作用 |
|---|---|---|
| `raw_one_step` | 归一化原始状态 | 固定目标下递归 k 次 |
| `raw_any_step` | 归一化原始状态 | 显式 embedding；一次预测 |
| `latent_one_step` | 冻结 TempDATA phi | 固定目标下递归 k 次，再解码 |
| `latent_any_step` | 冻结 TempDATA phi | 显式 embedding；一次预测，再解码 |

其他任务使用 `configs/datasets/pointmaze-large.yaml`、`antmaze-medium.yaml`、
`antmaze-large.yaml`，并使用独立 data/output 目录。默认均为官方 `navigate` 数据。
所有任务在 loader、sampler、四模型和数值 evaluation 中支持；绘图仅针对 PointMaze。
未对未知任务的 observation/XY 映射做通用假设。

## 预期输出

```text
data/pointmaze-medium/
  train.npz                     # 状态序列、offsets、真实数据来源索引
  validation.npz                # 独立官方验证轨迹
  validation_tuples.npz         # 每个 horizon 的固定表及索引、split、fingerprint
  manifest.json                 # 原始文件 SHA256、数据身份、配置、依赖版本
outputs/pointmaze-medium/
  representation/
    normalizer.npz              # 只从训练集拟合
    encoder.pt                  # 双 phi、target phi、架构、训练数据身份
    decoder.pt                  # 单独训练的 decoder、关联 encoder hash
    config.json
    validation_metrics.json
    provenance.json
  models/
    raw_one_step.pt
    raw_any_step.pt
    latent_one_step.pt
    latent_any_step.pt
    training_tuple_indices.npy  # [updates,batch,6]，memmap 流式写入
    training_tuple_columns.json
    training_metrics.json
    metadata.json              # 四模型公共 config、tuple stream hash、冻结模块 hash
  evaluation/
    metrics.json               # 按模型和 k：误差分布、计时、参数量
    metrics.csv                # 每行 model/k/metric，包含 mean/median/tail
    predictions.npz            # 全量预测 endpoint 与逐样本误差
    validation_tuples.npz       # 评估实际使用的表，便于审计和绘图
    evaluation_config.json
    endpoints-k20.png
```

默认 `100000 × 256 × 6` 的 int64 训练审计约占 1.23 GB 磁盘，但不一次载入内存。
checkpoint 用于冻结推理，本版没有中途恢复 optimizer 的 resume 功能。

`normalized_state_*`、`physical_state_*`、`maze_xy_*` 适用于全部四模型；
latent 模型另外输出 `latent_*`、`decoder_only_*` 和 `full_prediction_decoder_*`。
每类包括 L2/MSE 的均值、中位数、p90/p95/p99 和 worst-5% mean。
详细定义和计时包含范围见 [SOURCES.md](SOURCES.md)。

PointMaze 图片只画当前状态、最终 goal、真实 k-step endpoint 与预测 endpoint 的标记。
它不连接 endpoint，也不把欧氏距离或散点图解释为实际执行轨迹、可达性或墙体拓扑。

## Phase 5 前的边界

Phase 1–4 单元与数值流程测试通过后，仍需要真实离线数据 smoke test、正式训练与模型误差审查。
当前源码不包含 Phase 5–7。即使上述测试通过，也不会自动启动 relabeling、GCIQL 或 policy evaluation。
