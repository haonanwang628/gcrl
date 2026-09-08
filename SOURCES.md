# 源码映射与实现偏离（Phase 1–4）

本项目的模型是 **Goal-Conditioned Future Model**，输入 `(current, goal, k)`。
它不是 ADMPO 原始的 action-sequence dynamics，也不是当前策略的环境动力学模型。
本文件中的 source 路径相对于上级工作区；新文件路径相对于本项目。

## 精确来源

| 来源文件 | 实际类／函数 | 新文件 | 保留／适配内容 |
|---|---|---|---|
| `TempDATA/src/special_networks.py` | `LayerNormMLP`, `LayerNormRepresentation`, `GoalConditionedPhiValue.setup/get_phi/__call__` | `representations/encoder.py` | 双独立 phi；第一个 head 导出潜表示；negative Euclidean value；平方距离下限 `1e-6`；GELU 后 LayerNorm；末层线性 |
| `TempDATA/src/special_networks.py` | `Decoder.setup/__call__` | `representations/decoder.py` | 无视觉模块时的 MLP decoder，输出完整归一化状态 |
| `TempDATA/jaxrl_m/networks.py` | `default_init` | `representations/encoder.py` | `variance_scaling(1, fan_avg, uniform)` 对应 Xavier uniform；bias 为零 |
| `TempDATA/src/agents/TempDATA.py` | `expectile_loss`, `compute_value_loss`, `repr_fn` | `representations/losses.py` | 双 TD target；target-min 构造 advantage；target-mean 当前 value；expectile；原始整批局部平滑约束 |
| `TempDATA/src/agents/TempDATA.py` | `TempDATA.repr_update` | `representations/losses.py:update_target`, `representations/training.py` | target EMA 使用 optimizer 更新前的 online 参数 |
| `TempDATA/src/agents/TempDATA.py` | `compute_decoder_loss`, `auto_fn`, `TempDATA.dec_update` | `representations/training.py` | 只提取 decoder MSE 分支，移除 auto_fn 中动态模型训练 |
| `TempDATA/src/dataset_utils.py` | `GCDataset.sample_goals/sample` | `datasets/samplers.py:TemporalSampler` | 表示预训练的 current/future/random 目标采样；几何 future offset；当前状态索引成功标签 |
| `TempDATA/How_to_pretrain_representation.ipynb` | `p_currgoal`, `p_trajgoal`, `p_randomgoal` 配置单元 | `configs/default.yaml` | 表示采样默认 `0 / 0.625 / 0.375` |
| `ADMPO/dynamics/arm.py` | `Swish`, `ResBlock.forward` | `future_models/networks.py` | Linear → Swish/SiLU → residual add → LayerNorm 的结构 |
| 官方 `ogbench/utils.py`，安装包 **1.2.1** | `make_env_and_datasets`, `download_datasets`, `load_dataset` | `datasets/ogbench.py` | 直接调用 `dataset_only=True, compact_dataset=False`，不创建环境 |
| 官方 `ogbench/utils.py` | `load_dataset` 的 regular dataset 分支 | `datasets/trajectories.py` | 使用其显式真实 next observation 重建轨迹，保留末状态 |
| 官方 `ogbench/locomaze/point.py`, `ant.py` | `get_ob`, `get_xy` | `evaluation/future_model_eval.py` | 本版 state-based PointMaze／AntMaze 的 observation 前两维为物理 XY |

官方源码：https://github.com/seohongpark/ogbench 。本项目通过固定版本依赖使用 OGBench，
没有复制或改写它的下载器、环境实现或 agent。源码快照哈希另见 `source_manifest.json`。

## 明确的实现偏离

1. **框架：**TempDATA 的 JAX/Flax 实现适配为 PyTorch。保留目标函数与梯度边界，
   不声称 checkpoint 或随机初始化逐位兼容。GELU 使用 tanh approximation，LayerNorm epsilon
   使用 `1e-6`；Xavier uniform 与 source 的初始化分布相匹配，随机数流不同。
2. **模块拆分：**不建立 `TempDATANetwork` 的 policy、skill、critic 和 action-conditioned dynamics。
   表示与 decoder 分阶段训练，各自独立 optimizer；encoder 冻结后 decoder 才开始训练。
3. **decoder 标度：**移除 `compute_decoder_loss` 无条件的 `/255`，因为输入是连续状态。
   本项目预测 `N(state)`，通过训练集拟合的固定 mean/std 反归一化。无 clipping、白化或在线统计更新。
4. **保留易误改语义：**局部约束是整个 batch 的 Frobenius norm，不是每个样本分别求 norm；
   当前 `phi(s)` stop-gradient，`phi(s_next)` 可微。target 参数不参与梯度，EMA 使用更新前参数。
   `tests/test_representation.py` 检查公式、平滑项梯度和 Polyak 顺序。
5. **表示目标采样：**保留 current/future/random 的概率语义和几何 future 采样。
   改用独立 NumPy Generator；将原始等价的多次 Bernoulli 选择写成 categorical 选择，随机流不同。
   random 目标只来自当前 split，可以来自不同轨迹，这是表示学习的负／远目标；
   future-model 的三个状态则必须来自同一轨迹。完整末状态也允许作为表示目标、decoder 训练样本。
6. **ADMPO：**只适配 `ResBlock` 的结构。`ARModel`、GRU、动作序列、概率输出和 ADMPO
   多步 target 构造均未复用。新模型的 goal conditioning、显式 horizon embedding、真实轨迹 target
   采样是本项目新实现，不能标成 ADMPO 原有模块。
7. **Any-Step：**一个网络共享所有配置 horizon 的参数；一次 `delta` 网络调用输出 endpoint。
   新的 lookup embedding 只支持已配置的 k，不声称能插值或外推未训练 horizon。无 dropout。
8. **一步对照：**一步网络没有 horizon embedding，k 决定递归次数。每一步固定同一个 goal，
   输入模型自己的预测；训练和评估都不 teacher-force。使用递归 endpoint MSE 对相同多 horizon
   tuples 监督，属于“一步参数化、递归多步监督”，不是仅在 k=1 训练的经典动力学基线。
9. **目标变量：**`g=s[t+H]` 且 `1<=k<H`；H 是审计字段，不作为模型输入。预测条件不含动作。
   FutureSampler 均匀采样 k，再均匀采样该 k 的合法起点，再均匀采样合法 H。
10. **超参数：**表示 batch=256、hidden=[512,512,512]、latent=32、expectile=.95、
    discount=.99、tau=.005、smooth=.01。网络／步数等均是明确可配置的 V1 工程默认值，
    不声称复现原论文结果。smoke 配置会缩小网络和训练次数，仅验证执行流程。
11. **目标评价：**仅计算 endpoint 几何误差；直线距离不是可行路径长度，不证明拓扑一致或可达。
    不添加 TMD、reachability loss、最近邻投影、拓扑筛选或不确定性过滤。

## 数据边界与身份

OGBench 原始 `.npz` 含终止 sentinel observation；regular loader 将它移入最后一条转移的
`next_observations`。重建后每条有 L 条转移的轨迹保存 L+1 个真实状态。
`source_rows` 对官方文件记录原始 NPZ 行号，`is_next=True` 表示取该行真实转移的 next state，
而不是声称末状态有一个真实 outgoing action。

`terminals/terminated/truncated/timeouts/truncations` 按 OR 切断轨迹；文件末尾也切断。
未标记的 observation 不连续会报错，不跨 reset 猜测连接。官方 train/val 文件分别重建；
`split_trajectories` 为没有官方 val 的显式调用场景提供 whole-trajectory fallback，
官方 loader 不会在缺少 `-val.npz` 时悄悄重新划分数据。

Prepared dataset 保存原始文件 SHA256、数据内容 fingerprint、split、trajectory ID、state index、
source row 和末 next-state 类型。Normalizer 只拟合 train fingerprint；representation、decoder、
future checkpoint 之间的哈希关联在加载时检查。Phase 3 四个模型每步共享同一个 tuple batch，
完整整数索引以 memmap 流式写盘，不随着训练步数累积内存。

## 评价指标定义

每个 k、每个模型分别报告 per-sample L2 与逐坐标 MSE 的 mean、median、p90、p95、p99、
以及大于等于 p95 的 tail mean。raw、normalized、XY 指标适用于全部模型；latent 指标
只适用于 latent 模型。`decoder_only_*` 是 `D(E(real_target))` 的误差；
`full_prediction_decoder_*` 是 `D(F(E(s),E(g),k))` 的误差，不用两种误差相减估算模型误差。

两种 latent 模型使用同一个 encoder 和 decoder；逐 horizon 的 decoder floor 应相同。
参数计数分别列 predictor、完整 twin encoder、实际使用的第一个 encoder head、decoder 和
实际推理总参数。计时包含 warmup 和 CUDA 同步；分别测 predictor 与完整表示/解码链路，
排除 NumPy normalization、host→device 传输和文件读取。amortized time 不是单请求延迟。
