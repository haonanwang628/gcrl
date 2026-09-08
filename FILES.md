# 新文件及用途

以下均位于 `any_step_mher_ogbench/`；参考仓库未修改。

| 文件 | 用途 |
|---|---|
| `pyproject.toml` | 可安装 Python package、`asm` CLI、测试配置 |
| `requirements-wsl.txt` | WSL 兼容依赖版本 |
| `.gitignore` | 排除数据、运行产物和 Python 缓存 |
| `__init__.py` | 包版本 |
| `cli.py` | prepare／train-representation／train-future／evaluate／plot／smoke 命令 |
| `common.py` | 配置合并与校验、种子、冻结、哈希、JSON、数值检查 |
| `datasets/__init__.py` | 数据模块包标记 |
| `datasets/ogbench.py` | 官方 dataset-only loader、四任务 ID 映射、原始文件身份与行索引 |
| `datasets/trajectories.py` | 显式转移重建、末 next observation、边界校验、split 隔离、保存加载 |
| `datasets/samplers.py` | 严格 k<H future tuples；表示预训练专用 temporal sampler |
| `datasets/normalization.py` | 固定 train-only normalizer，保存与加载 |
| `representations/__init__.py` | 表示模块包标记 |
| `representations/encoder.py` | TempDATA 双 phi、距离 value、MLP 架构 |
| `representations/decoder.py` | 从第一个 phi 恢复完整归一化 observation |
| `representations/losses.py` | TempDATA expectile TD、整批 smooth loss、pre-step target EMA |
| `representations/training.py` | encoder→冻结→decoder→冻结，验证与 checkpoint 关联 |
| `future_models/__init__.py` | 未来模型包标记 |
| `future_models/networks.py` | 参考 ADMPO 的残差 MLP block |
| `future_models/horizon_embedding.py` | 已配置 k 的显式 lookup embedding |
| `future_models/one_step.py` | 固定 goal、无 teacher forcing 的递归预测 |
| `future_models/any_step.py` | 单次直接 horizon-conditioned residual 预测 |
| `future_models/losses.py` | 真实 endpoint 的 MSE 监督 |
| `future_models/training.py` | 四模型共享 tuple 流、冻结检查、磁盘索引审计、模型存取 |
| `evaluation/__init__.py` | 模型评估包标记 |
| `evaluation/future_model_eval.py` | 同表、逐 k 指标；CUDA 同步计时；参数量；输出预测 |
| `evaluation/reporting.py` | L2/MSE、分位数、tail 汇总与 CSV |
| `evaluation/visualize_pointmaze.py` | PointMaze endpoint 散点比较，不绘制推定轨迹 |
| `configs/default.yaml` | 共用正式实验默认配置 |
| `configs/smoke.yaml` | 小网络、每阶段三次更新的流程验证配置 |
| `configs/datasets/pointmaze-medium.yaml` | PointMaze Medium navigate |
| `configs/datasets/pointmaze-large.yaml` | PointMaze Large navigate |
| `configs/datasets/antmaze-medium.yaml` | AntMaze Medium navigate |
| `configs/datasets/antmaze-large.yaml` | AntMaze Large navigate |
| `tests/conftest.py` | 可区分轨迹的数值 fixtures、测试线程约束 |
| `tests/test_datasets.py` | 边界、末状态、排序、horizon、split、官方 loader 接口测试 |
| `tests/test_representation.py` | 原始损失方程、梯度边界、target 更新、冻结测试 |
| `tests/test_future_models.py` | 单次 Any-Step 调用、显式 k、递归自身预测、固定 goal、反传测试 |
| `tests/test_pipeline.py` | Phase 1–4 完整流程、同 tuple 审计、保存加载、指标与隔离验收 |
| `README.md` | WSL 环境、测试／训练／评估命令、输出结构 |
| `SOURCES.md` | 精确源码映射、全部实现偏离与指标定义 |
| `source_manifest.json` | 本次实际参考文件 SHA256 |
| `FILES.md` | 本文件 |
| `VALIDATION.md` | 实际验证结果、未通过的外部验证和 Phase 5 前事项 |

## 自动生成的文件

`outputs/` 中的每次 smoke run 包含 README 所列 data、representation、models、evaluation
产物，另有 `smoke_result.json`。fixture 数据和结果均明确标注为数值测试，不能作为真实 OGBench
实验结果。`outputs/test-results.xml` 保存最终 pytest 的机器可读结果。

`.pytest_cache/`、`__pycache__/` 与 `*.egg-info/` 是测试／editable install 生成的缓存和安装元数据。
它们不是研究实现，已加入 `.gitignore`。WSL 独立 venv 位于
`/home/junf/.venvs/any-step-mher-ogbench/`，是依赖环境，不属于源码。
