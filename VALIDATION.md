# Phase 1–4 验收记录

验证环境：Ubuntu-22.04 / WSL，Python 3.9，
`/home/junf/.venvs/any-step-mher-ogbench/bin/python`。
GPU：NVIDIA GeForce RTX 3060 Laptop GPU（6 GB），PyTorch `2.5.1+cu121`。

## 已完成

| 检查 | 结果 | 证据 |
|---|---|---|
| 单元与集成测试 | **30 passed**，5.59 秒 | `outputs/test-results.xml` |
| 四个任务的官方 loader 接口 | 通过；使用官方格式小型 fixture，并禁止 `gymnasium.make` | `tests/test_datasets.py` |
| 最后有效 next observation、五种终止／截断标记、k<H、split 隔离 | 通过 | `tests/test_datasets.py` |
| TempDATA 损失方程、平滑梯度、pre-step Polyak、decoder 冻结 | 通过 | `tests/test_representation.py` |
| Any-Step 单次网络调用、显式 k、one-step 自身预测递归 | 通过 | `tests/test_future_models.py` |
| CPU Phase 1–4 完整数值流程与同 tuple 审计 | 通过 | `tests/test_pipeline.py` |
| CUDA Phase 1–4 完整数值 smoke | **通过，fixture=true** | `outputs/fixture-cuda-verified/smoke_result.json` |
| per-k 指标、checkpoint 保存加载、PointMaze endpoint 图生成 | 通过 | `outputs/fixture-cuda-verified/evaluation/` |

最终执行命令（Ubuntu 中）：

```bash
source /home/junf/.venvs/any-step-mher-ogbench/bin/activate
cd /mnt/e/Vscode/gcrl/any_step_mher_ogbench
python -m pytest tests -q --junitxml=outputs/test-results.xml
asm smoke --fixture --device cuda --output outputs/fixture-cuda-verified
```

重跑 smoke 时请使用新的空 output 路径。fixture 验证集故意与训练集位置错开，用于发现 split
或归一化泄漏；误差很大是这个诊断设置和三步训练的结果，不能解释成模型有效性比较。

## 尚未完成：真实 OGBench 数据 smoke test

已尝试官方：

```text
https://rail.eecs.berkeley.edu/datasets/ogbench/pointmaze-medium-navigate-v0.npz
```

官方 Python 下载路径返回原始错误：

```text
urllib.error.HTTPError: HTTP Error 403: Forbidden
```

独立 curl 检查得到 `302`，重定向至：

```text
https://iris.eecs.berkeley.edu/datasets/ogbench/pointmaze-medium-navigate-v0.npz
```

随后返回 `404 Not Found`。用户已确认本地没有原始 train/validation 文件。
本项目未生成环境转移、未使用未验证镜像替换官方数据、未声称真实数据 smoke 通过。

按用户要求再次从已安装的 `ogbench.utils.DATASET_URL` 动态读取下载基址，直接 GET
train 与 validation 两个文件：两者均重定向至 `iris.eecs.berkeley.edu` 并返回
`HTTP Error 403: Forbidden`。独立 WSL curl HEAD 请求两者均返回重定向后的 404。
本次直接下载尝试与库版本、完整 URL、最终 URL 已记录在 `outputs/download_retry.json`；
复试脚本为 `outputs/download_retry.py`，不修改 OGBench 或研究模型代码。

因此，**实现的单元／集成／CUDA 数值测试通过，但真实 OGBench 数据验收仍有外部阻塞**。
四个任务的大规模数据加载及正式训练效果尚未验证。

恢复步骤：将可验证的官方原始 `pointmaze-medium-navigate-v0.npz` 与
`pointmaze-medium-navigate-v0-val.npz` 放入同一目录，然后：

```bash
asm smoke --dataset-dir /path/to/original/ogbench --device cuda \
  --output outputs/pointmaze-medium-real-smoke
```

## Phase 5 之前

1. 解决原始数据获取，完成上述真实数据 smoke test。
2. 执行正式 encoder／decoder／四模型训练；检查不同 k 的误差、decoder floor 与长尾。
3. 按需要验证其余三个任务的数据与运行；当前配置和接口支持不等于四任务实验完成。
4. 本次没有实现 Phase 5–7；不得把数值测试通过解释为自动启动后续阶段的授权。

## 安装记录

最初旧 pip 选择了 MuJoCo 源码包，报 `RuntimeError: MUJOCO_PATH environment variable is not set`。
已在独立 venv 升级 pip，并固定 `mujoco==3.2.7`、`dm-control==1.0.27`、
`gymnasium==1.0.0`、`ogbench==1.2.1`；使用预编译包安装成功。
原 `rl-env` conda 环境和三份参考仓库未修改。
