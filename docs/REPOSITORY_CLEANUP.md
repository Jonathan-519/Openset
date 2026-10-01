# 仓库库存与兼容清理

## 审计范围

库存基线为 `1d7e4e432c5ceca2abd5e9f3c9530132ea9b1a0f`。该提交共有 3,383 个已跟踪文件，其中 Python 源文件 155 个、29,279 行，JPEG 3,157 个、YAML 配置 17 个。数字来自 Git 清单及对基线全部 Python 文件的 AST 解析，不依赖当前是否完整检出图片。新增算法和清理文件使当前数量增加，可用下述工具重新生成当前库存。

审计覆盖全部 Python 文件的语法、导入、顶层定义、CLI 入口、重复函数体与配置路径声明；人工进一步核查入口、模型/数据加载关系、版本规划器、源文件 hash 绑定、重复 I/O 与测试收集。这里的“全仓库存”不表示逐字阅读图片、权重、历史实验输出，也不表示已经训练或验证了每个历史算法。

| 模块 | 基线 Python 文件数 | 主要职责 |
| --- | ---: | --- |
| 仓库根目录 | 39 | 历史训练/校准/测试 CLI、路由器、评估工具 |
| `loader` | 11 | 数据集、采样器、变换、层级与树工具 |
| `losses` | 2 | 原有损失与导出接口 |
| `models` | 14 | CLIP、CoOp、CoCoOp、MaPLe、分类器与树切分 |
| `optim` | 2 | 优化器与学习率调度 |
| `prepro` | 6 | 已知类视图与不同版本的数据划分 |
| `taxosafe_visual` | 6 | 视觉支持、残差证据、校准、指标 |
| `taxosafe_hier` | 5 | v4 局部层级证据与适配器 |
| `taxosafe_witness` | 4 | v5 局部支持验证器 |
| `taxosafe_dual` | 3 | v7 语义/形态双证据 |
| `taxosafe_regime` | 3 | v8 分组部分池化校准 |
| `taxosafe_interlock` | 3 | v9 层级安全联锁 |
| `taxosafe_dcbs` | 6 | v11 深度条件边界合成 |
| `taxosafe_support` | 10 | 新支持条件层级学习及阶段协议 |
| `tests` | 23 | 历史算法和协议的 CPU 单元测试 |
| `tools` | 18 | 规划、执行、诊断、审计、安装与打包 |

## 已完成的安全清理

1. 将 `calibrate_taxolocal_v21.py` 与 `taxosafe_eval_utils.py` 的五个重复纯 I/O 函数移入 `taxosafe_io.py`：`load_yaml`、`resolve_run_dir`、`sha256_file`、`write_json`、`write_jsonl`。原模块保留原名称、参数和默认值的兼容 wrapper。UTF-8 编码、中文输出、JSON 键排序/缩进/末尾换行、JSONL 顺序、路径解析、分块 hash 及不修改输入记录的行为保持不变。随抽取删除只用于这些函数的重复 imports。新 I/O 模块不导入 PyTorch 或模型。
2. 新增 `tools/audit_code_inventory.py`。它只解析源文件和配置，不 import 项目算法，不读取图片像素、权重或数据清单内容。Git 模式保留 sparse checkout 中的已跟踪路径；ZIP 普通目录自动使用 `inventory_mode=filesystem`、`base_commit=null`，跳过 `raw`、`runs`、`.git`、虚拟环境、`__pycache__` 和 `weights`。缺失配置路径作为报告项呈现，不因为旧数据未安装而崩溃。重复函数与未使用导入只是人工复核候选。
3. 新增 `tools/run_unit_tests.py`，把测试顶层固定为仓库根目录，只发现 `tests/`。保留历史测试的裸模块 fixture import 兼容，不改变根目录推理 CLI。
4. 将 suite/followup 共用的临时数据配置移到 `tests/_fixtures.py`。研究数据不再是这些单元测试的前置条件，原协议、hash、防覆盖、checkpoint 复用等断言全部保留。
5. `tests/test_taxolocal.py` 不再读取服务器上可能不存在的历史生成结果。测试在临时目录构造图像和已知类清单，调用原始数据生成器，继续检查物种互斥、不加入新训练数据、119/117 张图像计数。生产预处理代码未变。
6. 修复 `tools/plan_taxosafe_partial_pooling.py` 的输入验证契约，balanced-v3 通过同一函数受益：基础数据和 taxonomy 仍全部验证；`lambda_oe > 0` 时必须有 OE 配置、文件和原冻结 hash；关闭 OE 时，如果旧 receipt 已冻结它，仍验证其字节；关闭且原计划从未冻结时，不要求或读取该无关文件。算法、阈值、CLI、checkpoint 和函数签名未变。

第 6 项是实际规划链修复：当前 `run_taxosafe_suite.make_plan` 正确地不冻结已禁用 OE，而旧 followup 验证器无条件要求 OE，造成一个正常生成的当前计划无法继续。测试现在同时覆盖当前无 OE 文件的计划、旧 receipt 的额外冻结约束，以及开启 OE 却缺配置或 hash 的拒绝路径，没有靠补造当前计划的 hash 绕过这个问题。

## 发现但保留的重复与历史接口

| 发现 | 处理与理由 |
| --- | --- |
| v5/v7/v8/v9 runner 的 `portable`、`verified`、`run`、`summarize` 等函数体重复 | 保留全部历史 runner 字节。各版本计划、依赖和输出语义不同，现有 receipt 还会绑定源码 hash。不能仅凭重复体合并整个 runner。 |
| `metrics_open.py` 与 `taxosafe_visual/metrics.py` 的部分指标实现重复 | 保留历史指标签名与版本来源，避免改写已冻结实验的解释。 |
| CLIP/CoOp/CoCoOp/MaPLe 有相似模型组件 | 保留版本与 checkpoint 属性结构；没有因当前 main 未导入而删除历史算法。 |
| dcbs/support 的部分随机种子、层级与加载逻辑相同 | 保留 v11 对照代码；本次算法演进只在新的支持实现内进行。 |
| 一些导入看起来未被本模块引用 | `__init__` 导出、跨模块导入以及注册/副作用均可能需要；未进行自动删除。只删除随五个 I/O helper 抽取而已能证明多余的 imports。 |
| `tools/audit_repository.py` 假定旧配置有 `full_data_root`/`ood_root` 及已生成清单 | 保留历史工具，新代码库存工具提供兼容新配置及 ZIP 目录的静态入口，不替换图像/数据协议审计。 |
| 根目录多个 `test*.py` | 都是历史推理 CLI，不是死测试或重复垃圾，全部保留。 |

既有图片、数据清单、配置、receipt、实验目录和历史 CLI 均未因“清理”被删除。本次没有将 AST 的“未被引用”直接等同于死代码。

`plan_taxosafe_partial_pooling.py` 的源码在第 6 项中发生了必要变化，因此绑定其旧字节的已冻结计划仍应从原始提交复现；不要重写旧 receipt 或其 hash。所有历史 `run_taxosafe_*.py` 执行器保持原字节。新的 followup 计划会正常绑定修复后的规划器。I/O wrapper 的输出兼容并不表示任何外部系统记录的整个源码文件 hash 会保持不变。

## 使用与验证

在已安装项目依赖的环境中运行：

```bash
python tools/audit_code_inventory.py
python tools/audit_code_inventory.py --output /tmp/openset-inventory.json
python tools/run_unit_tests.py
```

`--output` 使用独占创建，已有报告不会被覆盖。详细 JSON 包含逐模块定义/导入、候选重复函数位置、未检出文件、配置路径与测试发现风险；不指定输出路径时只打印摘要。

等价的手工测试发现命令为：

```bash
PYTHONPATH=.:tests python -m unittest discover -s tests -t . -v
```

避免省略 `-t .`：`tests/test_taxosafe_visual.py` 和根目录 `test_taxosafe_visual.py` 同名，混用裸模块导入会造成 discovery 的 import-path 冲突。也不应从仓库根目录默认发现 `test*.py`，因为它会 import 推理 CLI 及模型依赖。

清理前的首次完整回归收集了 188 个测试，其中 4 个错误来自原有测试依赖缺失的研究数据：三个 followup 用例缺 `Zooplankton_Taxonomic_Tree_clean_v1/gt_test_extra.txt`，另一个缺 `Zooplankton_Taxonomic_Tree_taxolocal_v1/split_protocol.json`。这不是四个算法断言失败。临时 fixture 消除了该环境依赖，并进一步暴露、修复上述关闭 OE 时的真实规划链问题。

本清理的定向检查覆盖 I/O 字节兼容、路径/YAML 行为、不引入模型导入、AST 不执行被审计代码、ZIP 无 Git 回退、固定测试发现顶层，以及原 suite/followup/TaxoLocal 回归。整合后的 `python tools/run_unit_tests.py --quiet` 共 224 项全部通过；同一轮库存解析 166 个 Python 文件，未发现语法错误或未检出的 Python 文件。CPU 测试不代表实际数据上的训练效果、CUDA 吞吐或四项任务指标已达标；这些由独立实验协议验证。
