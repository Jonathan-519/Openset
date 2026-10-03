# TaxoSafe new：支持集条件化的层级证据学习

> 2026-10-03 更新：relation trial 1 未改善总体表现。当前建议基于较优 reference v3 做独立精修，见 [冻结 reference 的细粒度拒识运行说明](docs/TAXOSAFE_REFERENCE_REFINE.md)。该入口复用已有权重，不需要重训 MaPLe，并明确报告父级路由造成的性能上限。

> 本页保留早期 `TaxoSafe_support_new.yml` 的使用说明。本轮新增的类别判别／成员性分离方案、TRAIN 内部严格留类验证及显式新配置运行命令，见 [TaxoSafe decoupled 运行说明](docs/TAXOSAFE_DECOUPLED.md)。

本分支基于 `taxosafe-v11-dcbs`，新增独立的 `taxosafe_support/` 方法、配置及训练／校准／测试入口。目标仍是 known 到正确叶、near 到正确父、extra 到根。已有图片、数据清单、旧方法入口和历史实验目录继续保留。

这是待真实 GPU 训练验证的实现。代码和 CPU 契约测试不代表四项指标已经达标，也不证明优于历史方法。速度设计采用一次视觉主干前向，但与 v11 同速必须在同一服务器上实测。

## 1. 本次实现与研究设计的对应关系

| 内容 | 本分支的实际实现 |
| --- | --- |
| 图像编码 | 共享一次 MaPLe/CLIP 视觉前向，父层与叶层使用独立的轻量 token 适配和池化；不是分别运行两个完整视觉主干 |
| 训练来源 | 仅干净 known TRAIN 参与梯度训练；真实 near/OOD 只用于既定 development 校准和冻结后的测试 |
| 层级任务 | 查询图片保持不变，分别使用完整支持、移除查询物种、移除查询父类的支持条件 |
| 配对约束 | 移除物种时降低叶接受证据并保护父级证据；移除父类时降低根内接受证据 |
| 对照干预 | 移除无关类别，检查“支持集合变小”本身是否造成拒识；在条件允许时匹配活跃叶类别数量 |
| 候选一致性 | 全局、局部和支持距离证据针对同一个候选节点计算，不拼接不同头各自的最优类别 |
| 拒识监督 | 来自真实 known 查询的支持干预；主方法不使用 v11 的混合特征合成 STOP 池 |
| 解码 | 在根未知、各父类内部未知、各已知叶的联合输出空间中决定停靠深度 |
| 校准 | 固定模型，在既定 development 集上仅选择父层和叶层两个偏置；根偏置固定为零 |

这个实现对之前的“双层编码”建议做了计算预算约束：共享视觉主干与 MaPLe 提示，采用父／叶独立 token 适配与池化。它不等于两套完全独立的提示网络，也不应描述成已复现 ProHOC。开放任务的查询路径保留梯度，能够更新可训练提示及适配模块；支持缓存本身不依靠反向传播更新。

单叶父类无法构造“移除本叶但保留同父其他叶”的条件，相关任务跳过，不用 softmax=1 代替已知性。模型仍使用绝对支持证据；单叶分支和稀疏分支的监督覆盖需要在训练日志中核对。匹配候选数量也不意味着每类实际参考图片数量完全相同，少样本类别应单独解释。

## 2. 数据协议和四项验收指标

默认配置为 `configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_new.yml`，继续引用 v11 已去重 known 清单和已有 unknown Dev/Test 清单。

| 阶段 | 允许读取的拟合／评估数据 |
| --- | --- |
| 梯度训练 | known TRAIN；known 验证用于模型选择，不参加梯度更新 |
| 支持统计 | 仅 known TRAIN；查询内容身份从参考中排除；干预后父统计按剩余支持计算 |
| 校准 | `val_known`、`val_intra`、`val_extra`；不更新网络参数 |
| 测试 | 既定 `test_known`、`test_intra`、`test_extra`；只读取冻结产物，不重新拟合支持统计或偏置 |

`prepro/raw` 是图片存放位置，不是“其中所有图片都可以训练”的授权。不要更换为扫描整个目录的训练方式，不要将 unknown Dev 或 Test 移入 TRAIN。不要使用曾接受真实 unknown/OE 梯度的 v10 checkpoint 作为 known-only 初始化或教师。

| 主指标 | 比较符号 | 目标 |
| --- | --- | ---: |
| known 端到端正确叶准确率 | 严格大于 | 90% |
| near 正确父类回退率 | 大于等于 | 85% |
| extra 根拒识率 | 严格大于 | 90% |
| 开放世界叶输出精确率 | 严格大于 | 90% |

验收使用唯一图片口径。若现有测试仍是 known 450、near 204、extra 272 张唯一图片，则前三项目标分别至少需要正确 406、174、245 张；最终以该次审计实际数量为准。第四项分母是全部集合中被接收到已知叶的图片，不能通过减少未知输入量改善指标。

校准完成和四项达标分别报告。找不到同时满足目标的偏置时，保留可行性／冲突诊断；不能降低目标、查看测试结果后再调整偏置，或把旧版 80%／70%／75% 门槛当成本版成功线。已有测试曾在历史研发中被分析过，应如实披露，不能重新称为全程未查看的独立测试。

## 3. 在 Linux 服务器准备

以下命令在已经同步 `new` 分支代码的项目中执行。若现有目录没有 `.git`，不要在其中执行 `git checkout` 或 `git pull`；先把本分支新增代码同步到该目录，保留原始数据和历史结果。

```bash
conda activate ProTeCt
cd /home/ubuntu/hdd/data/qz/Openset
```

沿用原有 ProTeCt、PyTorch CUDA 和 CLIP 环境。完整训练需要 CUDA；CLIP 预训练权重使用服务器已有缓存，缺失时按原项目权重加载流程获取。新方法从原始预训练初始化训练，不加载历史 v10 的未知监督权重。

如 v11 干净 known 清单尚未生成，执行一次：

```bash
python prepro/build_taxosafe_v11_known_splits.py
```

这是沿用原有身份去重准备脚本，包含 known 测试图片字节身份审计，不是用测试特征训练。它不重新划分 unknown 测试集合；若已有输出与当前数据不一致，应检查原因，不直接覆盖以绕过审计。

## 4. 预检、训练、校准和测试

先运行不下载 CLIP 的 CPU 契约与回归测试：

```bash
python -m unittest discover -s tests -p 'test_taxosafe_support*.py' -v
python -m unittest tests.test_taxosafe_dcbs tests.test_taxosafe_dcbs_pipeline tests.test_taxosafe_dcbs_delivery tests.test_taxosafe_v10_protocol tests.test_taxosafe_suite -q
```

三个新入口均支持 `--preflight`，预检不加载 CLIP，不需要 GPU，按阶段审计相应输入及配置／代码签名。预检自身不执行前序 checkpoint 的完整加载校验；正式校准和测试会检查前序产物及哈希绑定。测试预检也会读取测试图片字节做身份审计，因此仍按下面顺序在方法和校准冻结后执行。

```bash
python train_taxosafe_new.py --trial 1 --seed 1 --variant main --preflight
python -u train_taxosafe_new.py --trial 1 --seed 1 --variant main
```

训练成功结束后再校准：

```bash
python calibrate_taxosafe_new.py --trial 1 --seed 1 --variant main --preflight
python -u calibrate_taxosafe_new.py --trial 1 --seed 1 --variant main
```

冻结校准产物后再测试：

```bash
python test_taxosafe_new.py --trial 1 --seed 1 --variant main --preflight
python -u test_taxosafe_new.py --trial 1 --seed 1 --variant main
```

默认 run 为 `runs/taxosafe_new/main/trial_1/`。不要覆盖旧 `runs/taxosafe_v11_dcbs/` 或其他历史目录，也不要对同一 run 同时启动多个写进程。

中断或修改配置后，使用新的目录，三阶段保持相同参数：

```bash
python -u train_taxosafe_new.py --trial 1 --seed 1 --variant main --run-dir runs/taxosafe_new/main/trial_1_retry
python -u calibrate_taxosafe_new.py --trial 1 --seed 1 --variant main --run-dir runs/taxosafe_new/main/trial_1_retry
python -u test_taxosafe_new.py --trial 1 --seed 1 --variant main --run-dir runs/taxosafe_new/main/trial_1_retry
```

可选短调试使用独立目录：

```bash
python -u train_taxosafe_new.py --trial 1 --seed 1 --variant main --debug
```

debug 产物不能用于校准或测试。调试还可能包含支持统计和验证遍历，不是只执行两个 batch 后立即返回。

默认训练上限 120 epoch、warmup 5 epoch、开放任务渐增 5 epoch、patience 25。配置保留 `center_crop` 输入方式，同时实际启用训练随机翻转；评估是确定性的。`letterbox` 可用于已知验证上的独立输入方式对照，不能通过最终测试成绩来选择。中心裁剪是否损伤形态必须查看图片或做对照，代码变化本身不能证明原因。

模型只按 known 验证叶准确率优先、结构化负对数似然次优选择，默认须经过 warmup 及至少 5 个开放任务 epoch 才能入选。训练支持库在开始时、每个 epoch 后及最终 checkpoint 上重建；每轮产物供下一轮复用，不对每个干预重跑视觉主干。

| 目录／文件 | 查看用途 |
| --- | --- |
| `training/config.json`、`inputs.json` | 有效配置、输入身份审计、签名 |
| `training/train.jsonl` | 每轮损失、任务覆盖、known 验证、checkpoint 入选、训练／支持遍历／验证计时 |
| `training/model_cost.json` | 总参数量、可训练参数量及速度测量边界 |
| `training/best.pth`、`support.pth`、`completed.json` | 冻结模型、训练支持库及完成记录 |
| `calibration/development_scores.jsonl`、`development_predictions.jsonl` | 开发集分数和冻结路由后的逐图片输出 |
| `calibration/router.json`、`validation_report.json` | 两个深度偏置、可行性和四项指标；来源留一诊断在 router 中 |
| `calibration/development_metrics.json`、`completed.json` | 完整指标及校准完成／目标通过的独立标志 |
| `test/predictions.jsonl`、`metrics.json`、`metrics_all_rows.json` | 逐行预测、唯一图片主指标、全部清单行辅助指标 |
| `test/summary.json`、`gates.json`、`completed.json` | 四项验收、分组诊断和冻结测试完成记录 |
| `calibration/inference_timing.json`、`test/inference_timing.json` | 各阶段推理计时，不是与 v11 的同机对照 |

`calibration/completed.json` 的 `fit_completed=true` 只表示偏置拟合已冻结；`targets_passed` 才表示四项目标是否同时通过。若搜索网格无共同可行点，程序仍冻结其选定的有限偏置并记录 `targets_passed=false`，允许后续按冻结方案进行诊断性测试。这不是通过验收。真正的数据／数值异常会终止阶段，部分校准异常会写 `failed.json`；不要绕过异常手动生成完成记录。

## 5. 消融与仍需开展的研究验证

| `--variant` | 含义 |
| --- | --- |
| `main` | 完整支持干预、配对约束、父／叶 token 适配和局部证据 |
| `no_pair` | 去除配对干预约束，保留各条件任务监督 |
| `no_parent` | 去除父类留出任务；并非去除所有父级监督，near 配对中的父证据稳定约束仍保留 |
| `shared` | 父／叶共享适配表示，检验独立层级表示的作用 |
| `no_local` | 去除局部匹配，保留全局支持证据 |
| `classification` | 分类训练消融；不是正式 B0，也不是外部 HND／ProHOC 复现 |

各 variant 使用独立 run 并从头训练，不交换 checkpoint、支持库或 router。先验证 main 的训练／冻结流程，再开展预先选定的消融。多种子采用 `--trial 2 --seed 2`、`--trial 3 --seed 3` 等独立目录，报告全部结果，不按测试结果挑最好 seed。

本分支未实现“整物种或整父类在该折微调阶段完全留出”的独立训练折。当前支持移除任务中的查询类别仍可能被图像编码器在其他训练 batch 中见过，不能把它写成编码器从未见过该类的真实未知验证。HND／ProHOC 的公平外部复现、完整训练留类折和标准化训练预算比较仍需后续实验。

局部提示、留类任务、近邻距离和层级概率已有相关工作。本方法的候选贡献是支持条件与配对停靠约束的联合学习，是否有独立增益必须由消融验证；不能将多个已有机制直接包装成已证实的新创新。

如原 v11 训练产物仍可通过原实现的哈希验证，可在同一 GPU、相同 known 验证图像、相同 batch size 下比较端到端评分时间：

```bash
python tools/benchmark_taxosafe_new.py --run-dir runs/taxosafe_new/main/trial_1 --v11-run-dir runs/taxosafe_v11_dcbs/main/trial_1 --batch-size 16 --repeats 5 --warmup 1
```

旧实验实际位于其他目录时修改 `--v11-run-dir`。输出为新 run 的 `speed_comparison.json`，默认拒绝覆盖；需要重复测量时使用新的 `--output` 路径。测量仅使用 known 验证集，不读取测试图片或改变阈值。`new_over_v11_time` 大于 1 表示本次新方法评分较慢；该比较不等于训练同速，训练速度还需结合同机 `train.jsonl` 中各阶段时间判断。旧产物缺失或签名不匹配时，不伪造 v11 对照时间。

## 6. 打包结果供分析

在阶段结束或失败后，可以打包已产生的诊断；至少需要该 run 的 `training/config.json`。默认打包 trial 1：

```bash
python tools/pack_taxosafe_new_review.py --run-dir runs/taxosafe_new/main/trial_1
```

脚本打印生成的 `.tar.gz` 完整路径，把这个文件发回来即可。重跑目录需传实际 `--run-dir`。也可指定输出文件，已有同名包会被拒绝覆盖：

```bash
python tools/pack_taxosafe_new_review.py --run-dir runs/taxosafe_new/main/trial_1 --output runs/taxosafe_new_trial1_review.tar.gz
```

包中仅包含该 run 内的文本配置、审计、日志、分数、预测、指标和门禁诊断，不包含图片、模型权重、二进制支持缓存或 run 外的数据清单。`archive_manifest.json` 给出每个成员的字节数及 SHA256。逐图片预测含源路径／物种等分析信息，保留这些字段是为了定位具体物种和来源的问题。

若正式训练尚未开始、仅预检失败，请直接提供预检终端报错；不要为了打包伪造完成标志。也不要因 gate 为 false 而删除诊断目录，失败结果同样需要完整分析。
