# TaxoSafe decoupled：类别判别与支持成员性分离

本方案沿用 `train_taxosafe_new.py`、`calibrate_taxosafe_new.py`、`test_taxosafe_new.py`，通过新配置显式启用。原入口、原默认配置 `TaxoSafe_support_new.yml`、原有算法模块及数据清单继续保留。运行本文方案时，每个阶段都必须传入 `TaxoSafe_support_decoupled.yml`，不能只在训练时指定。

新方案应从原始 CLIP 初始化重新训练，不能接着旧 `trial_1` checkpoint 继续训练。输出使用独立的 `runs/taxosafe_new/decoupled/trial_1/`；旧 checkpoint、校准结果、测试预测和审查包都应保留。当前没有真实 CUDA 训练结果，不能据此声称精度、拒识率或训练速度已经改善。

## 1. 修改目的与实际方法

旧支持条件方法将同一个候选匹配分数同时用于“最像哪个类”与“是否属于已支持类别”。新配置设置 `support.decoupled: true`，为父候选和叶候选分别建立独立的成员性证据，使相对类别判别和绝对支持接受分开学习。

| 证据 | 要回答的问题 |
| --- | --- |
| 父类身份分布 (q_p) | 在当前候选父类中，相对最符合哪个父类？ |
| 父类成员性 (r_p) | 该图是否有足够证据属于候选父类 (p)？ |
| 父内叶身份分布 (t_{c\mid p}) | 在该父类的候选物种中，相对最符合哪个物种？ |
| 叶成员性 (a_c) | 该图是否有足够证据属于候选物种 (c)？ |

成员性与身份证据始终对应同一个候选。对物种 (c) 及其父类 (p)，联合输出组织为：

\[
P(c\mid x)=q_p r_p t_{c\mid p}a_c,
\]

\[
P(U_p\mid x)=q_p r_p\sum_{c\in p}t_{c\mid p}(1-a_c),
\]

\[
P(U_{root}\mid x)=\sum_p q_p(1-r_p).
\]

没有活跃支持候选时，概率质量进入根未知。单叶父类的叶 softmax 即使为 1，也仍要通过独立成员性证据，不能自动接收为已知物种。这些量是模型组织评分的方式，不等于真实未知概率已经校准准确。

图像仍只进行一次共享 MaPLe/CLIP 视觉前向，父／叶 token 适配与池化在该表示上工作。新增小型成员性匹配器不需要为每种支持干预重跑视觉主干；训练是否与 v11 同速仍须在同一 GPU 上实测。已知训练查询的开放任务损失保留到可训练提示及适配模块的梯度，参考库来自 TRAIN，不从真实 near/OOD 学习参数。

新增监督包含以下两部分，并保留原有分类、支持干预、配对和锚定损失：

| 配置中的损失项 | 训练含义 |
| --- | --- |
| `membership_parent`、`membership_leaf` | 对活跃父／叶候选做正负均衡的二元成员性监督。正确类别被移除时，剩余候选可以全部为负；不虚构正类 |
| `parent_cross_species` | 父表示将同父不同物种作为正对、不同父类作为负对，学习跨物种的共同证据 |
| `leaf_sibling` | 叶表示将同物种的不同图片作为正对、同父兄弟物种作为负对，学习父内区分 |

对比监督排除同内容图片及自身配对，只有同时存在有效正对与负对的查询才产生该项损失；单叶父类或不满足条件的 batch 不伪造配对。训练日志中的有效任务数量和表示配对数量用于检查覆盖，不能把配置了非零权重等同于实际获得了充分监督。

新配置的两个成员性权重各为 1.0，父跨物种和叶兄弟对比权重各为 0.1。默认 `hidden_dim=32` 时两个新增成员性 MLP 共增加 322 个参数；批内对比另有相似度计算，参数量很小不等于总耗时必然不变。训练仍为最多 120 epoch，warmup 5 epoch、开放任务渐增 5 epoch、patience 25。

## 2. 数据边界与验收目标

| 用途 | 允许的数据与行为 |
| --- | --- |
| 梯度训练与支持统计 | 原有干净 known TRAIN；不得将真实 near/OOD、验证或测试图片并入梯度训练 |
| 主训练模型选择 | known 验证集；不能依据最终测试选择结构、checkpoint 或 seed |
| 主运行校准 | 既定 `val_known`、`val_intra`、`val_extra`，只选择固定评分上的校准参数 |
| 最终测试 | 既定三个 test 清单及冻结 checkpoint、支持统计、router；不拟合任何参数 |

图片都位于 `prepro/raw` 不表示可以扫描整个目录作为训练集。保持已有 TRAIN／Dev／Test 清单与内容身份审计，支持库排除查询本身的内容身份，干预后按剩余参考重新计算统计。

四项硬指标及比较符号不变，主指标按唯一图片计算：

| 指标 | 目标 |
| --- | ---: |
| known 端到端正确叶准确率 | **>90%** |
| near 正确父类回退率 | **≥85%** |
| extra 根拒识率 | **>90%** |
| 开放世界叶输出精确率 | **>90%** |

逐物种、逐父类和逐来源结果用于解释错误；不能用平均总准确率替代四项独立验收，也不能通过减少测试未知图片来提高叶输出精确率。历史测试已被研发分析这一事实仍需披露。

## 3. 在原 Linux 目录更新与检查

服务器项目路径为 `/home/ubuntu/hdd/data/qz/Openset`，即使目录没有 `.git` 也能运行。可从仓库 `new` 分支下载完整源代码 ZIP，在项目之外解压后合并代码；保留本地 `prepro/raw/`、`prepro/data/`、`runs/` 和模型权重。不要删除现有项目后用 ZIP 整体替换，也不要使用会删除目标端文件的同步选项。

完整源码下载：[new 分支 ZIP](https://github.com/Jonathan-519/Openset/archive/refs/heads/new.zip)。这是源代码包，不是本地训练数据与历史运行结果的备份。更新前应停止正在使用待更新代码的训练／校准进程，旧实验要用其对应的代码版本读取。

```bash
conda activate ProTeCt
cd /home/ubuntu/hdd/data/qz/Openset
sha256sum -c SUPPORT_DECOUPLED_FILES.sha256
python tools/run_unit_tests.py --quiet
```

本次软件验证记录见 `SUPPORT_DECOUPLED_VERIFICATION.json`，全仓 224 项 CPU 测试通过。`SUPPORT_NEW_VERIFICATION.json` 保留的是上一版历史验证记录。单元测试中部分合成 gate=false 用于检查失败路径，以测试最终 `OK` 为准；真实实验的 `targets_passed` 必须单独检查。

CPU 测试验证协议、概率组织、梯度连接和软件行为，不能替代 GPU 学习效果验证。完整训练仍需要原项目可运行的 CUDA 环境和 CLIP 预训练权重。若已有 v11 干净 known 清单，不必重新生成；缺失时才运行：

```bash
python prepro/build_taxosafe_v11_known_splits.py
```

该准备步骤可能读取 known 测试图片字节进行身份去重，不提取测试特征或调参。已有清单与当前图片身份不一致时应检查原因，不绕过审计覆盖输出。

## 4. 主实验：完整命令

下面始终使用新配置和新 run 目录。`--preflight` 按阶段检查输入，不加载 CLIP；正式阶段还会校验前序产物和签名。

### 4.1 训练

```bash
python train_taxosafe_new.py --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_decoupled.yml --trial 1 --seed 1 --variant main --run-dir runs/taxosafe_new/decoupled/trial_1 --preflight
python -u train_taxosafe_new.py --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_decoupled.yml --trial 1 --seed 1 --variant main --run-dir runs/taxosafe_new/decoupled/trial_1
```

看到 `training/completed.json` 并成功退出后，再开始校准。不要把短调试 `--debug` 产物用于校准／测试。中断后重跑应使用全新的目录，例如 `runs/taxosafe_new/decoupled/trial_1_retry`；三个阶段传入同一个新目录，不能仅删除完成记录后覆盖旧文件。

### 4.2 校准

```bash
python calibrate_taxosafe_new.py --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_decoupled.yml --trial 1 --seed 1 --variant main --run-dir runs/taxosafe_new/decoupled/trial_1 --preflight
python -u calibrate_taxosafe_new.py --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_decoupled.yml --trial 1 --seed 1 --variant main --run-dir runs/taxosafe_new/decoupled/trial_1
```

校准结束后，先查看开发集报告及完成标志：

```bash
python -m json.tool runs/taxosafe_new/decoupled/trial_1/calibration/validation_report.json
python -m json.tool runs/taxosafe_new/decoupled/trial_1/calibration/completed.json
```

`fit_completed=true` 表示校准已冻结，不能读作四项目标已达标；查看独立的 `targets_passed` 及各项指标。不要在未检查开发集失效原因时反复运行最终测试，也不要依据最终测试结果回调阈值。

新配置采用 `calibration.policy: known_first`，仅在固定 development 分数上选择父层、叶层两个偏置。选择顺序是：先选四项目标共同通过的点；若没有共同通过点，但存在 known 端到端准确率 **>90%** 的网格点，就只在这些保住 known 的点中选择；若连 known 目标也无法满足，则保存尽力选择结果并明确标注。原配置缺省的 `balanced` 策略保持原行为。

| `status` | 含义 |
| --- | --- |
| `feasible` | 本次网格找到四项目标共同通过的点 |
| `best_effort_known_preserved` | 四项未共同通过，但选定点保住 known 目标 |
| `best_effort_known_unavailable` | 网格内 known 目标也不可达，选定点仍未通过完整验收 |

查看 `selection_diagnostics.sampled_known_feasible_count` 与 `selected_known_gate_passed`，区分“选中了损伤 known 的点”与“当前候选网格中不存在 known 通过点”。`fixed_score_feasibility` 进一步报告候选正确性上界和两个偏置的必要条件冲突。其中 `continuous_infeasibility_proven=true` 只表示固定开发分数、当前联合解码规则下存在不可行证明，不能推出其他模型都不可能达标；为 false 也不代表连续偏置一定可行。

### 4.3 冻结测试

模型与校准方案确定、前序阶段完整之后执行：

```bash
python test_taxosafe_new.py --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_decoupled.yml --trial 1 --seed 1 --variant main --run-dir runs/taxosafe_new/decoupled/trial_1 --preflight
python -u test_taxosafe_new.py --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_decoupled.yml --trial 1 --seed 1 --variant main --run-dir runs/taxosafe_new/decoupled/trial_1
```

测试只应用冻结产物，不使用测试图片更新支持库、成员性网络或偏置。测试预检会读取本阶段清单及图片身份，因此也放在模型和校准冻结之后。

其他种子仍须显式使用同一新配置；例如 seed 2 的三个命令均应设置 `--trial 2 --seed 2 --run-dir runs/taxosafe_new/decoupled/trial_2`，并保留 `--config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_decoupled.yml --variant main`。不得遗漏配置后误跑旧默认方案。

## 5. TRAIN 内部严格留类验证

`tools/validate_taxosafe_support_holdout.py` 为每个选定折从原始 CLIP 重新训练。被留出的整物种／整父类图片不参与该折梯度、支持库或 checkpoint 选择，其候选文本也不参与该折可训练文本编码。这区别于训练后从支持库临时删类。原始分类树编号和原始数据清单不变。

工具只读取 known TRAIN：剩余类别的 TRAIN 图片再按内容身份确定性划分内部训练和 known 验证；held-out 类在模型选定后进行一次无校准评估。它不读取原 `val_known`、真实 near/OOD Dev 或最终 Test，也不能把产生的折模型交给正式未知校准或锁定测试入口。

先查看计划，不进行训练：

```bash
python tools/validate_taxosafe_support_holdout.py --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_decoupled.yml --variant main --seed 1 --kind both --max-folds 1 --run-dir runs/taxosafe_new/decoupled_holdout_smoke/seed_1 --preflight
```

确认资源预算后，可以先运行一个按 seed 确定的折检查完整流程：

```bash
python -u tools/validate_taxosafe_support_holdout.py --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_decoupled.yml --variant main --seed 1 --kind both --max-folds 1 --run-dir runs/taxosafe_new/decoupled_holdout_smoke/seed_1
```

单折只验证被选中的那一种留类任务，不代表物种和父类两种任务都覆盖。要运行全部合格折，使用不同的新目录并去掉 `--max-folds 1`：

```bash
python tools/validate_taxosafe_support_holdout.py --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_decoupled.yml --variant main --seed 1 --kind both --run-dir runs/taxosafe_new/decoupled_holdout/seed_1 --preflight
python -u tools/validate_taxosafe_support_holdout.py --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_decoupled.yml --variant main --seed 1 --kind both --run-dir runs/taxosafe_new/decoupled_holdout/seed_1
```

每折需要完整训练一套模型，全部折的成本会显著高于一次主训练。也可用 `--kind species` 或 `--kind parent` 限定类型，使用可重复的 `--fold-id` 选择预检中列出的合格折；`--max-folds 0` 表示全部合格折，不表示不运行。默认 `--known-val-fraction 0.2`、`--min-train-per-leaf 2`；样本不足的类别可能没有内部 known 验证覆盖，计划会明确列出。

物种折要求该父类仍有其他已知兄弟物种，因此单叶父类不生成物种折，但可参加父类留出。输出位于指定目录的 `holdout/`：

- `plan.json`：所选折、各图片身份与角色划分、配置及源码签名。
- `summary.json`、`completed.json`：两种任务的微平均、折宏平均和完成记录。
- `species_编号/` 或 `parent_编号/`：每折 `fold.json`、独立 `training/`、`predictions.jsonl`、`metrics.json`、`completed.json`。

`heldout_correct_rate` 在物种折表示正确父类回退率，在父类折表示根拒识率；解码是未经真实未知开发集校准的联合 argmax。它是 known TRAIN 内部的类间迁移诊断，不替代真实 near/extra 测试指标，也不直接等同于四项正式验收。

## 6. 结果文件与审查包

| 文件 | 用途 |
| --- | --- |
| `training/config.json`、`inputs.json` | 有效配置、输入权限及身份审计 |
| `training/train.jsonl` | 损失、任务覆盖、已知验证、模型选择和每轮计时 |
| `training/model_cost.json` | 参数量及计算设计说明 |
| `training/best.pth`、`support.pth`、`completed.json` | 冻结权重、TRAIN 支持库及完成凭据 |
| `calibration/router.json`、`validation_report.json` | 校准策略、偏置搜索、四项指标及可行性诊断 |
| `calibration/development_scores.jsonl`、`development_predictions.jsonl` | 开发集逐图证据和最终输出 |
| `test/metrics.json`、`summary.json`、`gates.json` | 唯一图片主指标、分组结果与四项验收 |
| `test/metrics_all_rows.json`、`predictions.jsonl` | 全清单行辅助统计和逐图审查 |

打包命令必须指定本次 decoupled 目录，否则打包器的默认值仍指向早期 `main/trial_1`：

```bash
python tools/pack_taxosafe_new_review.py --run-dir runs/taxosafe_new/decoupled/trial_1
```

脚本打印输出 `.tar.gz` 路径。包中包括该 run 内的文本日志、配置、证据、预测和指标，不包含图片、权重或二进制支持库；附逐成员 SHA256。校准 gate 为 false 的诊断同样应保留并打包，不删除失败运行。

严格留类总目录没有主运行的 `training/config.json`，不要直接把它传给该打包器。先提供 `holdout/plan.json`、`holdout/summary.json`；需要某一折详细分析时，将 `--run-dir` 指向该折实际的 `holdout/species_编号/` 或 `holdout/parent_编号/` 目录即可。

## 7. 兼容范围与验证边界

- 保留旧主入口名称、默认配置、数据路径和旧算法模块；不指定新配置的命令继续代表旧默认方案。
- 新增成员性模块的参数结构与旧 checkpoint 不同。新方案需要重新训练，不能把旧权重或 router 移入新 run。
- 实现代码、配置、训练支持和 router 受签名绑定。更新代码后，旧 run 可能被正确地拒绝加载；旧算法可以在当前代码上创建新 run，但旧完成凭据并不因此自动兼容。此前 `new` 版本产物需要继续跨阶段运行时，使用对应的 `1d7e4e4` 历史代码；其他历史实验使用各自绑定版本。不要覆盖旧 `training/config.json`、`completed.json` 或 router，也不能删除签名检查来强行读取。
- 仍使用 known-only 梯度训练；真实 near/OOD 参与校准与否的权限不改变。
- 单次共享视觉前向是计算设计，不是速度实测结论。真实 GPU 训练耗时、显存、闭集准确率与未知拒识率本轮尚未验证。
- HND／ProHOC 等外部方法未由本次工程改动自动完成公平复现；候选贡献仍需同协议基线、消融与多种子实验支持。
