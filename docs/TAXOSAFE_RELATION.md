# TaxoSafe relation：论文依据、实现与运行说明

> 后续实验更新（2026-10-03）：relation trial 1 四项 TEST 指标为 90.22% / 56.37% / 54.78% / 80.24%，均低于 reference v3。本页保留该实验的复现说明；新的独立精修路线见 [冻结 reference 运行说明](TAXOSAFE_REFERENCE_REFINE.md)，不继续把 relation 当作已经验证更优的基线。

本轮依据 `trial_1_cuda_fix` 的 TRAIN/DEV 诊断实现独立 `support_relation_v4` 配置。原结果分析及全部逐物种表见 [实验复核](TAXOSAFE_RELATION_RESULT_REVIEW.md)，机器可读审计见 [审计JSON](REFERENCE_TRIAL1_AUDIT_20261003.json)。reference v3 的94.00% / 60.78% / 80.88% / 83.10%不是新relation模型的结果。

## 1. 为什么改变匹配证据

DEV已知保留与近域拒识所需叶阈值相互冲突，父阈值也存在近域保留和域外根拒冲突；同时第50轮训练配对损失已很低。因此本轮保留训练轮数、学习率、损失权重、数据清单、两阈值校准及验收门槛，改变配对证据的表达和困难负对的监督。

| 一手论文 | 原方法及可借鉴点 | 本轮采用范围与限制 |
| --- | --- | --- |
| [Learning to Compare: Relation Network for Few-Shot Learning，CVPR 2018](https://arxiv.org/abs/1711.06025)，[作者代码](https://github.com/floodsung/LearningToCompare_FSL) | 比较查询/支持的完整特征图，以非线性关系网络学习匹配，而非只预设固定距离 | 主借鉴：小型共享比较器读取投影后的通道差异及乘积。这里使用对称特征、BCE与层级支持协议，不是原文拼接特征图+MSE的复现，也没有借论文结果保证OSR提升 |
| [From Coarse to Fine-Grained Open-Set Recognition，CVPR 2024](https://langnico.github.io/fine-grained-osr/)，[作者代码](https://github.com/langnico/osr-coarse-to-fine) | 层级监督对粗粒度与细粒度OSR的作用不同；原文用粗层分类梯度反转抑制熟悉性陷阱 | 保留同父异种独立负对组和独立父/细粒度表示。没有实施HA/GRL；本任务还要求可靠父类回退，不能直接破坏共享父类表征 |
| [Multi-Similarity Loss with General Pair Weighting，CVPR 2019](https://arxiv.org/abs/1904.06627) | 度量学习中的信息对挖掘与权重分配，会影响学习效果 | 借鉴困难负对思想，具体实现为本项目每参考物种top-k负对BCE；不是MS loss公式，也不声称该论文直接验证了本项目拒识方法 |
| [Class-Specific Semantic Reconstruction for Open Set Recognition，TPAMI 2023（在线2022）](https://arxiv.org/abs/2207.02158)，[作者代码](https://github.com/xyzedd/CSSR) | 每类语义自编码器通过重构残差衡量归属，提供点原型之外的证据 | 支持保留通道残差信息的动机。未加入每类AE；本项目归一化CLIP特征不能照搬原文依赖特征幅度的归一化误差。永久类别头还需严格区分支持移除和真实未见类验证 |
| [DeepEMD，CVPR 2020](https://arxiv.org/abs/2003.06777)，[作者代码](https://github.com/icoz69/DeepEMD) | 密集局部描述子之间求最优运输匹配，参考区域权重用于降低背景影响 | 借鉴局部结构比较；本轮采用双向最近局部对应及通道残差，没有求解EMD/OT，不应标成DeepEMD复现 |
| [Adversarial Reciprocal Points Learning for Open Set Recognition，TPAMI](https://arxiv.org/abs/2103.00953)，[作者代码](https://github.com/gary23ai/ARPL) | 通过互反点及边界约束建模类外空间；增强版本生成混淆样本 | 可作后续独立基线；本轮不替换整个分类头、不加入生成器，也不把普通困难负对称为ARPL |

选择关系比较器是因为它能继续使用可移除的真实支持证据，不需要新图像主干、测试类名称或真实未知图像梯度。容量增加仍可能导致过拟合，必须通过对照和严格类别留出测量，不能仅看训练损失降低。

## 2. 实际实现

### 2.1 保留通道信息的对称关系比较器

`relation.py` 的 `RelationMatcher` 在父、叶深度各使用一个共享于所有类别的头。默认把512维向量投影为32维并归一化，输入包括：

1. 全局余弦、双向局部覆盖均值、双向局部覆盖最小值。
2. 全局投影向量的逐通道绝对差与逐通道乘积。
3. 局部匹配投影向量的逐通道绝对差与逐通道乘积。

局部对应由原始归一化局部向量的余弦最近匹配确定；双向各占一半，完全并列的匹配等权，避免仅由参考排列决定证据。投影后的残差保留差异所在通道。输入共 `3+4×32=131` 维，后接32维隐藏层和一个logit；末层零初始化，初始仍为原有余弦先验。

不输入物种ID、父类ID或支持数量。缺少局部特征时局部坐标通道置零；余弦分支仍可运行。TRAIN编码缓存显式detach，但比较器共享投影在查询端及参考端都可学习。局部比较按查询/参考分块，减少单个临时张量大小；整批关系图在五个支持干预之间复用。

继续在池化前排除查询自身内容、被移除物种/父类，叶层取最高两个有效参考分数平均，父层先在每个子物种内聚合、再对子物种等权平均。层级输出、原始四个证据头和校准接口保留。

### 2.2 与接纳池化对应的困难负对

主配置设置 `support.pair_negative_topk: 2`。在每个查询、每个负参考物种内部，选最高两个有效关系logit监督；不在整个支持库里统一取top-k。父层跨父负对、叶层同父异种负对、叶层跨父负对分别处理。

所有有效正对保留，原有按参考物种、监督组和查询等权的规则保留。自身哈希和无效支持先屏蔽；不足两个取实际数量，空组为有限的零损失。日志里的负对曝光数对应实际选中的负对，warmup及零权重时有效曝光为零。

这让少量高分负参考不会被同物种内大量容易负对稀释，但并不创造真实未知数据，也不能解决所有未见形态泛化。困难对也可能包含标注噪声，故保留全部正对、原损失权重和既有训练预算，并提供原平均负对损失的对照。

### 2.3 选模规则和诊断

主配置 `training.selection: candidate` 采用与membership部署一致的身份规则：父排名argmax，再在该父内取叶排名argmax。仅在known验证集计算 `candidate_leaf_accuracy`，以其为首选指标、原 `structured_nll` 为第二指标。

候选准确率没有应用拒识阈值，不能称为校准后端到端准确率。阈值仍在训练完成后只用DEV校准，真实未知图像不进入选模梯度或epoch选择。旧配置默认 `selection: text`，返回字段和旧选模行为保持原样。

新日志 `diagnostics` 单独记录父/细粒度局部token非对角余弦，以及正确叶成员分数减最困难兄弟叶成员分数的间隔。它们已detach，不加入损失或选模。余弦长期接近1可提示局部池冗余；训练间隔增大仍不能代替真实未知验证。

## 3. 配置与可行性

| 配置 | 比较器 | 负对监督 | 选模 | 用途 |
| --- | --- | --- | --- | --- |
| `TaxoSafe_support_reference.yml` | 原三标量 | 原平均BCE | text | 保留reference v3算法路径 |
| `TaxoSafe_support_relation_only.yml` | 新关系比较器 | 原平均BCE | text | 优先用于隔离比较器贡献的对照 |
| `TaxoSafe_support_relation.yml` | 新关系比较器 | 每物种top-2困难负对 | candidate | 本轮完整主方案 |

完整主方案和relation-only同时改变的是负对监督及选模。若要分别归因，可从relation-only复制配置，仅增加 `pair_negative_topk: 2`，再另建一份仅设 `training.selection: candidate`；均需独立run，不能原地改已训练run配置或依据TEST挑最佳组合。

默认额外可训练参数为40,960，仍一次查询图像主干前向。没有测得新GPU耗时/显存，不能由参数少推断速度完全不变。训练仍最多120轮、warmup5轮、patience25；支持上限仍每物种8张。主配置保持原CenterCrop与采样，以免把数据处理变化混入比较器实验；letterbox、代表性支持采样应作为独立消融。

## 4. 更新与快速检查

服务器目录不是git仓库也可运行。下载 [new分支完整源码ZIP](https://github.com/Jonathan-519/Openset/archive/refs/heads/new.zip)，放在 `/home/ubuntu/hdd/data/qz/Openset-new.zip`。现有数据、模型结果和CLIP缓存保留。

可使用 [旧运行说明第2节](TAXOSAFE_REFERENCE.md#2-更新服务器代码与检查) 中的Python合并脚本：该脚本适用于本ZIP，自动备份有变化的源码，跳过原图、清单、权重与runs。仅使用其中的合并脚本；本轮校验和运行命令如下，不再执行旧reference配置的训练指令。

```bash
conda activate ProTeCt
cd /home/ubuntu/hdd/data/qz/Openset
sha256sum -c SUPPORT_RELATION_FILES.sha256
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
python -m unittest tests.test_taxosafe_support_relation_pipeline.RelationCudaContract.test_cuda_relation_training_loss_and_backward -v
```

这个GPU检查无需下载CLIP或读取真实图像，覆盖五种支持干预、关系头、困难负对、单子类计数和反向传播。应显示 `OK`，不能把 `skipped` 当作GPU成功。完整软件回归可用 `python tools/run_unit_tests.py --quiet`。GPU检查通过仍不证明真实数据已训练完成。

## 5. 训练、校准、测试和打包

新结构需从原始CLIP初始化，使用独立目录；不从 `reference/trial_1_cuda_fix` 续训，也不把旧checkpoint/router搬到新目录。旧结果已完成，无需删除。更新源码会改变签名；历史run跨阶段读取使用对应旧源码提交 `54dff0e`。

```bash
cd /home/ubuntu/hdd/data/qz/Openset
cfg=configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_relation.yml
run_dir=runs/taxosafe_new/relation/trial_1

python train_taxosafe_new.py --config "$cfg" --trial 1 --seed 1 --variant main --run-dir "$run_dir" --preflight &&
python -u train_taxosafe_new.py --config "$cfg" --trial 1 --seed 1 --variant main --run-dir "$run_dir" &&
python -u calibrate_taxosafe_new.py --config "$cfg" --trial 1 --seed 1 --variant main --run-dir "$run_dir"
```

先读取 `calibration/validation_report.json`，区分 `fit_completed` 与 `targets_passed`。四门槛仍为known >90%、near正确父回退≥85%、extra根拒>90%、开放叶精确率>90%。失败报告也是有价值的诊断，不应通过改测试阈值伪装为通过。

规则冻结后执行测试：

```bash
python -u test_taxosafe_new.py --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_relation.yml --trial 1 --seed 1 --variant main --run-dir runs/taxosafe_new/relation/trial_1
```

打包命令必须显式使用本轮路径：

```bash
python tools/pack_taxosafe_new_review.py --run-dir runs/taxosafe_new/relation/trial_1
```

若某阶段报错，保留终端和该run诊断；已有 `training/` 不能覆盖启动，另选新目录。不要更改旧运行目录的签名文件。

relation-only对照使用 `TaxoSafe_support_relation_only.yml` 和 `runs/taxosafe_new/relation_only/trial_1`；训练/校准/测试三个阶段始终使用同一配置、seed和run目录。

## 6. 严格类别留出验证

支持移除episode的编码器训练过被移除的类别，所以不是严格未见类实验。现有严格验证器会从TRAIN中完全排除留出类，独立重训模型和支持库，inner-known验证也只来自TRAIN。先查看计划：

```bash
python tools/validate_taxosafe_support_holdout.py --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_relation.yml --seed 1 --kind species --max-folds 1 --run-dir runs/taxosafe_new/relation_holdout_species --preflight
python tools/validate_taxosafe_support_holdout.py --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_relation.yml --seed 1 --kind parent --max-folds 1 --run-dir runs/taxosafe_new/relation_holdout_parent --preflight
```

去掉 `--preflight` 才会执行真实GPU重训，每折相当于独立训练任务。一个fold只能检查流程；研究结论需覆盖预定完整fold及多种子，不能把一个fold当总体结果。

严格验证器仍以 `uncalibrated_joint_argmax` 输出诊断正确率，另导出与生产一致的候选及四个原始头，并明确 `membership_thresholds_applied=false`。因此不能把这些未校准fold正确率直接当作生产两阈值路由的验收分数。

## 7. 已验证和未验证

软件验证及限制见根目录 `SUPPORT_RELATION_VERIFICATION.json`。旧v1、prototype v2、reference v3与真实 `54dff0e` 源码逐项比较，初始化、五种支持干预、损失、计数和梯度保持一致。CPU合成数据流程覆盖训练、校准、测试和严格类别留出；CUDA专项测试保留，当前执行环境无GPU，明确跳过。

新relation模型尚无真实图像训练和TEST指标。本次实现增加可用证据和可审计对照，不承诺细粒度门槛必然通过。首先应验证relation-only相对旧方法的DEV/严格留出表现，再报告完整方案及独立消融，避免把多个修改的效果混为一项创新。
