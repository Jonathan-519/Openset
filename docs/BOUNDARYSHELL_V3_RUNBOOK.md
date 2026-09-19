# TaxoLocal-v3 BoundaryShell：安装、训练、校准与回传

实现基于 Jonathan-519/Openset 的 master 提交 52501c2b3125d28c17c64aff641538e339122da8。
GitHub 分支：codex/taxolocal-v3-boundaryshell。

这是可运行的研究实现，不是已验证达到 >90% 的模型。真实 CUDA 训练尚未在交付环境运行。
旧 TaxoLocal-v2 配置引用的清单在当前仓库中缺失，默认使用当前 v9 rebuild 的 23 类数据。
旧测试集已经被查看过，新结果必须称为 development comparison，不能称为新的盲测结果。

## 安装

将 taxolocal_v3_boundaryshell_20260919.tar.gz 上传到 /home/ubuntu/hdd/data/qz/，执行：

```bash
cd /home/ubuntu/hdd/data/qz/Openset/
tar --no-same-owner -xzf ../taxolocal_v3_boundaryshell_20260919.tar.gz -C .
sha256sum -c BOUNDARYSHELL_V3_FILES.sha256
python -m unittest tests.test_boundaryshell_v3
```

压缩包根目录就是项目相对路径，不包含额外的 Openset 外层目录。只增加独立 v3 文件，不覆盖
旧训练器、旧 router、图片、清单或实验结果。同名 v3 文件会由解压覆盖。
保留服务器已经可以运行原项目的 CUDA PyTorch/torchvision，不要安装 CPU 版覆盖它们。
新增代码没有引入其他模型库。若原项目运行环境缺少依赖，可安装：

```bash
python -m pip install pyyaml numpy pandas pillow scikit-learn tensorboardX tqdm ftfy regex termcolor
python -c "import torch, torchvision; print(torch.__version__, torch.cuda.is_available())"
```

Python 建议 3.10+、PyTorch 2.0+，使用与 CUDA 版本匹配的 torchvision。首次 MaPLe 初始化沿用
仓库原来的 CLIP ViT-B/16 权重缓存/下载方式，不使用额外真实 OOD/OE 数据。

## 数据准备：执行一次，可重复校验

```bash
bash tools/run_boundaryshell_v3.sh prepare 1
bash tools/run_boundaryshell_v3.sh preflight 1
```

新清单位于 `prepro/data/Zooplankton_BoundaryShell_v3/`。
不移动/删除图片，不改变 test_known/test_intra/test_extra 清单。
按 test > val > train 优先级排除 SHA256 完全重复的训练/验证图片；仅为排查污染读取测试图片哈希，
测试特征、预测及指标不参与训练和校准。
验证样本不足两张的类别，只从剩余训练样本划入，仍为每类保留至少两张训练图片。
准备步骤保留完整明细；已存在且哈希一致时复用，发生冲突时拒绝覆盖。

在当前仓库上审计的结果：训练 1606，已知验证 223；已知测试 451，near 测试 204，far 测试 272。
19 条训练/验证重复记录被排除，4 张训练图片划入验证；测试内部的一条重复记录只报告，保持不动。
这是完全重复审计，不是近重复/pHash 去重或采集批次泄漏审计。
验证集固定按类别分为 checkpoint selection 与 calibration 两部分，分区种子固定为 1729，
各训练 seed 共用相同分区以便比较。严禁将它们重新合并调参。

## 训练

```bash
export CUDA_VISIBLE_DEVICES=0
bash tools/run_boundaryshell_v3.sh train 1
```

训练默认顺序执行：

1. 分类阶段：最多 100 epochs，patience=20；只训练 MaPLe prompt/VPT。
   leaf CE + 0.2 parent CE + 0.05 morphology contrastive loss。
   保留原 hierarchical_episode sampler 与 morphology_pool。以 known selection leaf accuracy 选 best。
   学习率调度器每 epoch 更新一次。分类 logits 保持全局 leaf 预测，不强制经过 parent gate。
2. 开集阶段：冻结分类模型，提取一次 train_reference 特征；仅训练独立残差 projection 30 epochs。
   全类支持域与收缩半径只从训练特征估计。边界壳层在冻结分类空间生成，必须在所有已知支持域外；
   taxonomy 模式还要求仍在父级邻域内。再将这些样本送入可训练的 open 空间，学习 inside、
   separation、shell rejection 和 parent preservation。支持域每 epoch 刷新，最终再用全训练集重建。
   使用类别平衡的特征采样；不使用真实 unknown development 数据或已知 calibration 部分挑选 head。

输出目录：`runs/boundaryshell_v3/seed_1/`。
FP32 是为了避免半精度 prompt 梯度不稳定，v3 单独将 MaPLe 中强制 half 的浅层投影转换回 FP32。
不改动旧模型文件。没有 LoRA、全 backbone 重训、第三阶段联合微调；这些属于上一方案的可选项。
不在主模型中叠加旧 unknown text prompt loss、open_treecut pseudo-holdout、witness、arbitration。
Stage 1 显式优先保证 closed accuracy，旧模块仍留在仓库供基线实验使用。

若要分别执行两阶段：

```bash
bash tools/run_boundaryshell_v3.sh classifier 1
bash tools/run_boundaryshell_v3.sh open 1
```

`train` 与上述分阶段命令二选一，不要重复跑已经完成的阶段。
程序拒绝覆盖已有 checkpoint。没有 optimizer 中断续训；分类阶段中断后需要新 run 重新训练，
或明确使用已经保存的 best 进入 open 阶段。两种启动方式的 open 阶段随机种子保持一致。

## 校准

```bash
bash tools/run_boundaryshell_v3.sh calibrate 1
```

仅使用固定 known calibration 部分。保守 root gate 是 parent semantic score 与 parent manifold
score 都不足才拒绝；local gate 使用 global predicted leaf 的 open-space 距离/半径比。
校准联合验证整个级联，要求经验 known E2E **严格 >0.90** 且相对同一模型 closed accuracy
下降不超过 0.02。只满足 coverage 不算通过。保留阈值边界的相等分数，避免 ties 降低准确率。
不使用真实 unknown 调阈值。

成功输出 `calibration/thresholds.json` 与 `calibration/validation_scores.jsonl`。
若 closed accuracy 本身不足，输出 `calibration/gate_failure.json`，返回非零退出码；不产生
可测试 router，也不会自动降低 90% 要求。这时打包已有结果回传即可，不要强行继续 test。
约百张的 calibration 样本下，该约束仅为经验约束，不是泛化保证或 conformal 保证。

## 测试

```bash
bash tools/run_boundaryshell_v3.sh test 1
```

只在这一步提取测试特征和预测。校准文件绑定 classifier、open head、训练/验证清单和 hierarchy 哈希。
文件不匹配会拒绝测试。测试输出不允许覆盖。
输出含 known/intra/far、near AUROC/FPR95/OSCR、每已知类别、每未知物种、父级回退、Wilson 区间和
最终测试集 research gate。所有率均为 [0,1]，0.92 表示 92%。
测试 >90% 是报告项：若不达标仍完整保留结果，不得拿测试结果重新挑阈值并声称为盲测。

## 三个随机种子

如果 seed 1 尚未执行，可以一次运行三组：

```bash
export CUDA_VISIBLE_DEVICES=0
for seed in 1 2 3; do
  bash tools/run_boundaryshell_v3.sh all "$seed" || break
done
```

如果 seed 1 已经完成，只运行 seed 2、3：

```bash
for seed in 2 3; do
  bash tools/run_boundaryshell_v3.sh all "$seed" || break
done
```

每个 seed 单 GPU；有两张卡时可分别用 CUDA_VISIBLE_DEVICES=0 和 1 启动不同 seed。
不要对同一 seed 并发运行。单次耗时取决于服务器，尚未实测半天/一天预算。

## 需要回传的文件

无论成功或在哪一步失败，都可执行：

```bash
bash tools/run_boundaryshell_v3.sh pack 1
```

把 `runs/boundaryshell_v3/seed_1_review.tar.gz` 发给我。
其他 seed 同理；可统一再打一个包：

```bash
for seed in 1 2 3; do
  bash tools/run_boundaryshell_v3.sh pack "$seed"
done
tar -czf boundaryshell_v3_all_seeds_review.tar.gz -C runs/boundaryshell_v3 \
  seed_1_review.tar.gz seed_2_review.tar.gz seed_3_review.tar.gz
```

回传包自动包含存在的以下文件，不需要上传图片、CLIP 权重或大型 .pth：

- config.json、training_inputs.json、prepare_report.json、preflight.json、validation_partition.json。
- classifier/history.jsonl、classifier/selection.json。
- open/history.jsonl、open/summary.json 或 open/generator_failure.json。
- calibration/thresholds.json、calibration/validation_scores.jsonl 或 calibration/gate_failure.json。
- test/metrics.json、test/predictions.jsonl。
- logs/*.log。

## 调参和泛化实验边界

`boundaryshell.neighborhood` 支持 taxonomy、knn、random 三种邻域；knn/random 仅在确实同父级时
施加 parent-preserving loss，不能把跨父级混合样本冒称有可靠父标签。
单子类父级没有合法 sibling，不生成 taxonomy shell；过少有效壳层样本会报错并保留诊断。

核心几何模块不限定父类别名称；现有数据/模型入口沿用浮游动物的七父级约定。
iNaturalist/CIFAR 的数据适配和独立 benchmark 实验尚未实现，不能声称已验证跨数据集泛化。
本次不承诺论文新颖性或目标指标已达到；需要真实多 seed 结果和相同数据协议下的基线比较。
