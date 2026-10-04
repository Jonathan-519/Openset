# ParentRisk：冻结 reference 的父证据与校准审计

## 版本与实施范围

本版以 GitHub `new` 分支 `20b927e10076d6e127ff597f586c6681c1bf47a9` 为基线，落实 2026-10-03 HANDOFF 与后续设计审查的第一批开发。旧 reference、refine、geometry、local、训练入口、导入白名单及原配置不修改。新增模块为 `taxosafe_parentrisk`，独立入口为 `refine_taxosafe_parentrisk.py`。

本版实现的是冻结模型证据采集、TRAIN 统计拟合及 DEV 校准；没有重新训练 backbone、prompts 或参考支持库。CPU 软件测试不能证明服务器 CUDA 全图像流程已经完成，也不能证明四项研究门禁已经达标。实际效果以新目录产生的服务器结果为准。

### 已落地的改动

1. 新采集器所有 split 均显式传递 `query_hashes`，在支持匹配排序/聚合之前排除相同内容。导出每图自匹配排除量及各叶/父有效参考数；完整 taxonomy 没有有效支持时明确报错，不填造分数。
2. 将 `encoder_evidence.parent_text_logits` 与原 `support_evidence.parent_logits` 分开保存。前者是学习过 MaPLe prompts 的图文父分数，后者是支持库父排名，两者不能混称。
3. `ParentEvidenceGeometry.score_parents()` 为每个父分别计算 RMD，导出完整 `[P]` 向量。第二候选父不会复用第一父的距离。
4. 父文本、父 membership、父 RMD 的尺度仅由已知 TRAIN 拟合，每个分数族共享一个 median/MAD 标准化器；同时保留原四个候选几何/成员性通道。TRAIN 自匹配修正可能改变支持分数和尺度，不能据此宣称直接改善表示。
5. 同时留出 known 与未知来源的固定嵌套校准审计；按内容哈希去重，外层每张图恰好一次输出，外层评价不参与选择。空类和稀有类保留并报告证据不足。
6. 父模块先在支持排名 top-2 内选择一次父，再检查绝对兼容性及间隔；不逐个尝试直到某个父过门，也不要求新父先超过旧 membership 阈值。首次父模块只处理原 reference 的非叶输出。
7. 拒识规则限制为共享叶拒识和共享根拒识，避免首轮继续扩大父类局部规则自由度。拟合阶段逐图保护原正确 known、已经正确回退的 near，并保护原父候选正确、原父门已通过的 near 不被转移到根；组合也重新审计。
8. 独立产物协议绑定代码、配置、源 reference、TRAIN 统计、校准 router 与折计划；默认脚本只做 DEV，TEST 必须显式开启。

### 三个固定消融配置

| 配置 | 用途 |
|---|---|
| `TaxoSafe_reference_parentrisk_audit.yml` | 新校准/风险审计与共享拒识，不启用新父模块 |
| `TaxoSafe_reference_parentrisk_parent_only.yml` | 单独测试父兼容性；保留所有原叶输出，包括未知误接受的叶 |
| `TaxoSafe_reference_parentrisk.yml` | 父模块与共享拒识的组合实验 |

同一 reference 可以用于三个独立目录，但不能把一个配置的 fit 产物拿到另一个配置中校准。parent-only 保持所有原叶输出，因此其正确 known 叶数和开放叶精确率不变；它不能清除未知叶误接受，不能独自完成全部目标。

## 服务器安装

压缩包顶层为 `Openset/`。先确认没有训练、校准或测试进程使用此目录，再将压缩包放入 `/home/ubuntu/hdd/data/qz/` 并解压：

```bash
cd /home/ubuntu/hdd/data/qz
tar -xzf Openset_parentrisk_full_20261003.tar.gz
cd Openset
sha256sum -c SUPPORT_PARENTRISK_FILES.sha256
```

不要删除 `Openset`，不要在其内部解压带同名顶层的包，不使用 `rsync --delete`。包不包含也不覆盖 `prepro/raw/`、`prepro/data/`、`runs/`、模型权重和 `.git/`；保留服务器原有数据、清单及运行目录。BPE tokenizer 资源随源码提供。

沿用服务器已成功运行的 `ProTeCt` 环境，不升级 torch/torchvision/CUDA。激活环境后执行软件回归及已有 CUDA 历史故障回归：

```bash
conda activate ProTeCt
python tools/run_unit_tests.py --pattern 'test_taxosafe_parentrisk*.py' --quiet
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"' &&
python -m unittest tests.test_taxosafe_support_reference.ReferenceEvidenceTest.test_cuda_reference_training_loss_and_backward -v
```

完整旧流程回归可运行 `python tools/run_unit_tests.py --quiet`，不要搜集根目录 `test*.py`。

## 运行协议

源目录必须是一套完整且冻结的 reference。至少需保留原 `training/{config.json,inputs.json,completed.json,best.pth,support.pth}` 与 `calibration/{router.json,completed.json,development_scores.jsonl}`。review 包不能恢复权重。若原 `runs/taxosafe_new/reference/trial_1` 仍完整，可直接复用；用户于 2026-10-03 后续确认已删除训练产物，因此应先按下一节重新训练。

新实验目录必须未使用；成功或失败的目录均保留，另取名字重跑。不要删除 `completed.json`、锁文件或修改签名强行续跑。三个阶段之间不得修改代码、配置或源 reference（包括给源补跑 TEST）。

三个配置的模型统计均只来自 known TRAIN；near/extra 只用于 DEV 校准。默认不读取 TEST 图像，也不扫描 raw 自动增加训练样本。源 TEST 收据若存在，只按原导入协议核验其绑定。

### reference 产物已删除：先重训再运行新方法

以下命令假设原始图像、锁定清单、taxonomy 与数据准备审计仍在。使用 stage-native `--preflight` 核验；不要调用旧 `prepare_taxosafe.py` 默认流程重建划分。全部命令在同一终端执行，任何失败都会停止后续步骤。

```bash
cd /home/ubuntu/hdd/data/qz/Openset
conda activate ProTeCt

stamp=$(date +%Y%m%d_%H%M%S)
base="runs/taxosafe_new/reference/trial_1_retrain_${stamp}"
run_dir="runs/taxosafe_new/reference_parentrisk/trial_1_${stamp}"
ref_cfg=configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_reference.yml
new_cfg=configs/Zooplankton_Taxonomic_Tree/TaxoSafe_reference_parentrisk.yml

python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"' &&
python -m unittest tests.test_taxosafe_support_reference.ReferenceEvidenceTest.test_cuda_reference_training_loss_and_backward -v &&
python train_taxosafe_new.py --config "$ref_cfg" --trial 1 --seed 1 --variant main --run-dir "$base" --preflight &&
python -u train_taxosafe_new.py --config "$ref_cfg" --trial 1 --seed 1 --variant main --run-dir "$base" &&
python -u calibrate_taxosafe_new.py --config "$ref_cfg" --trial 1 --seed 1 --variant main --run-dir "$base" &&
python -u test_taxosafe_new.py --config "$ref_cfg" --trial 1 --seed 1 --variant main --run-dir "$base" &&
bash tools/run_taxosafe_parentrisk.sh --config "$new_cfg" --reference-run-dir "$base" --run-dir "$run_dir"
```

顺序是 reference 训练 → reference 校准 → reference 基线 TEST → 新方法 TRAIN 证据拟合 → 新方法 DEV 校准与折外审计 → 文本打包。基线 TEST 只用于冻结后的对照，不用于后续权重或阈值选择。若要基线 TEST，务必放在新方法 fit 之前；完成新方法 fit 后再给源补跑 TEST 会改变源绑定，不能继续旧下游运行。

reference 保持原训练预算（最多 120 epoch、patience 25），实际早停由原代码决定。ParentRisk 的 fit 没有反向传播或优化器更新。重训后的参数与此前被删除的模型未必相同，不能保证复现原 94.00% 已知识别率。

上述脚本结束后，可在同一终端显式评估已经冻结的新方案并重新打包：

```bash
python -u refine_taxosafe_parentrisk.py test --config "$new_cfg" --reference-run-dir "$base" --run-dir "$run_dir" --evaluate-test &&
python tools/pack_taxosafe_parentrisk_review.py --run-dir "$run_dir"
```

若换了终端，先把 `base`、`run_dir`、`new_cfg` 设置成上一次真实使用的路径，不要重新生成时间戳后对不存在的模型执行 TEST。以下其他消融示例中的旧源路径也须替换为新 `base`。

### 首次运行：固定的三项 DEV 消融

下面命令在三个新目录中依次执行 preflight → fit → calibrate → review 打包。`stamp` 仅用于避免覆盖目录，不改变随机种子。任一步失败立即停止；保留失败目录及日志。

```bash
cd /home/ubuntu/hdd/data/qz/Openset
stamp=$(date +%Y%m%d_%H%M%S)
for mode in audit parent_only combined; do
  case "$mode" in
    audit) cfg=configs/Zooplankton_Taxonomic_Tree/TaxoSafe_reference_parentrisk_audit.yml ;;
    parent_only) cfg=configs/Zooplankton_Taxonomic_Tree/TaxoSafe_reference_parentrisk_parent_only.yml ;;
    combined) cfg=configs/Zooplankton_Taxonomic_Tree/TaxoSafe_reference_parentrisk.yml ;;
  esac
  bash tools/run_taxosafe_parentrisk.sh \
    --config "$cfg" \
    --reference-run-dir runs/taxosafe_new/reference/trial_1 \
    --run-dir "runs/taxosafe_new/parentrisk_${mode}/trial_1_${stamp}" || break
done
```

若只先运行父模块隔离实验：

```bash
bash tools/run_taxosafe_parentrisk.sh \
  --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_reference_parentrisk_parent_only.yml \
  --reference-run-dir runs/taxosafe_new/reference/trial_1 \
  --run-dir runs/taxosafe_new/parentrisk_parent_only/trial_1_first
```

不要把软件 `--preflight` 通过写成正式运行完成。完整图像流程默认 CUDA；CPU 合成集成验证不承诺完整 MaPLe 图像流程可以在 CPU 运行。

### 冻结方案后的 TEST 与打包

TEST 是显式阶段：先完成 DEV 检查并固定方案，然后将 `run_dir` 改成该配置已经完成校准的真实新目录。不得根据 TEST 改动方案，再在同一结果上宣称独立验证。

```bash
cfg=configs/Zooplankton_Taxonomic_Tree/TaxoSafe_reference_parentrisk_parent_only.yml
run_dir=runs/taxosafe_new/parentrisk_parent_only/trial_1_first
python -u refine_taxosafe_parentrisk.py test \
  --config "$cfg" --reference-run-dir runs/taxosafe_new/reference/trial_1 \
  --run-dir "$run_dir" --evaluate-test &&
python tools/pack_taxosafe_parentrisk_review.py --run-dir "$run_dir"
```

`--evaluate-test` 也可显式添加到首次 wrapper 命令，使其完成 DEV 后继续评估；默认关闭。已有 TEST 阶段不能重复覆盖。review 包包含文本诊断、折计划和折外逐图结果，不包含模型、图片或特征缓存。

### 重点查看的文件

| 文件 | 解释 |
|---|---|
| `parentrisk/fit_report.json`、`scales.json`、`train_scores.jsonl` | TRAIN-only 统计、自匹配排除与证据 |
| `calibration/baseline_reproduction.json` | 冻结 reference 的原 DEV 分数与决策复现 |
| `calibration/fold_plan.json` | 固定外层哈希与 held-known/未知来源分配 |
| `calibration/oof_predictions.jsonl`、`oof_baseline_predictions.jsonl` | 每张唯一图至多一次折外预测；不完整覆盖会明确标记 |
| `calibration/outer_audit.json` | 完整选择过程的外层评价、是否退化、覆盖是否完整 |
| `calibration/router.json` | 最终生产规则、内层选择和每条规则的支撑 |
| `calibration/full_fit_audit.json`、`validation_report.json` | 完整 DEV 配对风险与原四门禁，不等于折外成绩 |
| `test/summary.json`、`predictions.jsonl` | 显式 TEST 后的冻结方案结果，唯一内容计量 |
| `test/paired_risk_report.json` | TEST 的已知新增伤害、near 路径损失、逐类与来源 macro |

可另行执行只读错误诊断，默认只读 DEV，不创建正式阶段收据：

```bash
python tools/diagnose_taxosafe_failure_modes.py \
  --run-dir runs/taxosafe_new/parentrisk_parent_only/trial_1_first
```

## 评价与报告边界

保留四项原始验收条件：known 正确叶准确率 **>90%**，near 正确父回退 **≥85%**，extra 根拒识 **>90%**，所有叶输出中的正确 known 精确率 **>90%**。near 根输出、extra 父输出均计错。

新增已知伤害 `H_K` 的分子为原正确 known 叶被新方法破坏的图片数，分母为所有 known；near 父路径伤害 `H_N^P` 的分子为原父候选正确且已通过父门、后来输出根的 near，分母为所有 near。二者都报告分子/分母，并保留 source macro、逐叶和逐父覆盖。拟合时零伤害只是观测约束，不是泛化保证。

外层 known DEV 曾被冻结 reference 用于 checkpoint 选择，所以本版审计明确标记：

```text
validation_scope = postprocessor_conditional_on_frozen_reference
independent_model_level_validation = false
```

它不能被称为整个模型的独立未见验证。完整 DEV 生产 router 的报告与外层逐图结果分开保存。不能堆叠多次 held-known/来源组合输出虚增样本数，不能利用外层结果改选一条规则后仍声称同一外层是无偏验证。

全 DEV 四门禁未通过、内层拒绝新动作、折外退化、证据不足分别报告。软件测试通过、运行阶段完成、校准保护通过、研究达标是四种不同状态。

## 本版没有实施的后续研究

原始空间 patch/FRN 重构、新父分支梯度训练、E-HND/State-FGOD 终端损失、严格未见物种训练适配都不在首轮实现中。本版不声称完成这些研究。严格留类的 inactive/null 分数不能直接输入正式完整 taxonomy decoder；将来需要独立 active-mask/全局 ID 适配和折内统计，不能填 0 或大负数冒充。

本版也未引入父类局部阈值修正或部分共享，以先控制自由度。若三个消融在开发协议下没有稳定收益，保留负结果，再依据空间特征诊断选择一项后续表征/损失实验。不得针对已知 TEST 物种写规则、调阈值或选最佳 seed。

## 发布验证

完整正式单测 runner 共运行 **449** 项，**442 通过、7 项因 CUDA 不可用跳过**；新增 53 项中 52 通过、1 项跳过。222 个 Python 文件通过 Python 3.8 语法解析，shell 脚本语法及新 CLI 帮助检查通过。语法检查不代表已在 Python 3.8 或用户 CUDA 环境执行。

合成集成验证使用真实训练/导入与新阶段调用链，但以确定性小模型和图像替代真实 CLIP/浮游动物数据。它覆盖源权重及支持库不变、TEST 不拟合、产物修改阻断、重复图计量和 parent-only 原叶保持。未在本次助手环境进行真实服务器图像实验；具体记录见 `SUPPORT_PARENTRISK_VERIFICATION.json`。
