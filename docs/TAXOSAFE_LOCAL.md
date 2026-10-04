# reference 局部拒识与父类回退校准（2026-10-03）

入口复用原 reference 的 MaPLe、排序头、成员性头和支持库，以及已知 TRAIN 的层级相对 Mahalanobis 统计。没有新的训练损失或优化器，`fit` 为统计拟合。原 reference、重构精修、全局几何融合的配置和入口仍可使用；本次方法必须显式选用 `TaxoSafe_reference_local.yml`。

实测限制必须保留：在已上传几何分数上，先固定 DEV 规则，再回放 TEST，known 从 94.00% 降到 92.67%，near 从 60.78% 升到 64.71%，extra 从 80.88% 升到 83.09%，叶级精确率从 83.10% 升到 86.16%。仍然只有 known 通过验收。Oithona plumifera 的正确父类回退从 4/32 降到 3/32。这是拒识与已知准确率的取舍，不是全面超过 reference。完整逐物种结果见 [实验回放报告](REFERENCE_LOCAL_TRIAL1_REVIEW_20261003.md)。

## 为什么修改

上次全局融合在 8 组权重、8,712 个阈值组合中没有保护条件可行点。进一步枚举相同权重的完整分数决策边界，也无法同时保留全部 DEV 已知正确图片与原叶级精确率。因此保留原始决策，以局部规则补充证据；分数不再在所有图片上全局混合。

每条规则由两项证据的交集触发，先校准共享规则，再在 DEV 已知候选图片不少于 10 张的父类内校准局部规则。少样本父类只使用共享规则。这一数量门槛属于正则化约束，不代表统计置信保证。阈值取该规则适用的拟合图片的实际分数；每轴最多 257 个值，超过时确定性抽样，不能声称任意数据规模都穷尽全部联合解。

1. `leaf_reject`：reference 输出已知物种，但原叶级成员性和几何叶级证据同时偏低时，回退父类。先执行这一阶段，后续根拒识必须保留已恢复的正确父类回退数。
2. `root_reject`：reference 已通过父门，但原父级成员性与几何父级证据同时偏低时，拒绝到根。
3. `root_rescue`：reference 拒绝到根时，尝试恢复为父类；不会恢复到已知物种。
4. `parent_repair`：仅对 reference 未输出物种的图片，独立细粒度排名提出父类候选；它必须同时为父类排名第二、且原始父成员性通过基线父阈值，才允许校准有界的父类改选。物种排序不修改。

每步保护所有原先正确的 DEV 已知图片、逐物种已知正确数、各未知来源正确数和叶级精确率；规则不允许把基线拒绝的图片提升为已知物种，也不替换已知物种 ID。候选字段 `candidate_parent` / `candidate_leaf` 保留原排名，实际父类回退以 `parent` 为准，`route_parent` 和 `alternative_parent` 单独记录。

## 来源留出与本次启用结果

四种动作分别进行未知物种来源留出，每折重新拟合原 baseline 阈值与所有局部规则，阈值网格和父类可用性只由该折拟合数据决定。某动作必须在所有留出来源上不减少正确未知路由数、不增加错误物种接受数，并在至少两个不同来源上改善其中一项。这里将减少未知图片的错误物种接受计为精确率相关收益，区别于上一版只统计正确回退数。

通过的动作重新组合、重新拟合，再做一次组合留出检验。不能用组合总收益掩盖其中一个来源的退化。没有通过的规则时按 reference 原决策精确回退。

本次 `leaf_reject` 和 `root_reject` 通过，组合在 8 个留出来源上无上述两类退化，在 6 个来源上有收益。`root_rescue` 在留出的 Brachyura zoea larva 上退化，被禁用；`parent_repair` 没有得到足够正收益，也被禁用。没有因为 Oithona 的 TEST 表现而写入物种专属规则或强制启用父类修复。

留出检验参与模型选择，因此是 DEV 稳定性检查，不是独立泛化估计。已知样本保护是在拟合 DEV 上的约束；TEST 上实际少正确识别 6 张，说明不能把该约束当成对未见图片的保证。该 TEST 已多次用于项目研发，论文结论还需要未参与研发决策的数据或外层验证。

## 覆盖安装与运行

压缩包顶层为 `Openset/`，包含完整分支源码、配置、测试、文档及 CLIP 分词资源；不包含 `runs/`、`prepro/raw/`、`prepro/data/`、模型权重或 `.git/`。先停止使用本项目源码的运行进程，再覆盖：

```bash
cd /home/ubuntu/hdd/data/qz
tar -xzf Openset_local_full_20261003.tar.gz
conda activate ProTeCt
cd /home/ubuntu/hdd/data/qz/Openset
sha256sum -c SUPPORT_LOCAL_FILES.sha256
python tools/run_unit_tests.py --pattern 'test_taxosafe_geometry*.py' --quiet
bash tools/run_taxosafe_local.sh
```

该脚本依次 preflight、fit、calibrate、test，并自动打包新实验的文本诊断。默认来源是 `runs/taxosafe_new/reference/trial_1`，新目录是 `runs/taxosafe_new/reference_local/trial_1`。不需要重训 reference；需要保留它的完整权重、支持库、DEV 分数、配置和完成记录。不能以旧文本 review 包代替服务器上的完整源产物。

若新目录已经运行过，显式换一个未使用的目录，避免覆盖中途产物：

```bash
bash tools/run_taxosafe_local.sh \
  runs/taxosafe_new/reference/trial_1 \
  runs/taxosafe_new/reference_local/trial_1_retry
```

如需手动控制阶段，仍使用 `refine_taxosafe_geometry.py fit|calibrate|test`，指定新配置即可。三个阶段必须使用同一配置、源码和 reference 来源。此次 geometry 代码签名已变化，旧 geometry 产物不要直接拿来作为新生产运行的 fit 产物；旧分数可以用下面的独立审计工具复核。

## 不用 GPU 的旧分数审计

```bash
python tools/replay_taxosafe_local.py \
  --geometry-run-dir runs/taxosafe_new/reference_geometry/trial_1 \
  --reference-config runs/taxosafe_new/reference/trial_1/training/config.json \
  --output-dir runs/taxosafe_new/reference_local_score_audit/trial_1 \
  --evaluate-test
```

默认不加 `--evaluate-test` 只读 DEV。加该参数后，工具先保存 DEV 规则和 `selection_frozen.json`，再打开 TEST；检查 DEV/TEST 图片不重叠，并核对存档 reference TEST 决策。输出是分数诊断，不是可部署的模型/阶段完成记录。服务器从图片重新推理的结果仍需复核，数值边界附近可能受设备精度影响。

## 读取结果

- `calibration/router.json`：实际启用的动作、规则、逐动作留出结果、组合结果、DEV 保护审计。
- `test/summary.json`：四项指标与逐已知物种、近域未知物种、远域未知来源表现。
- `test/preservation.json`：候选保留数、根/叶门改变数、实际父类改选数。
- `test/component_scores.json`：连续成员性/几何分数的 AUROC。
- `test/predictions.jsonl`：每张图片的原始证据、最终结果和触发的规则编号。

局部路由没有一个统一连续阈值分数，公共 root/local knownness 明确记作 ±1 决策指示量，类型为 `local_route_indicator_not_probability`。`metrics.json` 中这些指示量的 AUROC/OSCR 仅描述离散工作点；比较连续分数的排序能力应使用 `component_scores.json`，不能混为一谈。

新版本校验使用 `SUPPORT_LOCAL_FILES.sha256`。包内旧校验清单保留作历史记录，不用于验证本次更新后的文件。
