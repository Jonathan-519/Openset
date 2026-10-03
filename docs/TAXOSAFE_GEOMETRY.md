# 冻结 reference 的层级相对距离验证

本入口使用已经完成的 `runs/taxosafe_new/reference/trial_1`，不依赖重构精修目录，也不重训 MaPLe。原 reference 与 reconstruction 入口、配置、模型实现保持不变。

本次上报实验中，重构精修令 TEST known 从 423/450 降为 419/450，near 从 124/204 升为 128/204，extra 仍为 220/272。新方案的真实识别指标尚未测得，软件测试通过不等于四项研究验收通过。

研究依据、相对距离公式及边界见 [文献与方法说明](TAXOSAFE_GEOMETRY_RESEARCH.md)。方法冻结原类别排序，在 parent/fine 两个特征空间分别拟合已知 TRAIN 的均值和收缩协方差，以相对 Mahalanobis 分数补充原成员性分数。没有新的梯度损失、优化器或训练轮数；`fit` 表示统计拟合。特征和原分数的标准化参数也只来自 TRAIN。

## 直接覆盖服务器源码

压缩包顶层目录为 `Openset/`。在 `/home/ubuntu/hdd/data/qz/` 解压，会合并到现有项目目录。包内含本分支完整源码、配置、测试、文档和 CLIP 分词资源；不含 `runs/`、`prepro/raw/`、`prepro/data/`、模型权重、缓存或 `.git/`，因此不会用打包时的旧清单覆盖服务器数据划分或覆盖运行产物。

先停止正在使用本项目源码的进程，再执行：

```bash
cd /home/ubuntu/hdd/data/qz
tar -xzf Openset_geometry_full_20261003.tar.gz
conda activate ProTeCt
cd /home/ubuntu/hdd/data/qz/Openset
sha256sum -c SUPPORT_GEOMETRY_FILES.sha256
python tools/run_unit_tests.py --pattern 'test_taxosafe_geometry*.py' --quiet
```

若提示 `FAILED`，不要训练，先检查压缩包是否完整解压或源码是否被另外修改。清单描述本次完整源码快照；旧版本的 SHA 清单继续保留作历史记录，不用于校验新版本工具文件。

## 复用现有 reference 产物

请保留整个 `runs/taxosafe_new/reference/trial_1`。必须有训练检查点、TRAIN 支持库、配置和审计记录，以及 `calibration/router.json`、`completed.json` 与 `development_scores.jsonl`。文本 review 压缩包不能代替这些二进制产物。新入口会验证来源文件哈希和旧 DEV 推理是否重现。

你刚完成的 reference 实验可以直接使用，不需要删除任何旧运行目录。以下新目录若已经使用过，请换一个未使用的目录。

```bash
conda activate ProTeCt
cd /home/ubuntu/hdd/data/qz/Openset

cfg=configs/Zooplankton_Taxonomic_Tree/TaxoSafe_reference_geometry.yml
base=runs/taxosafe_new/reference/trial_1
run_dir=runs/taxosafe_new/reference_geometry/trial_1

python refine_taxosafe_geometry.py fit --config "$cfg" --reference-run-dir "$base" --run-dir "$run_dir" --preflight &&
python -u refine_taxosafe_geometry.py fit --config "$cfg" --reference-run-dir "$base" --run-dir "$run_dir" &&
python -u refine_taxosafe_geometry.py calibrate --config "$cfg" --reference-run-dir "$base" --run-dir "$run_dir" &&
python -u refine_taxosafe_geometry.py test --config "$cfg" --reference-run-dir "$base" --run-dir "$run_dir"
```

三个阶段必须保持配置、源码和源 reference 目录一致。若修改配置或更换源模型，请使用新的输出目录。默认使用 CUDA；CPU 仅用于小规模测试或显式 `--device cpu`。

## 如何理解校准与测试

- 原分类候选固定，父级接受门可以调整，从而解除旧方案固定父阈值造成的上限；但候选父类本身错误的问题仍存在，会单独报告。
- 新方案需要保留 DEV 中每一个基线正确的已知样本，保护未知来源的正确数和开放世界叶级精确率。它不能保证 TEST 上逐样本不退化。
- 未知来源留出检验会在每折重新拟合基线阈值、新融合权重和新阈值，留出来源不参加该折选择。除了逐来源不退化，还要求至少两个留出来源的正确数严格增加；所有折都回退基线，或收益仍只集中于一个来源，不能作为启用新方案的依据。它是来源泛化诊断，使用过的整体 DEV 仍属于开发集，不是新的独立测试集。
- 没有合格改善或来源稳定性检查不通过时，自动使用原 reference 路由；报告明确记录回退原因。
- 原四项验收标准保持 known >90%、near ≥85%、extra >90%、开放世界叶精确率 >90%。程序能够完成 TEST，不代表验收达标。
- 保存 baseline 与新方案的成对指标、逐图片路由、逐物种结果和门变化记录；不能只报告变化后最好的单项。

不要根据 TEST 选择融合权重或修改某个物种的规则。该 TEST 已用于多轮研发比较，论文中的最终泛化结论需要独立的未参与决策数据或严格的外层验证。

## 打包实验结果

```bash
python tools/pack_taxosafe_new_review.py --run-dir runs/taxosafe_new/reference_geometry/trial_1
```

把生成的 `.tar.gz` 发回即可。打包只包含新目录的文本诊断；不会跟随源目录引用，也不会上传训练图片、特征缓存或权重。
