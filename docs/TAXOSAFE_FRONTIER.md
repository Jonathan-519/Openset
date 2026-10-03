# Frontier 安装与执行说明

本版本新增精确保护边界搜索和逐阶段拒绝原因报告。它是探索性校准改进，
不是已经达到四项研究目标的模型。是否启用规则，以新运行的
`calibration/router.json` 中 `selection_status`、`baseline_fallback` 和
`reject_rules` 为准；`workflow_completed=true` 仅表示流程完成。

## 安装到现有服务器

将 `Openset_frontier_full_20261003.tar.gz` 放到 `/home/ubuntu/hdd/data/qz/`：

```bash
cd /home/ubuntu/hdd/data/qz
tar -xzf Openset_frontier_full_20261003.tar.gz -C /home/ubuntu/hdd/data/qz
cd /home/ubuntu/hdd/data/qz/Openset
sha256sum -c SUPPORT_FRONTIER_FILES.sha256
```

压缩包顶层为 `Openset/`，包含完整源码、配置、测试、文档和 BPE tokenizer 资源。
包内没有 `prepro/raw/`、`prepro/data/`、`runs/`、模型权重、缓存或 `.git/`；
解压不会删除这些已有目录。无需 git clone、git pull 或升级现有环境。
使用原来的 `ProTeCt` 环境。

## reference 不需要重训

本次仍使用已经恢复的完整 reference：

```text
runs/taxosafe_new/reference/trial_1_retrain_20261003_185220
```

该目录中的 `best.pth`、`support.pth`、router 和完成收据必须完整且同源。
不要删除它，也不要改写或随后补跑其 TEST；这些操作会改变绑定。
如果该目录丢失，先恢复完整备份。本代码包及 review 包都不能恢复模型权重。

## 正式服务器复核（仅在需要验证图像执行链时运行）

```bash
cd /home/ubuntu/hdd/data/qz/Openset
conda activate ProTeCt
FRONTIER_RUN="runs/taxosafe_new/reference_frontier/trial_1_$(date +%Y%m%d_%H%M%S)"
bash tools/run_taxosafe_frontier.sh \
  --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_reference_frontier.yml \
  --reference-run-dir runs/taxosafe_new/reference/trial_1_retrain_20261003_185220 \
  --run-dir "$FRONTIER_RUN"
```

流程：只读预检 → 已知 TRAIN 的冻结特征／统计拟合 → DEV 嵌套校准 → 打包结果。
这是 `optimizer_steps=0` 的统计拟合，没有重新训练 reference 网络。
已有或失败 run 不原地重跑；另建时间戳目录，不删除收据或锁文件。

终端会分别输出流程完成、规则选择、是否回退、研究门禁状态。若仍然
`baseline_fallback=true`，不必为寻找不同结果重复训练或重复 TEST。
结果包位于运行目录的同级，名称含 `_review_`，不含权重，不是模型备份。

## TEST 仅在规则冻结后显式执行

保留上一步终端变量，或将 `FRONTIER_RUN` 设置为实际已完成的新目录：

```bash
python -u -m taxosafe_frontier test \
  --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_reference_frontier.yml \
  --reference-run-dir runs/taxosafe_new/reference/trial_1_retrain_20261003_185220 \
  --run-dir "$FRONTIER_RUN" --evaluate-test
python tools/pack_taxosafe_frontier_review.py --run-dir "$FRONTIER_RUN"
```

TEST 只评价已经选定的规则，不能反向挑选阈值、来源、动作或 seed。
历史 TEST 已被查看，本次结果不称为新的独立最终检验。

## 文本回放与测试

若只想复现本次 DEV 搜索，无需 GPU、原图或权重。先将 ParentRisk review 包
安全解压到项目外的单独目录，把其 `run/` 路径传给：

```bash
python tools/replay_taxosafe_frontier.py \
  --review-run-dir /absolute/path/to/extracted/run \
  --output /absolute/path/to/new_frontier_replay
```

回放验证原 TRAIN/DEV 文本收据之间的绑定及 DEV 文件哈希，不读取 TEST。
它没有完整模型二进制，也不执行图像推理，不能冒充正式阶段完成收据。

正式软件回归入口：

```bash
python tools/run_unit_tests.py --pattern 'test_taxosafe_frontier*.py'
```

## 实现边界

- 在全 taxonomy 共享最多一个叶拒识矩形、一个根拒识矩形；不按未知物种写规则。
- 保留原父／叶候选，不把原拒识样本提升为已知叶，不增加父恢复／修复动作。
- 只在折内观测分数上生成保护前沿；标签仅用于拟合与审核，推理不读取真值。
- 每动作分别检查双来源收益、逐图 known/near 保护、near 父路径与来源级保护。
- 部署候选在内层完成选择；外层仅报告。保留失败候选及无合格候选的回退。
- 所有旧代码及其签名保持，新方法使用独立模块、配置、阶段目录和签名。

论文与失败原因见 `docs/TAXOSAFE_FRONTIER_RESEARCH.md`。
