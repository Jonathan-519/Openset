# TaxoSafe reference：Linux 运行说明

本轮新增真实参考成员性方案。原训练／校准／测试入口保留，主方案通过 `configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_reference.yml` 显式启用，输出到独立的 `runs/taxosafe_new/reference/trial_1/`。方法和一手论文依据见 [研究说明](TAXOSAFE_REFERENCE_RESEARCH.md)。

新方案需要从原始 CLIP 初始化重新训练。不要续训或覆盖旧 `decoupled/trial_1`，不要把旧 checkpoint、支持库或 router 移入新目录。旧实验跨阶段加载需要与其签名一致的源码，例如上一版 decoupled 产物使用 `0273dbf` 对应代码；不能删掉哈希检查来强行兼容。

## 1. 本轮改变什么

| 部分 | 实际实现 |
| --- | --- |
| 编码与类别排名 | 保留一次共享 MaPLe/CLIP 视觉前向，以及已有父／叶表示和类别排名路径 |
| 成员性 | 查询与支持库内真实 TRAIN 图片成对匹配，使用全局余弦与双向局部覆盖特征 |
| 参考聚合 | 每叶取最高两个有效参考关系 logit 平均，不足两个时取实际有效数量；父层先在各子叶内聚合，再对活跃子叶等权平均 |
| 排除规则 | 同内容查询、干预移除的物种／父类在选择和聚合之前排除 |
| 成对监督 | 叶层区分同类、兄弟物种和跨父类；父层使用同父跨物种正对及跨父负对，单叶父类的同物种回退单独计数 |
| 主解码 | 对齐候选父／叶成员性的两阈值接受规则；旧联合解码另作对照 |
| 校准 | 仅既定 Dev 数据上的分位数阈值网格，采用 known-first 选择策略 |

参考缓存来自 TRAIN；查询图像损失保留到可训练提示及适配模块的梯度。新方法不使用真实 near/OOD 更新网络，不使用 TEST 专属物种阈值。当前没有完成新方法的真实 GPU 训练，因此不能承诺性能改善或与 v11 同速。

新 reference 关系头替换 decoupled 的两个原型成员性头，未在其上再叠加一套未使用网络；默认头参数量与 decoupled 相同。每个 batch 的成对图可在不同支持干预间复用。增加的是参考关系计算与监督，不能由参数量相同推断实际速度相同。

新配置的关键字段为：

```yaml
support:
  decoupled: true
  membership: reference
  reference_topk: 2
  max_per_leaf: 8
  loss:
    pair_parent: 0.25
    pair_leaf: 0.25
calibration:
  decoder: membership
  policy: known_first
  threshold_grid: quantile
  membership_grid_points: 49
  source_loo: true
```

上面是关键字段摘录，不能单独替代完整配置。训练保留最多120 epoch、warmup 5 epoch、开放任务渐增5 epoch和patience 25；直接参考关系监督在warmup后随开放任务权重渐增，不需要额外查询视觉前向。

## 2. 更新服务器代码与检查

服务器目录没有 `.git` 时仍可按原方式运行，无需 `git checkout`。从 [new 分支完整源码 ZIP](https://github.com/Jonathan-519/Openset/archive/refs/heads/new.zip) 下载代码，在项目外解压后合并代码；保护原有 `prepro/raw/`、`prepro/data/`、`runs/`、CLIP 缓存和模型权重，不删除项目再整体替换，也不使用删除目标端文件的同步选项。更新代码前结束使用这些文件的训练／校准进程，保留旧源码备份用于历史产物读取。

如将 ZIP 上传为 `/home/ubuntu/hdd/data/qz/Openset-new.zip`，可用以下标准库脚本合并。它不会删除目标目录中的文件，跳过原图、清单、运行结果和权重；有变化的已有文件先备份到项目旁的时间戳目录。ZIP 文件名不同则先修改 `archive_path`。

```bash
conda activate ProTeCt
python - <<'PY'
from datetime import datetime
from pathlib import Path, PurePosixPath
import os
import shutil
import stat
import tempfile
import zipfile

archive_path = Path('/home/ubuntu/hdd/data/qz/Openset-new.zip')
project = Path('/home/ubuntu/hdd/data/qz/Openset').resolve()
backup = project.parent / ('Openset_code_backup_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
protected = (('prepro', 'raw'), ('prepro', 'data'), ('runs',))
weight_suffixes = {'.pth', '.pt', '.ckpt', '.safetensors', '.bin'}
with zipfile.ZipFile(archive_path) as archive:
    members = [item for item in archive.infolist() if not item.is_dir()]
    roots = {PurePosixPath(item.filename).parts[0] for item in members}
    if len(roots) != 1:
        raise ValueError('Expected one GitHub archive root')
    payloads = []
    for item in members:
        source = PurePosixPath(item.filename)
        if source.is_absolute() or '..' in source.parts or stat.S_ISLNK(item.external_attr >> 16):
            raise ValueError('Unexpected archive member: ' + item.filename)
        parts = source.parts[1:]
        if not parts or any(tuple(parts[:len(prefix)]) == prefix for prefix in protected):
            continue
        relative = Path(*parts)
        if relative.suffix.lower() in weight_suffixes:
            continue
        destination = project / relative
        if any(path.is_symlink() for path in (destination, *destination.parents) if path != project):
            raise ValueError('Refusing to replace a symlink: ' + str(destination))
        mode = stat.S_IMODE(destination.stat().st_mode) if destination.is_file() else ((item.external_attr >> 16) & 0o777 or 0o644)
        payloads.append((relative, archive.read(item), mode))
    expected = Path('configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_reference.yml')
    if expected not in {relative for relative, _, _ in payloads}:
        raise ValueError('This ZIP does not contain the reference configuration')
    backup.mkdir(parents=True, exist_ok=False)
    changed = 0
    for relative, payload, mode in payloads:
        destination = project / relative
        if destination.is_file() and destination.read_bytes() == payload:
            continue
        if destination.exists():
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, saved)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        temporary_path.chmod(mode)
        os.replace(temporary_path, destination)
        changed += 1
print('Updated files:', changed)
print('Changed existing files backed up to:', backup)
PY
```

```bash
conda activate ProTeCt
cd /home/ubuntu/hdd/data/qz/Openset
sha256sum -c SUPPORT_REFERENCE_FILES.sha256
python tools/run_unit_tests.py --quiet
```

单元测试结束的 `OK` 表示软件检查通过；测试中故意构造的失败 gate 不表示测试失败。真实实验的 `targets_passed` 需要另行查看。CUDA、PyTorch 和 CLIP 预训练权重沿用原服务器环境。已有干净 known 清单时不重新生成；仅在清单缺失时使用原 `prepro/build_taxosafe_v11_known_splits.py`，不要更改既有 Test 划分。

本轮验证记录见仓库根目录 `SUPPORT_REFERENCE_VERIFICATION.json`：全仓 **260 项 CPU 测试通过**；针对 `0273dbf` 原实现的 v1/v2 初始化、前向、损失、梯度与有效计数进行逐位一致性检查；旧 v2 Dev 两阈值诊断另经独立复算核对。验证不包含新 reference 模型的真实图片 CUDA 训练、正式性能或速度实验，也没有把缺少原图时的预检当成通过。

## 3. 一次主实验：完整命令

每个阶段都保留同一 `--config`、`--seed`、`--variant` 和 `--run-dir`。旧默认配置不变，遗漏新 `--config` 会运行旧默认方案。

### 3.1 训练预检及训练

```bash
python train_taxosafe_new.py --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_reference.yml --trial 1 --seed 1 --variant main --run-dir runs/taxosafe_new/reference/trial_1 --preflight
python -u train_taxosafe_new.py --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_reference.yml --trial 1 --seed 1 --variant main --run-dir runs/taxosafe_new/reference/trial_1
```

训练成功退出并生成 `training/completed.json` 后进入校准。预检不加载 CLIP，不证明正式训练或下游哈希加载必然成功。`--debug` 仅用于独立短调试，其产物禁止用于正式校准和测试。

### 3.2 校准与先读报告

```bash
python calibrate_taxosafe_new.py --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_reference.yml --trial 1 --seed 1 --variant main --run-dir runs/taxosafe_new/reference/trial_1 --preflight
python -u calibrate_taxosafe_new.py --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_reference.yml --trial 1 --seed 1 --variant main --run-dir runs/taxosafe_new/reference/trial_1
python -m json.tool runs/taxosafe_new/reference/trial_1/calibration/validation_report.json
python -m json.tool runs/taxosafe_new/reference/trial_1/calibration/completed.json
```

校准仅使用原 `val_known`、`val_intra`、`val_extra`，不会把这些图片并入梯度训练。先找四项同时通过的点；若无共同通过点而存在 known 通过点，优先在保住 known 的点中选择。没有共同可行点时，仍保留冻结结果和失败诊断，不能将 `fit_completed=true` 理解成验收通过。

`router.json` 中两个阈值叫 `parent_threshold` 和 `leaf_threshold`，单位是原始成员性 **logit**，不是0到1概率，也不是旧版深度偏置。先固定父排名第一及该父内叶排名第一的候选，再检查对应成员性；分数 **≥** 阈值表示接受，低于父阈值到根，父通过但叶未通过则回退父类。不会为通过门禁更换候选。

默认各深度从本次拟合 Dev 分数生成49个分位数位置，并加入全通过／全拒绝端点后去重，因此实际候选数不固定为49。来源留一的网格和阈值仅从该折保留来源拟合，被留来源不参与该折网格生成。有限分位数搜索不是所有阈值操作点的穷举。

`status=feasible` 表示网格内四项共同通过；`best_effort_known_preserved` 表示未共同通过但保住 known；`best_effort_known_unavailable` 表示网格内 known 也无法通过。`fixed_score_feasibility` 的必要条件只针对当前固定候选、当前原始分数及 membership 解码，不能沿用旧 joint 两偏置的不可行证明；未证明不可行也不等于已证明可行。

四项原门槛保持不变：

| 指标 | 目标 |
| --- | ---: |
| known 端到端正确叶准确率 | **>90%** |
| near 正确父类回退率 | **≥85%** |
| extra 根拒识率 | **>90%** |
| 开放世界叶输出精确率 | **>90%** |

开发集未过 gate 时，先保留报告分析冲突；程序允许对有效冻结产物进行诊断性测试，但这不等于通过研究验收。不要根据最终 Test 结果回调阈值或给某个测试物种增加例外。

### 3.3 冻结测试

模型与校准规则确定后，再执行测试预检和测试：

```bash
python test_taxosafe_new.py --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_reference.yml --trial 1 --seed 1 --variant main --run-dir runs/taxosafe_new/reference/trial_1 --preflight
python -u test_taxosafe_new.py --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_reference.yml --trial 1 --seed 1 --variant main --run-dir runs/taxosafe_new/reference/trial_1
```

Test 只读取冻结 checkpoint、TRAIN 支持库、router 和既定测试清单，不重新拟合支持统计、成员性网络或阈值。指标按唯一图片内容统计，全部清单行另存辅助结果。测试已在历史研发中被查看这一事实仍需披露。

### 3.4 打包结果

```bash
python tools/pack_taxosafe_new_review.py --run-dir runs/taxosafe_new/reference/trial_1
```

将命令打印的 `.tar.gz` 文件发回即可。该脚本只打包指定 run 内诊断文本，不包含图片、模型权重或二进制支持库，附文件 SHA256。必须显式指定本次 reference 路径，打包器原默认路径仍是早期的 `main/trial_1`。校准未过门禁或运行失败的诊断也应保留；没有 `training/config.json` 的预检失败直接提供终端报错。

## 4. 训练日志与结果检查

| 文件 | 应检查的内容 |
| --- | --- |
| `training/config.json`、`inputs.json` | 实际 reference 配置、known-only 输入身份和哈希签名 |
| `training/train.jsonl` | 分类、成员性、`reference_parent`／`reference_leaf` 直接关系损失，配对曝光、已知验证、选择和各阶段计时 |
| `training/model_cost.json` | 参数量、单次查询主干前向设计和实际速度测量边界 |
| `training/completed.json` | 训练完整性、最佳 epoch、checkpoint 与支持库哈希 |
| `calibration/router.json`、`validation_report.json` | 实际 decoder、阈值、网格与 known-first 选择结果、四项是否共同通过 |
| `calibration/development_scores.jsonl`、`development_predictions.jsonl` | 原始开发证据和冻结路由后的逐图片输出 |
| `test/summary.json`、`gates.json`、`metrics.json` | 四项主指标、唯一图片口径、逐物种和来源结果 |
| `test/predictions.jsonl`、`metrics_all_rows.json` | 重复内容权重、实际输出及全清单行辅助统计 |

配对曝光尤其需要区分父层跨物种正对与单叶父类同物种回退，以及叶层兄弟负对与跨父负对。损失权重非零不意味着每个分支都有充分有效监督；不能把缺少跨物种证据的单叶父类描述为已验证跨物种泛化。

对应计数键为 `parent_cross_species_positive_pairs`、`parent_singleton_fallback_positive_pairs`、`parent_other_parent_negative_pairs`、`leaf_positive_pairs`、`leaf_sibling_negative_pairs`、`leaf_other_parent_negative_pairs`。这些是训练中的有效配对曝光计数，不是独立图片或独立未知样本数量。

## 5. 严格 TRAIN 留类验证与重跑

原 `tools/validate_taxosafe_support_holdout.py` 继续用于该折微调阶段完全不见被留出物种／父类的验证。本轮是兼容新方法，没有预先完成全部真实 GPU 留类折。先检查一折计划：

```bash
python tools/validate_taxosafe_support_holdout.py --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_reference.yml --variant main --seed 1 --kind both --max-folds 1 --run-dir runs/taxosafe_new/reference_holdout_smoke/seed_1 --preflight
```

确认计划后，去掉 `--preflight` 才会实际从 CLIP 训练该折。`--max-folds 1` 只覆盖被抽中的一个折，不同时验证两种任务；全部合格折需另用新的目录并去掉该限制。工具只使用 known TRAIN 内部分割，不使用原 known 验证、真实未知 Dev 或 Test；每折模型只能作为留类诊断，不能交给主运行的正式校准／测试入口。

留类工具仍使用 `uncalibrated_joint_argmax` 评估，未在 held-out 查询上拟合 membership 阈值。reference 模式的输出另保留 `support_evidence` 四组原始头分数、`support_candidate_parent`／`support_candidate_leaf` 及其成员性 logit；不活跃节点以 `null` 编码。`membership_thresholds_applied=false` 明确这些只是候选对齐的原始证据诊断，不能把留类结果称为本主解码在真实 unknown 上的正式成绩。

主实验中断或配置改变后，使用独立重跑路径，例如 `runs/taxosafe_new/reference/trial_1_retry`，三阶段保持该路径一致。不要删除旧完成标志或覆盖旧 `training/config.json`。其他种子也须显式带新配置，如 seed 2 使用 `--trial 2 --seed 2 --run-dir runs/taxosafe_new/reference/trial_2`；不按 TEST 挑最好 seed。

## 6. 旧 joint 解码的独立对照

现有六种 `--variant` 保持原样，没有名为 `joint` 的新 variant。要比较旧联合解码，可创建独立配置，仅改变校准解码方式，保留 reference 训练机制。下列命令拒绝覆盖已有同名配置：

```bash
python - <<'PY'
from pathlib import Path
import yaml

source = Path('configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_reference.yml')
destination = Path('configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_reference_joint.yml')
config = yaml.safe_load(source.read_text(encoding='utf-8'))
config['name'] = 'TaxoSafe-Reference-Membership-v3-joint-control'
config['calibration'].update(decoder='joint', grid_points=49, bias_min=-6.0, bias_max=6.0)
config['calibration'].pop('threshold_grid', None)
config['calibration'].pop('membership_grid_points', None)
with destination.open('x', encoding='utf-8') as handle:
    yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
print(destination)
PY
```

在独立目录训练、校准和测试，每个阶段仍须先完成前序阶段并查看开发报告：

```bash
python -u train_taxosafe_new.py --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_reference_joint.yml --trial 1 --seed 1 --variant main --run-dir runs/taxosafe_new/reference_joint/trial_1
python -u calibrate_taxosafe_new.py --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_reference_joint.yml --trial 1 --seed 1 --variant main --run-dir runs/taxosafe_new/reference_joint/trial_1
python -u test_taxosafe_new.py --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_reference_joint.yml --trial 1 --seed 1 --variant main --run-dir runs/taxosafe_new/reference_joint/trial_1
```

该配置对照使用同样训练机制，但属于独立训练 run。因为完整配置已绑定 checkpoint，不能对训练好的 membership run 原地改 decoder 后重新校准；如果要报告严格同一 checkpoint 的解码对照，需要另外实现只读评分比较，而非声称上述命令已经完成该实验。比较方案和 Test 使用计划应预先确定，不凭最终测试结果择优命名主方法。

## 7. 兼容和结论边界

原算法入口与旧默认配置保留，新的成员性路径由新配置选择。当前源码可以为旧算法建立新 run，但旧产物的哈希签名不会自动兼容新源码；历史分析使用对应源码版本。新训练遵守 known-only 梯度，不改变原图、测试清单及旧运行目录。

单次视觉前向、有限参考预算和小型关系头用于控制计算成本，但真实 GPU 速度、显存及四项性能仍待本次实验测量。CPU 测试、预检或冻结校准完成只能证明相应软件步骤完成，不能证明四项 gate 通过，更不能证明达到 SOTA。
