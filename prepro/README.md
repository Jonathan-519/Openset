# TaxoSieve 数据准备

当前构建器为 `build_taxosieve_dataset.py`。图片在 `prepro/data/images/`，当前模型清单在 `prepro/data/taxosieve_v1/`。`prepro/raw` 是兼容旧代码的链接，不是图像副本。

## 数据目录

| images 下目录 | 角色 |
|---|---|
| `Zooplankton_TT` | 23 已知物种，按原身份保留 TRAIN/DEV/TEST |
| `Zooplankton_NearUnknown_Dev` | 4 近域未知开发物种，保留校准与未启用保留池 |
| `Zooplankton_NearUnknown_Test` | 5 锁定近域未知测试物种 |
| `Zooplankton_OOD_Dev` | 4 域外未知开发来源，保留校准与未启用保留池 |
| `Zooplankton_OOD_Test` | 6 锁定域外未知测试来源 |

Known/Near 的目录层级是“父类/物种/图像”，Extra 是“来源/图像”。类别 ID、物种角色和原图像身份由 `protocols/taxosieve_seed/seed.json` 固定。新增物种不会自动猜测其 Known/Near/Extra 角色，而是直接报错，需显式创建经过审核的新 taxonomy/seed 协议。

## 本次版本

| 划分 | 清单条数 | 独立内容数 | 用途 |
|---|---:|---:|---|
| Known TRAIN | 1610 | 1610 | 梯度训练及支持库 |
| Known DEV | 219 | 219 | 模型选择与校准 |
| Known TEST | 451 | 450 | 冻结后测试，1个重复 alias |
| Near DEV | 69 | 69 | 阈值校准 |
| Near TEST | 182 | 182 | 冻结后测试 |
| Extra DEV | 88 | 88 | 阈值校准 |
| Extra TEST | 210 | 210 | 冻结后测试 |
| Near 保留池 | 98 | 98 | 不参与当前模型训练/校准 |
| Extra 保留池 | 127 | 127 | 不参与当前模型训练/校准 |

现有3,073个文件中，19个重复拟合记录明确排除；所有图像均被审计解释。全库存唯一内容为3,053张。`Calanopia_thompsoni` 的原验证图像与测试内容重复，去重后仍无独立验证图像；这一历史限制已在统计中保留，不能据此声称覆盖了全部已知物种的独立验证。相对原3157路径基准缺少84个测试路径：Near22、Extra62。逐路径详情见 `taxosieve_v1/reconciliation.json`。这是一份新版本协议，不能作为原完整TEST的恢复证明。

## 预览、生成与核验

从仓库根执行：

```bash
python prepro/build_taxosieve_dataset.py --dry-run
python prepro/build_taxosieve_dataset.py
python run_taxosieve.py preflight
```

前两条读取图像字节并解码以检查完整性，不执行模型或拟合阈值。已存在且字节完全相同的输出可幂等验证；任何不同的现有输出都拒绝覆盖。默认输出已包含在本仓库，训练前运行完整 preflight 即可。

未来图像库存再次变化时，使用一个新版本：

```bash
python prepro/build_taxosieve_dataset.py \
  --output prepro/data/taxosieve_v2 --dataset-version taxosieve_v2
```

随后将 `configs/taxosieve_reference.yml` 中全部清单、树和 preparation audit 指向同一个新版本，完整预检，并使用新 run。不要把不同版本的清单、审计和模型混用。`--project-root`、`--image-root`、`--seed-dir` 可用于独立项目内的对应路径；构建器拒绝越出项目或经符号链接读取图像。

可选 `--report <新文件.json>` 导出构建报告。`dataset.sha256` 给出每项生成资产的校验值：

```bash
(cd prepro/data/taxosieve_v1 && sha256sum -c dataset.sha256)
```

`inventory.jsonl` 给出每张图的SHA256、角色、分配依据、最终归属、尺寸与重复处理；`split_statistics.json` 给出逐物种数量；`protocol.json` 描述新增图像的确定性分组规则与未知保留池的禁用状态。

## 历史文件

原制作工具已移入 [legacy_tools](legacy_tools/README.md)。旧 v9/v11 清单、旧分类树和 `splits/` 保存历史协议；它们不是当前模型输入。原文件名刷新备份已迁入 `reproducibility/history/filename_refresh/`，基准在 `reproducibility/history/filename_baseline.json`。固定3157图像的纯改名工具不适用于当前数量已变更的数据，不能放宽它的审计门槛来通过。
