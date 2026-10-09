# 对比实验源码

`legacy/` 保存清理前提交 `abe3a982c4a6236cfbd57d7ca4169a6f5117950f` 中其他实验的 Python 源码、YAML 配置、测试与旧 CLI，并保留原相对目录及文件字节。当前主实验名称为 **TaxoSieve**，入口为根目录 `run_taxosieve.py`；其算法沿用原 H02 的 D05 支持干预验证器与分阶段层级拒识。此处用于历史算法对照和原模型契约校验，不接管新的数据制作。

**历史数据版本与新数据版本分开。** 当前图像存放在 `prepro/data/images/`，新主实验清单存放在 `prepro/data/taxosieve_v1/`。旧配置继续绑定各自的 v9/v11 等历史清单；不能把新清单直接覆盖旧清单，也不能把旧指标当作新样本量下的结果。2026-10-08 交接文档记录旧 TEST 清单引用的 84 张图像不在当时仓库内，当前数据重建不代表这些原字节图片已恢复。运行历史比较或导入旧模型前，仍须核验所需原图片、模型、缓存、收据和数据身份。

## H04_subspace_root 在哪里

| 内容 | 路径 |
| --- | --- |
| H04 / H00–H11 实验定义 | [TaxoSafe_parent_domain.yml](legacy/configs/Zooplankton_Taxonomic_Tree/TaxoSafe_parent_domain.yml) |
| subspace / PPCA / domain bank | [taxosafe_domain/core.py](legacy/taxosafe_domain/core.py) |
| 训练、来源继承和评分分派 | [taxosafe_domain/backend.py](legacy/taxosafe_domain/backend.py) |
| 12 臂实验协议与阶段运行 | [protocol.py](legacy/taxosafe_domain/protocol.py)、[runner.py](legacy/taxosafe_domain/runner.py) |

其余 D00–D10、Recovery、Boundary、RouteAlign、Geometry、Refine、Frontier、Local、ParentRisk、Morphology、Evidence/Joint，以及更早的 TaxoLocal / v10 / DCBS 等源码均在 `legacy/` 的原同名包内。旧配置集中在 `legacy/configs/`，旧测试集中在 `legacy/tests/`。

## 运行方式

用独立子进程切换到旧源码目录，避免和主目录同名 `models` / `loader` / `taxosafe_support` 混用：

```bash
python comparison_experiments/run.py -m taxosafe_domain --help
python comparison_experiments/run.py -m taxosafe_discovery --help
```

例如运行原 Domain 对比 suite：

```bash
python comparison_experiments/run.py -m taxosafe_domain \
  --discovery-run-dir /home/ubuntu/hdd/data/qz/Openset/runs/taxosafe_new/discovery/trial_1_20261005_175231 \
  --run-dir /home/ubuntu/hdd/data/qz/openset_comparison_runs/domain_01 \
  --device cpu
```

原 suite 仍使用其原有的多臂实验协议，不能把内部 worker 的 `--arm` 当作已经支持的独立单臂正式运行接口。要运行当前主实验，使用根目录的 `run_taxosieve.py`。

所有 source run 参数推荐使用绝对路径，输出使用独立新目录。旧版本运行所需的模型/缓存应由原服务器提供；此处不附带历史模型文件。对比实验额外依赖见本目录 `requirements.txt`，它引用根目录的主运行依赖清单。根依赖清单根据当前源码导入补全，不是已验证的历史环境锁；优先保留已能运行的 ProTeCt、torch/torchvision/CUDA 组合。正式历史权重导入由 `run_taxosieve.py import-d05` 调用 `_export_d05.py`，该进程只进行原契约校验和必要资产导出，不训练其他模型。历史模型的数据绑定必须与其对应配置完全一致；新的 TaxoSieve 数据集需要新的训练目录。

## 为什么保留旧的同名基础模块

原始 source signature 会对完整 Python 依赖及其相对路径计算哈希。归档保存当时的基础模块，使旧 source 合法校验、旧新数值对照和对比实验仍有明确基准。当前主目录则使用收窄后的 reference / D05 / H02 实现。Git 以相同 blob 去重相同字节。

交接文档记录归档 reference 签名和 Discovery code signature 的历史验证值为：

```text
Discovery code:
ec445a6908d0e56d14060e4a2ad9561dcbcd9421fdf34877bc3de82f2059fe78

Reference code:
7b6b7e02dc7b0181d93467e6d5c25bc1e42823d3d3c2c90088c28e8aa66b671b

Reference support_code:
77e239bed3c7a8c89d1fbcfaa8da0d097c202a26a68013dbeb644dba8f22faae
```

`legacy/prepro/` 的原制作脚本独立保留。以下四个相对符号链接已恢复，以共享资产并保留原代码路径：

| 归档路径 | 相对链接目标 |
| --- | --- |
| `legacy/prepro/data` | `../../../prepro/data` |
| `legacy/prepro/splits` | `../../../prepro/splits` |
| `legacy/prepro/raw` | `../../../prepro/data/images` |
| `legacy/models/bpe_simple_vocab_16e6.txt.gz` | `../../../models/bpe_simple_vocab_16e6.txt.gz` |

符号链接仅修复资源入口，不重建历史缺失图片或替换历史数据绑定。主目录新改动的数据 CLI 不覆盖归档中的原算法。仅供旧实验的数据制作入口整理到 `prepro/legacy_tools/`；保留的历史工具与 `legacy/` 内原始源码分开，避免破坏源代码签名。

历史 `runs/`、大量重复 JSON/JSONL、日志及一次性 verification 报告不在此目录中。2026-10-08 交接复核时，上述原提交已无法从远端读取；其 SHA 仅用于溯源，不能保证还能下载历史输出或黄金回放输入。需要原服务器或其他已核实的备份提供这些资产。文件名刷新记录及原清单备份保存在 `reproducibility/history/`，不作为新实验结果使用。主目录默认测试覆盖当前主链和数据制作代码；归档测试由对比任务自行显式选择。
