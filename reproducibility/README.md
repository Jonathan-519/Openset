# 验证资产与适用范围

## 本轮目录与数据重构

本轮基点为 `638853b47b0e7b8d2d7fe2ad387e030b255512c8`。最终验证记录见 `main_clean_validation.json` 和 `main_clean_tests.log`。

本轮目标是保持当前 TaxoSieve 主实验，不更改模型或阈值算法。主数据清单逐行保留上一轮现役图像的内容身份、标签和顺序；所有当前图像字节保留。补回的84张历史测试图有单独清单，默认主实验不使用。三组 OOD 来源按原内容角色归位。当前数据统计见 `../prepro/data/`。

本轮验证包括全图解码和身份核查、主清单逐行对账、reference 来源摘要、核心代码/配置等价性，以及 CPU 上的 D05、校准、OOF、TEST、收据/缓存和多 seed 快照回归。没有在目标 GPU 上全量重训，也没有据此声称新准确率或 seed5 的 GPU 结果逐位复现。

历史对比实现和配套数据已从本分支删除。可选数值等价测试需显式提供一个外部原始快照：

```bash
TAXOSIEVE_ORIGINAL_SOURCE=/absolute/path/to/original/source \
  python -m unittest discover -s tests -p 'test_taxosieve_reference.py' -v
TAXOSIEVE_ORIGINAL_SOURCE=/absolute/path/to/original/source \
  python -m unittest discover -s tests -p 'test_taxosieve_d05.py' -v
```

此路径须直接包含 `taxosafe_support/`、`taxosafe_discovery/` 等原始模块；比较使用相同的当前分类树字节。没有设置外部快照时，相关归档比较测试明确跳过；指定无效快照则报错。`H02_ORIGINAL_SOURCE` 是同义环境变量，也是历史 `import-d05` 桥使用的变量。该桥继续校验原 Discovery 全量代码摘要以及模型、缓存、收据链，不放宽验证。

## 保留的历史 H02 回放证据

以下文件保持原字节，记录的是旧数据上的历史验证，不是本轮训练结果：

| 文件 | 用途 |
|---|---|
| `reference_router.json` | 历史 reference 冻结 membership router |
| `d05_router.json` | 历史 D05 全局 router |
| `h02_router.json` | 历史 H02 分阶段根/叶阈值 |
| `h02_expected.json` | 固定提交、权重和黄金输入的精确身份 |
| `verification.json` | 历史完整分数、预测及 OOF 回放证明 |
| `taxosieve_validation.json`、`taxosieve_tests.log` | 上一轮数据工程验证的时间点记录 |

历史来源提交为 `abe3a982c4a6236cfbd57d7ca4169a6f5117950f`。完整回放输入并未保存在本分支；历史下载提交此前已不可访问，应优先使用已保存且身份匹配的黄金输入：

```bash
python tools/verify_taxosieve_replay.py --golden-dir /absolute/path/to/saved/golden
```

工具要求精确匹配的字节数、Git blob 和 SHA256，验证 DEV 376 行、TEST 927 行及11折报告。这里的历史 TEST 是926张独立图，与当前默认 TEST 842张不同。下载失败或缺少黄金输入不表示主训练失败；不能用当前数据替代黄金文件。

历史reference/D05权重不在仓库中，router也不能替代编码器与验证器权重。历史数据/文件名备份已经删除，原内容可由本轮基点追溯；保留的冻结seed已自包含数据重建所需身份与分类树，不依赖这些旧备份。
