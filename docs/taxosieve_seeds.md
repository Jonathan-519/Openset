# TaxoSieve 多 seed 驱动器

唯一实现是 `tools/run_taxosieve_seeds.py`。根目录的
`run_taxosieve_seeds.py`、`run_taxosieve_seeds_batch.py` 保留为兼容入口，
三个入口支持相同参数。它们运行的是当前主模型的独立重复训练，
不依赖已删除的历史对比实验目录。

## 保持主实验不变

根入口 `run_taxosieve.py` 继续使用固定 seed=1 的原配方。
多 seed 驱动器先复制当前运行源码和配置，仅修改新 suite 的 `runtime/`
副本，使 reference、D05、D05 calibration 和 staged calibration 使用同一个 seed。
TRAIN/DEV/TEST 划分不随训练 seed 改变；数据路径从当前 reference YAML 读取，
使用 `prepro/data/image/` 和 `prepro/data/` 下的主实验标签及审计文件。

源码、输入清单、配置和独立 `driver.py` 都写入 SHA256 冻结记录。
每次执行阶段之前、之后均校验；阶段回执进一步比对各 seed 实际解码图像的身份。
因此准备 suite 后，不要改动其输入数据或配置，不要移动该 suite 的绝对路径。

## 启动新的同配方重复

先按根目录 README 完成数据构建和预检，然后运行：

```bash
python -u tools/run_taxosieve_seeds.py --seeds 2 3 4 5
```

默认图像 batch=12，每轮240批，每批2个父类×3个物种×2张图片。
默认输出为新目录 `runs/taxosieve/seed_sweep_<时间>_<标识>/`。
旧命令 `python -u run_taxosieve_seeds.py --seeds 2 3 4 5` 仍可用。

可以先准备快照，不启动训练：

```bash
python tools/run_taxosieve_seeds.py --seeds 2 3 4 5 --prepare-only
```

终端会打印完整的 `python <suite>/driver.py --suite <suite>` 命令。
此 `driver.py` 是完整实现的冻结副本，启动时不依赖仓库中的 wrapper 或 tools 模块。
`--suite` 只用于**已经准备但从未启动**的 suite，不能与准备阶段参数混用。
失败或部分完成的 suite 不会被自动覆盖、自动重训或自动续跑；保留其日志与原始回执。
检查已有 seed 时，应使用它自己的 `runtime/run_taxosieve.py inspect --run-dir ...`。

## 保留 batch 参数

```bash
python -u run_taxosieve_seeds_batch.py \
  --seeds 8 18 28 38 48 --train-batch-size 24
```

这一可选设置只修改新快照 reference YAML 的三项：

| 设置 | 默认 batch=12 | batch=24 |
| --- | ---: | ---: |
| `data.batch_size` | 12 | 24 |
| `data.sampler.images_per_species` | 2 | 4 |
| `data.sampler.batches_per_epoch` | 240 | 120 |
| 每轮抽样图像数 | 2880 | 2880 |

图像评估 batch、D05 特征 batch、学习率、损失和校准门槛保持原值。
batch=24 减少每轮优化步数，不能视为只改变 seed 的同配方重复，
也不能承诺提升准确率或拒识率。`reference_recipe` 明确记录有效批量与预算，
默认输出为 `runs/taxosieve/seed_batch24_<时间>_<标识>/`。
合法 batch 为不小于12、是6的倍数且整除2880的整数。

## 选择与输出

执行顺序固定为：所有 seed 完成 TRAIN/DEV → 按 DEV-OOF 冻结选择 → 所有 seed 执行 TEST。
选择要求 OOF 完整且四项指标为有效的0–1值；优先四门全过，随后按四指标最小值、
Near、Known、Extra、Leaf PPV 依次排序，完全并列取较小 seed。TEST 不参与程序选择。

`selection.json` 记录冻结推荐；`comparison.csv/json` 仅是同一主模型各次训练的结果汇总，
名称保留以兼容原有结果导出流程。其内容包括 DEV-fit、DEV-OOF、TEST、批量信息、
各项 TEST 均值和样本标准差。缺失指标保留为空值，不计为0。
不应把某个 TEST 最优 seed 的人工选择写成 DEV-OOF 的程序推荐。

本版本可读取原有 v1/v2 suite 的元数据，但历史运行应优先使用其原始 `driver.py`
和 `runtime/`。源码整理或数据重命名会改变来源签名，不能把新代码覆盖到旧 suite
来假装延续旧实验身份。
