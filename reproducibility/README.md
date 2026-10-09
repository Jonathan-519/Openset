# H02 复现资产与验证边界

这里保留有效模型参数和紧凑来源证据，避免把历史运行目录整体放回主目录。

| 文件 | 用途 |
| --- | --- |
| `reference_router.json` | 原 reference 的冻结 membership router；保留原字节及来源绑定 |
| `d05_router.json` | 原 D05 全局 router；H02 根预算和折内比较的来源 |
| `h02_router.json` | 原 H02 分阶段根/叶阈值及完整根状态 |
| `h02_expected.json` | 固定提交、原权重 SHA、锁定数据、历史指标和 16 份大回放输入的精确身份 |
| `verification.json` | 完整分数/预测/OOF 回放的紧凑验证证明 |

原来源提交为 [abe3a982c4a6236cfbd57d7ca4169a6f5117950f](https://github.com/Jonathan-519/Openset/tree/abe3a982c4a6236cfbd57d7ca4169a6f5117950f)。完整 DEV / TEST 分数、逐图输出及 11 折审计由工具按固定提交按需获取：

```bash
python tools/verify_taxosieve_replay.py --golden-dir /tmp/openset_h02_golden
```

工具校验每份文件的字节数、Git blob SHA 和 SHA256，再比较全部增强分数、完整 D05/H02 routers、完整 DEV 拟合报告、DEV 376 行 / TEST 927 行预测，以及完整 11 折 OOF 报告。回放 TEST 按 926 张独立图计量。

## 原权重来源

| 模型文件 | 原 run 内路径 | SHA256 |
| --- | --- | --- |
| reference checkpoint | `reference/trial_1_retrain_20261003_185220/training/best.pth` | `e4c39c378cb3685533ebc76658338b8aed09cc52662d8967cca113d14a279775` |
| reference support 库 | 同 run `training/support.pth` | `bdb8ee6213d89260d0443d23ae48afc0a679049f5905c7eff174a8c759f1d89f` |
| D05 model | `discovery/trial_1_20261005_175231/arms/D05_episode_bce/training/model.pth` | `1f6fe81a909eb8c848bf5e7fd5e99568d3114cd5f9956c3826fdc96a86027a94` |

路径均以原 `runs/taxosafe_new/` 为前缀。原 Git tree 没有这些 checkpoint，只有其收据和结果；需要原服务器资产或按主 README 重训。`h02_router.json` 只能解码已计算的 H02 分数，无法单独替代图像编码器和 D05 模型。

## 本次确认到哪一层

- 原 reference 与精简内核，在独立 CPU 进程中验证了相同初始化/RNG、前向、损失、梯度及合成数据上的真实训练/校准/测试；17 组快照中 642 个 tensor、1 个 ndarray、1322 个标量逐位相等。
- 原 D05 与精简实现的 episode 输入/权重/顺序、BCE 权重、归一化、geometry 和逐记录评分一致；旧 payload 通过纯加载恢复。
- H02 完整冻结分数回放和全部 conditional OOF 报告与原结果一致。
- 新阶段接口在合成数据上完成训练到 TEST，并验证来源/模型/缓存/阈值被修改时拒绝继续、TEST 不参与拟合，以及 alias 的独立图像计量。
- 数据制作脚本以临时数据检查输出顺序、去重和树序列化；35 份原 `data/splits` 文件字节未变，原图在 Git tree 中按原 blob 保留。

本次没有 GPU 和原模型 checkpoint，未执行原图上的完整神经网络推理或全量 GPU 重训。原四项研究指标联合门槛未通过的结果保留。复现配置、算法和已有结果的证据与“重新训练得到同一权重”属于不同验证范围，报告中均明确记录。

## TaxoSieve 数据重建后的边界

以上均为旧 H02 数据及历史验证，不能当作 TaxoSieve 新数据版本的结果。旧来源提交当前不可下载；上述回放命令需要已保存且哈希匹配的黄金输入。原文件名刷新基准与完整备份迁至 `history/filename_baseline.json` 和 `history/filename_refresh/`，内容未改。当前数据统计见 `../prepro/data/taxosieve_v1/`，本次工程验证见 `taxosieve_validation.json`。
