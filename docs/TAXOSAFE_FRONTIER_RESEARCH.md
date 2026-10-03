# Frontier：本次问题分析、论文依据与验证边界

日期：2026-10-03。基于 `new` 分支的 ParentRisk 运行记录开展研发。

## 原因定位

ParentRisk 已执行 TRAIN 统计拟合、DEV 内层选择与外层审计，最终回退 reference。
不是代码没有加载，也不是没有搜索。原生产选择包含 1,848 次固定网格候选评估。
父兼容规则的收益伴随至少一个 extra 来源的正确根拒识减少；安全的叶／根拒识
候选则只有一个来源获益，不满足每动作至少两个来源的要求。

另一个可修复的问题是每轴七分位点的候选生成过粗。对于固定的单调矩形拒识
`x <= a AND y <= b`，完整观测边界能够发现分位网格遗漏的受保护可行区域。
但扩大搜索并不等于泛化改善：候选必须继续通过留出 known 与完整未知来源验证。

## 查阅论文及采用边界

1. **Lang et al., From Coarse to Fine-Grained Open-Set Recognition, CVPR 2024.**
   <https://openaccess.thecvf.com/content/CVPR2024/html/Lang_From_Coarse_to_Fine-Grained_Open-Set_Recognition_CVPR_2024_paper.html>
   论文区分语义距离与标签粒度，指出层级表征对粗粒度和细粒度开集的作用不同。
   本项目因此分别计量 near 正确父回退和 extra 正确根拒识；不把增加父级监督
   当成细粒度拒识必然改善的证据。本次没有复现其 hierarchy-adversarial 训练。

2. **Wallin et al., ProHOC: Probabilistic Hierarchical Out-of-Distribution
   Classification via Multi-Depth Networks, CVPR 2025.**
   <https://arxiv.org/abs/2503.21397>
   论文结合各层预测及节点内条件不确定性，把未知样本定位到内部节点。
   它的多深度网络、概率组合和层级距离目标与本项目现有 frozen reference 不同。
   不能将简单父／叶分数组合称为完整 ProHOC 复现，也不能把 near 拒到 root 算正确。
   原论文删去了只有一个子节点的内部节点；本项目必须保留固定 taxonomy，
   因而不能直接照搬这种数据结构变换。

3. **Goren et al., Hierarchical Selective Classification, NeurIPS 2024.**
   <https://papers.neurips.cc/paper_files/paper/2024/hash/c8b100b376a7b338c84801b699935098-Abstract-Conference.html>
   论文研究置信不足时沿分类树降低预测具体程度，以及阈值选择。
   本项目保留“叶→父→根”的动作定义，但仍按本任务严格终端正确性计量：
   known 退到父也算错，near 退到根也算错。本次没有获得论文中的独立校准保证。

4. **Xia & Bouganis, Augmenting the Softmax with Additional Confidence Scores
   for Improved Selective Classification with Out-of-Distribution Data, IJCV 2024.**
   <https://doi.org/10.1007/s11263-024-02029-3>
   会议前身为 ACCV 2022。SIRC 研究主分类置信度与辅助 OOD 证据的非线性组合，
   同时讨论已知正确／错误区分可能轻微退化的局限。它支持检查多证据边界，
   不能替代本项目的逐图保护或证明零损伤。

5. **Angelopoulos et al., Learn then Test: Calibrating Predictive Algorithms
   to Achieve Risk Control, arXiv:2110.01052.**
   <https://arxiv.org/abs/2110.01052>
   借鉴其将候选构造、选择与风险检验区分的思想。本次实现不是其多重假设
   检验程序，不宣称 conformal coverage、风险上界或有限样本统计保证。

## 不允许混淆的四类结论

- 软件测试通过：只能说明被测实现满足相应条件。
- 全 DEV 拟合可行：不说明未参与拟合的样本安全。
- 内层候选准入通过：属于模型选择，外层结果不能反向选择部署规则。
- 四项研究门禁通过：必须直接检查 known、near、extra 和开放叶精确率。

reference 已使用 DEV 选择 checkpoint，而且 DEV/历史 TEST 结果已被多次讨论。
本轮所有新候选选择只使用 TRAIN/DEV；外层输出不参与部署选择。
这些结果仍是对已开发数据的探索性审计，不能恢复成全模型独立或全新 TEST。

## 本轮 DEV 探索结果：未发现合格新规则

所有表中实验均只使用原 TRAIN/DEV 分数，未读取 TEST 进行选择。
保留每动作至少两个获益来源、逐图正确 known 保护、正确 near 回退保护、
near 父路径保护和来源级保护。表内“损失”均指相对同折重拟合 baseline 的新增损失。

| 方法 | 全 DEV 或折内拟合现象 | 内层留出结果 | 准入 |
|---|---|---|---|
| 精确共享矩形保护前沿 | 能发现七分位网格遗漏的双来源收益 | 额外损失 1 张正确 known；near/extra 正确数不增 | 否 |
| 固定安全间隔、TRAIN 已知保护包络 | 可抑制部分危险候选 | 有 known 损失，或无留出收益 | 否 |
| 固定低容量标量融合 | 全 DEV 可减少 2 张错误叶输出，正确 near/extra 不增 | 有收益的候选损伤 known；其余无收益 | 否 |
| 父内条件熵与叶 RMD | 某些拟合折可改善多个来源 | 损失 2 张 known，near 增加 2 张 | 否 |
| TRAIN 类条件收缩经验 CDF，固定收缩量 20 | 全 DEV near 30→32，known 198 保持 | 损失 5 张 known，near 增加 2 张 | 否 |
| 只作用于原 root 的父恢复，固定父候选 | 全 DEV 可恢复 2 张 near，逐张保留正确 extra root | 所有固定评分族均无留出收益 | 否 |

精确前沿是明确的搜索改进，发布代码只保留这个可解释的小改动及其审核机制。
其他失败的评分变体不作为默认算法堆入正式流程，也不通过反复观察外层输出
选择其中一个部署。本轮结果不能证明任意连续阈值、任意新表征或整个任务无解。

**本发布没有找到可诚实标记为“已合格”的新拒识规则。** 不降低双来源要求，
不放宽 known 保护，不手写物种例外，不把全 DEV 的小收益当作泛化收益。
不能承诺解压新版后 near/extra 会改善；相同输入和配置的安全回退会保留 reference。
复现与最终发布代码的结果详见 `docs/FRONTIER_DEV_REPLAY_20261003.json`。

最终代码回放确认：生产规则为空，全 DEV 与 reference 逐图一致。
三个内层动作计划中，叶／叶根组合额外损失 1 张 known，且留出收益来源不足；
根动作没有留出收益。外层四折审计中一折启用了叶规则，总体 known 正确数
195→193、near 正确数 29→30、extra 正确数 70→70，标记为
`heldout_degraded`。这进一步说明拟合可行不等于规则可靠，不能隐去该退化。
外层结果没有参与生产规则选择；生产回退发生于内层审核失败。

## 历史运行保护

新增代码独立放在 `taxosafe_frontier`，入口使用 `python -m taxosafe_frontier`。
不能新增根目录 Python 入口，因为旧 ParentRisk 的源码签名会枚举根目录 `.py`。
旧 support/reference/refine/geometry/parentrisk 代码、导入白名单及签名均保持。
新流程另建 run，重新拟合已知 TRAIN 统计，不需要重训 reference。
不删除旧权重、支持库、完成收据、数据清单，不扫描 raw 重建分割。
