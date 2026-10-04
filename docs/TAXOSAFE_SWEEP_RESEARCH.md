# Reference 微调消融：证据、文献依据与结论边界

本文件记录 2026-10-03 的研究设计。六项固定实验以同一套已训练 reference 为起点，比较更新范围、reference 蒸馏、已有层级损失权重和权重插值。它们是待服务器验证的工程消融，不是 KgCoOp、PromptSRC、ProGrad 或 WiSE-FT 的完整复现；目前没有产生新的合格模型或证明四项验收门禁得到改善。

## 为什么从冻结后处理转向受约束微调

依据是用户提供的 `trial_1_20261003_185220` review 包中的 `parentrisk/train_scores.jsonl`、`parentrisk/{fit_report,scales,source_binding}.json`、`calibration/development_scores.jsonl`、逐图预测和折计划。本节计数仅来自 TRAIN/DEV 读取与重放；没有使用 TEST 选择参数。

| 已核验的项目 | 结果及含义 |
|---|---|
| TRAIN 与支持库 | 1,610 张 known TRAIN，支持库 180 张。180 张 TRAIN 查询各排除 1 个相同内容参考，其余 1,430 张排除 0 个；最少有效叶参考数为 5。DEV 与支持库内容重合为 0。 |
| 统计来源 | 七个 robust scale 的重算结果与存档一致；TRAIN 哈希、源绑定及签名一致。父文本、父 membership、全父 RMD 均使用 TRAIN 尺度。 |
| DEV 组成 | known 219、near 69、extra 88，共 376 张唯一内容图片。 |
| 原 DEV 最终输出 | known 正确叶 198/219，near 正确父回退 30/69，extra 正确根拒识 71/88；全部叶输出 229 张。 |
| near 错误的互斥分解 | 父候选错误 9；父候选正确但输出根 8；父候选正确但误收为叶 22；正确父回退 30。仅修复父候选不能解决所有错误。 |
| 父排名的互补性 | 支持与文本的 known top-1 都为 217/219，near top-1 都为 60/69，正确集合完全相同。near top-2：支持 69/69、文本 64/69。独立文本 top-1 没有修复原来的 9 张父候选错误。 |
| 父绝对兼容性 | 在原支持 top-1 父上、门控之前，known+near 288 张为正、extra 88 张为负的 AUROC：文本 0.6293、membership 0.9360、RMD 0.9306。此结果只评价这个候选选择与正负定义，不是父排名准确率或最终开放集准确率。 |
| 原固定 ParentRisk 搜索 | 4 个父权重族 × 3 个内层折，每次搜索 154 个配置；共 1,848 个候选重放仍无符合保护约束及跨来源支撑条件的新规则，生产结果回退到 reference。 |

这些证据支持“当前冻结文本证据没有提供预期的独立修正信号”，不能推出“父语义全面无效”。例如固定文本+RMD 融合曾在全 DEV 排名中修复 1 张原根输出，但这不是文本 top-1 独立纠错，也没有形成通过原嵌套协议的生产规则。另一个原根输出救援原型在全 DEV 有 2 个 near 修复来源，内层折外没有收益、外层仍回退；不能把全 DEV 的拟合收益作为新方案有效的证据。

当前瓶颈因此不仅是阈值搜索：相同表示下的证据相关、near/extra 的局部重叠，以及少量来源上的收益，使零新增已知伤害与至少两个未知来源支撑难以同时满足。扩大网格、针对某个物种写规则或放松保护不等于获得新表示能力。

## 原代码已经具备的能力

`taxosafe_support/pipeline.py` 已有叶/父文本 CE、支持分类与 membership 监督、留叶/留父 episode、pair/control 项。`representation_losses` 已包含同父不同叶的父层正对与同父兄弟叶分离。原 reference 配置已经使用这些项，所以本次 hierarchy arm 是**既有损失的固定权重消融**，不应称作新增层级损失。

原训练的 `anchor` 为 warmup 后缓存的叶文本 logits 蒸馏（温度 2）；它并非原始 zero-shot CLIP，也并非完整已训练 reference 的父/叶表示与支持置信度教师。六项实验使用的教师必须明确标记为 **当前已训练 reference**，不能混写成“恢复 CLIP 预训练知识”。

父与细粒度分支虽然有独立 adapter/token pooling，仍共享 MaPLe 图像路径与 prompts；解冻共享 prompts 可以同时改变两个分支。该 taxonomy 有 7 个父、23 个叶，其中 4 个父只有一个已知叶；这些父没有真实“同父不同叶”的正对，不能把重复图或同叶对伪装成跨物种监督。

## 一级文献与借鉴范围

### KgCoOp，CVPR 2023

Yao 等，*Visual-Language Prompt Tuning with Knowledge-guided Context Optimization*。

- [CVF 论文页](https://openaccess.thecvf.com/content/CVPR2023/html/Yao_Visual-Language_Prompt_Tuning_With_Knowledge-Guided_Context_Optimization_CVPR_2023_paper.html)
- [作者提交的全文，§3.2、式 (3)–(5)](https://arxiv.org/html/2303.13283v1)
- 方法短摘录："minimize the distance"。

其正则项是已知类别的学习文本 embedding 与固定手工 prompt 的 CLIP embedding 之间的平均平方欧氏距离，和监督分类项共同优化。它说明冻结 backbone 并不自动阻止 prompt 过拟合，也提供了保持已有语义参考的思路。但原教师为 zero-shot CLIP 文本，不是本项目训练后的 reference；本次没有实现其文本 embedding 正则，不能标记为 KgCoOp 复现。论文的 base-to-new 分类结果也不能替代本项目 near 父回退、extra 根拒识和开放叶精确率评价。

### PromptSRC，ICCV 2023

Khattak 等，*Self-regulating Prompts: Foundational Model Adaptation without Forgetting*。

- [CVF 论文页](https://openaccess.thecvf.com/content/ICCV2023/html/Khattak_Self-regulating_Prompts_Foundational_Model_Adaptation_without_Forgetting_ICCV_2023_paper.html)
- [作者提交的全文，§3.2.1–3.2.3、式 (2)–(7)](https://arxiv.org/html/2307.06948v2)
- 方法短摘录："feature and logit levels"。

原方法以冻结、未加学习 prompts 的 CLIP 为参考，对图像和文本特征施加 L1 一致性、对 logits 分布施加 KL；另外包含多文本模板和沿训练轨迹的 Gaussian prompt 聚合。该工作为特征/预测共同约束提供依据，但本次用已训练 reference 蒸馏，没有实现完整的 CLIP 特征正则、多模板或 Gaussian 聚合。原文表 1、表 3 也有未胜过 MaPLe 的设置，不能将平均泛化收益解释成每个类别、每张图片或本项目四门禁都不会退化。

### ProGrad，ICCV 2023

Zhu 等，*Prompt-aligned Gradient for Prompt Tuning*。

- [CVF 论文页](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_Prompt-aligned_Gradient_for_Prompt_Tuning_ICCV_2023_paper.html)
- [作者全文，§3.2、式 (3)–(4)](https://arxiv.org/html/2205.14865v4)
- 方法短摘录："orthogonal direction"。

它计算任务 CE 梯度与 zero-shot CLIP 分布 KL 梯度；冲突时从前者移去沿参考梯度的冲突分量。其启发是显式控制适配与保留的冲突，不是只依靠早停。本次六项实验采用普通损失蒸馏，**未实现梯度投影**。即使实现投影，局部梯度条件也不等于有限步优化之后、尤其未见图片上的逐图零伤害保证；参考教师若本身错误也可能限制必要修正。

### WiSE-FT，CVPR 2022

Wortsman 等，*Robust Fine-Tuning of Zero-Shot Models*。

- [CVF 论文页](https://openaccess.thecvf.com/content/CVPR2022/html/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.html)
- [作者代码及插值公式](https://github.com/mlfoundations/wise-ft)
- 方法短摘录："ensembling the weights"。

作者实现检查两个 checkpoint 的参数键一致，再计算 `theta = (1-alpha)*theta_0 + alpha*theta_1`。本次仅借鉴同构权重插值：固定 alpha=0.5，端点为任务 reference 和 E04 学生，而非论文的 zero-shot 与 fine-tuned 模型。不能将缺少相同 prompts/adapters 参数的原始 CLIP 与学生直接相加。插值后的编码空间已变，必须重新生成支持库并重新进行 DEV 校准，不能复用端点的缓存。本套件不使用 RMD；若以后追加几何模块，其 TRAIN 几何/尺度也必须重新拟合。插值不保证损失、置信度或准确率单调改善。

### 层级监督与细粒度 OSR，CVPR 2024

Lang 等，*From Coarse to Fine-Grained Open-Set Recognition*。

- [CVF 原论文](https://openaccess.thecvf.com/content/CVPR2024/html/Lang_From_Coarse_to_Fine-Grained_Open-Set_Recognition_CVPR_2024_paper.html)
- [作者项目页](https://langnico.github.io/fine-grained-osr/)
- 发现短摘录："little effect on fine-grained OSR"。

该研究区分粗粒度和细粒度未知偏移：层级表示有助于前者，对后者收益有限，并研究通过 hierarchy-adversarial 学习减弱表示中的层级结构。它支持分别衡量父归属与兄弟叶拒识，而不是假定增大父层损失必然改善两者。本次仅调整现有父/叶表示损失权重，没有移植 hierarchy-adversarial 训练，也不应让细粒度抗层级目标直接破坏父分支任务。

## 六项固定实验及可回答的问题

唯一配置为 `configs/Zooplankton_Taxonomic_Tree/TaxoSafe_reference_sweep.yml`；固定次序由 `taxosafe_sweep/protocol.py::ARMS` 校验。所有适配从相同已训练源 checkpoint 独立启动，不能串联 E01→E02→E03；E05 明确使用 E04 端点。

| ID | 固定变化 | 主要比较与解释 |
|---|---|---|
| E00_reference | 不进行参数更新，保留冻结 reference 对照 | 原模型与原路由的可复现基线，不是新增训练预算的结果。 |
| E01_heads | 冻结共享 prompts/backbone，更新 parent/fine adapter 分支与 evidence heads；原损失 | 测试受限更新是否已足够；同时提供继续训练、无新增 reference KD 的对照。 |
| E02_heads_anchor | E01 的更新范围，加入已训练 reference KD | E02 对 E01 隔离 reference 蒸馏的影响。 |
| E03_prompt_anchor | E02 加入名称匹配 `prompt_learner`/`VPT` 的 prompts 参数，其他 CLIP 权重保持冻结 | E03 对 E02 测试共享 prompt 适配的收益与伤害。 |
| E04_hierarchy_anchor | E03 的 `parent_cross_species`/`leaf_sibling` 权重设为 0.3、`pair_parent`/`pair_leaf` 设为 0.5 | E04 对 E03 评估既有损失平衡；不是新造层级监督。 |
| E05_hierarchy_blend | E00 与 E04 同构权重固定 0.5 插值 | 测试缩小参数偏移是否改善保留/适配权衡；不再训练，不按结果选择 alpha。 |

默认最大训练预算为 20 epochs、patience 6、至少 3 epochs、每 epoch 240 batches；prompt LR=0.0005、head LR=0.001，seed=1。E01–E04 使用相同预算上限和 checkpoint 选择协议；实际早停可能导致更新步数不同，应在报告中保留。此设计是固定的单因素递进比较，不是超过六项的超参数搜索。

E01–E04 均保留初始 reference 对干净 TRAIN 图片的 leaf-logit anchor，权重 0.5、温度 2；因此 E01 的“无新增 reference KD”不代表完全没有原 anchor。E02–E04 额外让冻结教师与学生读取**完全相同的增强图片**，使用以下固定项：

```text
K = mean over {leaf_logits,parent_logits} of
    [T^2 * KL(softmax(teacher/T) || softmax(student/T))], T=2
F = mean over {fine,parent} of [1 - cosine(student,stop_gradient(teacher))]
L = L_existing_including_initial_anchor + 0.5*K + 0.5*F
```

这里每个 KL 使用 batchmean，特征距离对 batch 求均值；教师为当前已训练 reference 的 encoder，额外蒸馏未直接调用其 evidence heads。具体 targets、reduction、温度、权重与更新组由 `taxosafe_sweep/training.py` 及签名绑定日志记录。softmax KL 保留相对类别分布，**本身不保留独立 membership logits 的绝对标度，也不保留更新后的 RMD**；需要分别检查这些通道漂移并重建校准。蒸馏仍可保留教师错误，强约束也可能使改进空间不足。

E01–E04 的主要 checkpoint 从真正更新后的 epoch≥1 中选择；同时记录包含原始 epoch 0 的比较。这样可以如实暴露微调退化，不把未更新的源权重重新命名为“成功微调”；最终 DEV 选择仍可以推荐 E00。KD 臂额外有一次同增强教师视觉前向，训练成本应单独记录，不能借用原论文的速度数字。

## 评价边界与执行约束

1. 只有 known TRAIN 可以参与参数梯度和参考支持库构建。使用预先锁定的内容哈希划分；所有支持查询先排除相同内容。DEV 用于预先声明的模型/路由选择，TEST 不参与选择臂、epoch、损失权重、插值系数或阈值。
2. 每个编码器状态必须有自己的支持库和 DEV 产物；若另行启用统计后处理，还需重拟合相应 TRAIN 统计，记录源 checkpoint、配置、代码和数据签名。权重变化后复用旧特征会造成混合空间，得到的“改善”不可解释。
3. 首先报告实际分子/分母、原正确 known 被破坏的逐图计数、near 父路径损失、extra 根拒识变化，以及逐来源覆盖；再报告四门禁。研究目标保持 known 正确叶 >90%、near 正确父回退 ≥85%、extra 根拒识 >90%、开放叶精确率 >90%。不以总体平均分掩盖某一门失败。
4. 这六臂是在已经审阅过当前 DEV 的基础上提出的预设下一轮实验，不是对当前 DEV 的完全盲预注册。源 reference 已利用 known DEV 选择 checkpoint，微调又利用 DEV 选择 epoch/臂，所以后处理折外审计不构成整个模型的独立验证。严格未见物种研究需要折内排除该物种的梯度、教师来源、支持和统计，不能仅在 decoder 中遮蔽类别后宣称实现。
5. 没有任何正则项或文献可以担保新模型保留每张原正确图片；保护约束需要在可用开发证据上逐图审计，并把有限样本结论与泛化声明分开。不能根据 TEST 退化再调强 KD 后把同一 TEST 当作独立验证。
6. 当前助手工作环境没有该真实 reference 的 `best.pth`/`support.pth`，且没有可用 CUDA。review 文本不能重建模型。本文的实测计数属于原 TRAIN/DEV 审计；六臂的 GPU 图像训练、收益、速度和四门禁均尚未在本环境实测。CPU 单测通过只能证明被测试的软件契约。

若六臂都没有通过既定约束，应保留失败与回退结论，而不是将最大已知准确率、某张修复样例或某个全 DEV 阈值作为新方案合格的证据。
