# MemEvolve 题目一实验与分析报告

## 1. 环境与方法理解

论文来源：[MemEvolve: Meta-Evolution of Agent Memory Systems](https://arxiv.org/abs/2512.18746)，本地 PDF 为 `D:\cdl\sai\2512.18746v1.pdf`。

已完成的环境工作：

- 使用已有 conda 环境 `MemEvolve`，核心依赖可导入。
- 下载 xBench 官方仓库到 `xbench-evals-main`，并复制 `DeepSearch-2505.csv` 到 `data/xbench/DeepSearch.csv`。
- 安装 Playwright Chromium，使 `WEB_ACCESS_PROVIDER=crawl4ai` 可用，绕过 Jina 402 配额问题。
- 使用 `SearchSwarm\harness\.env` 中的兼容 OpenAI-style API 配置映射到 Flash-Searcher 所需环境变量；未写入或打印密钥。
- 修复 `eval_utils.generate_unified_report()`：xBench 使用 `score` 字段而非 `judgement` 字段，原报告会把正确结果统计为 0。

MemEvolve 的核心方法：

- EvolveLab 将 memory system 统一成四个模块：`Encode`、`Store`、`Retrieve`、`Manage`。
- 内循环是 experience evolution：候选 memory system 在任务轨迹上更新 memory base，并产生成功率、token、延迟等反馈。
- 外循环是 architecture meta-evolution：用 Pareto 排序选择候选，再通过 diagnosis-and-design 修改四个模块，生成新 memory architecture。
- 论文中的关键观点是：memory 不只应积累内容，也应进化“如何编码、存储、检索、维护经验”的架构。

## 2. 实验设置

数据集：xBench-DeepSearch，前 20 条任务。

运行框架：Flash-Searcher + DeepSeek-V4-Flash。

统一参数：

- `--task_indices 1-20`
- `--max_steps 12`
- `--concurrency 1`
- `--judge_model deepseek-v4-flash`
- `WEB_ACCESS_PROVIDER=crawl4ai`

三个 memory system：

| Memory system | 来源 | 说明 |
| --- | --- | --- |
| `lightweight_memory` | MemEvolve 自动进化 | stage-aware 的轻量记忆，包含短期工作记忆、战略/操作记忆 |
| `agent_kb` | EvolveLab baseline | hybrid DB / semantic retrieval 类型 |
| `dynamic_cheatsheet` | EvolveLab baseline | 将经验压缩成可复用 cheatsheet |

结果文件：

- `xbench_output/formal_lightweight_memory.jsonl`
- `xbench_output/formal_agent_kb.jsonl`
- `xbench_output/formal_dynamic_cheatsheet.jsonl`
- 汇总报告：`xbench_output/formal_summary/`

## 3. 实验结果

| Memory system | 正确数 | Accuracy | Total tokens | Avg tokens/task | Total time | Avg time/task | API calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lightweight_memory` | 17/20 | 85.0% | 3,748,427 | 187,421 | 100.0 min | 299.9 s | 443 |
| `agent_kb` | 17/20 | 85.0% | 2,944,563 | 147,228 | 85.7 min | 257.0 s | 290 |
| `dynamic_cheatsheet` | 18/20 | 90.0% | 3,651,135 | 182,557 | 77.4 min | 232.1 s | 337 |

错题分布：

| Task id | `lightweight_memory` | `agent_kb` | `dynamic_cheatsheet` | 题目类型 |
| ---: | --- | --- | --- | --- |
| 5 | 正确 | 正确 | 错误 | 标准/羽毛球数量推理 |
| 10 | 错误 | 正确 | 错误 | 北京三点等距地理计算 |
| 16 | 正确 | 错误 | 正确 | 视频/综艺人物定位 |
| 17 | 错误 | 错误 | 正确 | 历任校长姓氏统计 |
| 19 | 正确 | 错误 | 正确 | CVPR 论文数量检索 |
| 20 | 错误 | 正确 | 正确 | QS 排名年份对比 |

## 4. 成功与失败 case 分析

Memory 带来收益的情况：

- 多跳但结构清晰的任务。Task 2 的 Intel Gen9 FP32 算力需要识别公式 `EU * frequency * FP32 ops/cycle`，memory 记录中间公式和验证线索后，后续推理更稳定。
- 需要保留中间发现的长链搜索。Task 5 中 `lightweight_memory` 记录“羽毛球每个 16 根羽毛、左右翅刀翎限制、向上取整”等中间事实，最终得到 12；`dynamic_cheatsheet` 则用了“每鹅 20 根羽毛”的宽泛经验，得到 8。
- 需要实体消歧。Task 16、19、20 中，动态或短期 memory 能保存“最佳喜剧小队/第一轮作品/最后出场演员”、“韩晓光 CVPR2025”、“CUHK 2025 QS 36、2020 QS 46”等已确认事实，降低重复搜索成本。

Memory 没带来收益的情况：

- 原始证据不可得或网页抓取不稳定。Task 10 的地理等距点需要坐标和几何计算，`lightweight_memory` 和 `dynamic_cheatsheet` 都花了大量 token 但没有输出有效答案。
- 早期错误假设被 memory 放大。Task 17 中 `lightweight_memory`/`agent_kb` 把“王、陈并列”作为候选结论保留，后续没有充分修正；`dynamic_cheatsheet` 反而检索/总结到“王”。
- 评测抽取也会误伤。Task 19 中 `agent_kb` 的最终回答文本包含“5篇论文”，但 `extracted_answer` 为空，score=0。这是 judge/extraction 稳定性问题，不完全是 agent 解题失败。
- 长 memory 可能带来副作用。`dynamic_cheatsheet` 日志中出现 DeepSeek `Content Exists Risk` 400，说明过长或敏感的上下文注入会触发模型侧拒绝/重试。

### 系统间不一致的错误 case 逐题分析

本次 20 题中，真正有分析价值的是三套 memory system 结果不一致的 6 个 case：Task 5、10、16、17、19、20。它们能显示 memory 架构差异，而不只是最终分数差异。对应的逐题 review 文件已导出到 `xbench_output/case_reviews/`。

| Task | `lightweight_memory` | `agent_kb` | `dynamic_cheatsheet` | 主要问题 |
| ---: | --- | --- | --- | --- |
| 5 | 正确：12 | 正确：12 | 错误：8 | cheatsheet 把有条件的羽毛产量压缩成粗略经验 |
| 10 | 错误：空 | 正确：6.87km | 错误：空 | 地理外心/距离计算依赖工具链，不是普通事实记忆 |
| 16 | 正确：王建华 | 错误：刘旸 | 正确：王建华 | `agent_kb` 对节目人物细节的后续校验不足 |
| 17 | 错误：王和陈并列 | 错误：王和陈并列 | 正确：王 | 不完整名单被 memory 固化，漏掉王瑶琪 |
| 19 | 正确：5 | 判错：抽取为空 | 正确：5 | `agent_kb` 文本正确但 judge/extraction 失败 |
| 20 | 错误：14 | 正确：10 | 正确：10 | `lightweight_memory` 记住了错误年份排名 |

Task 5 是“过度压缩 memory 误导计算”的典型。三套系统都识别出《肖申克的救赎》台词后半句对应“羽毛”，也都识别出 GB/T 11881-2006 对应羽毛球，并知道每个羽毛球需要 16 根羽毛。差异出现在每只动物可提供多少根合格羽毛。`lightweight_memory` 和 `agent_kb` 保留了“合格刀翎、左右翅限制、每只约 7+7 根、160/14 向上取整”等约束，因此得到 12。`dynamic_cheatsheet` 则使用“一只鹅约 20 根可用羽毛”的粗略经验，直接得到 8。框架上看，`dynamic_cheatsheet` 会把相似轨迹压成 200 字以内的 cheatsheet，效率高，但容易丢失适用条件；这个 case 说明 semantic/procedural memory 必须保存证据边界，否则 shortcut 会变成错误规则。

Task 10 是“memory 无法替代工具计算”的典型。题目要求找到北京三个地点的等距点，并计算到三点的距离。`agent_kb` 得到约 6.87km，说明它可能通过历史经验给出了更明确的检索和几何计算路径。`lightweight_memory` 和 `dynamic_cheatsheet` 都没有输出有效答案，原因不是缺少可记忆事实，而是缺少稳定的坐标获取与外心计算链。这个 case 说明：当任务关键瓶颈是地图坐标、几何计算、数值工具调用时，memory 只能辅助组织步骤，不能替代工具能力。

Task 16 是“细粒度实体/视频证据追踪”的 case。题目需要依次定位最佳喜剧小队、第一轮作品、最后出场演员。`lightweight_memory` 和 `dynamic_cheatsheet` 都答出王建华，说明短期事实记录或 checklist 对综艺/人物定位有效。`agent_kb` 找到了“四士同堂”和《八十一难》，但最后把演员错判为刘旸。框架上看，`agent_kb` 主要在 BEGIN 阶段注入成功经验，后续不像 `lightweight_memory` 那样持续更新 task-level working memory；如果初始经验没有强制“逐段验证出场顺序”，就容易在最后人物细节上用成员常识代替证据。

Task 17 是“错误集合被 memory 固化”的 case。题目要统计清华建校到湖畔大学成立期间，中央财经大学历任校长中出现最多的姓氏。`lightweight_memory` 和 `agent_kb` 都得到“王和陈并列”，因为它们统计到王 3 次、陈 3 次，但漏掉王瑶琪。`dynamic_cheatsheet` 找到王瑶琪后得到“王”4 次。这个 case 说明集合统计任务中，memory 最危险的不是忘记，而是过早保存一个不完整列表；后续推理会围绕这个列表做加法，形成错误确认。更好的 memory 应保存“列表是否已覆盖完整来源”“是否还有缺失项”这样的验证状态。

Task 19 表面上是 `agent_kb` 错，实际更接近 evaluation harness 问题。`agent_kb` 的最终回答写明“韩晓光教授在 CVPR 2025 共发表了 5 篇论文”，与标准答案一致，但 `extracted_answer` 为空，因此 score=0。这个 case 说明分析 memory 效果时不能只看分数，要核对 `agent_result`、`extracted_answer` 和 judge 解释。对于数字型题目，评测链路应加入规则抽取兜底或独立 judge 二次确认。

Task 20 是“年份/排名类 semantic memory 精度不足”的 case。正确链条是香港中文大学 2025 QS 排名第 36、2020 QS 排名第 46，因此提高 10 位。`agent_kb` 和 `dynamic_cheatsheet` 保留了正确的年份-排名对应关系。`lightweight_memory` 使用了错误的 2025 排名第 32，于是算出 14。框架上看，`lightweight_memory` 的 task-level memory 能稳定保存中间事实，但如果早期检索到的事实错了，它也会稳定复用错误事实。年份、排名、版本号这类 semantic memory 必须带来源、日期和口径校验。

这些不一致 case 的总体结论是：memory 带来收益时，通常是它保存了可验证中间事实、计算模板和工具策略；memory 失败时，通常是它保存了不完整列表、错误数值或丢失约束的压缩经验。`lightweight_memory` 更适合任务内多步事实累积，但会放大早期错误事实；`agent_kb` 更适合复用成功轨迹和搜索策略，但对任务中后期的细粒度证据更新较弱；`dynamic_cheatsheet` 更轻、更快，但在高约束计算题上容易把经验压缩成误导性口诀。

### 追加 21-30 题的错误信息

为获得更多失败样本，我又追加运行了 xBench Task 21-30。该批次使用同样的 Flash-Searcher、DeepSeek-V4-Flash、`max_steps=12` 设置，输出前缀为 `extra_*`；没有清空 provider storage，因此它反映的是三套 memory 在完成前 20 题后的继续使用状态。该批次用于补充错误分析，不替代前 20 题的正式主结果。汇总文件为 `xbench_output/formal_summary/extra_results_summary.md`，逐题 review 文件在 `xbench_output/case_reviews_extra/`。

追加批次结果如下：

| Memory system | 正确数 | Accuracy | 错误/失败 task |
| --- | ---: | ---: | --- |
| `lightweight_memory` | 5/10 | 50.0% | 21、26、28、29、30 |
| `agent_kb` | 5/10 | 50.0% | 22、26、28、29、30 |
| `dynamic_cheatsheet` | 5/10 | 50.0% | 25、26、28、29、30 |

逐题结果：

| Task | `lightweight_memory` | `agent_kb` | `dynamic_cheatsheet` | 新增错误类型 |
| ---: | --- | --- | --- | --- |
| 21 | 判错但文本含 82 | 正确：82 | 正确：82 | judge/extraction 漏抽数字 |
| 22 | 正确：2 | 错误：None | 正确：2 | `agent_kb` 运行时空响应错误 |
| 25 | 正确：东直门涮肉 | 正确：东直门涮肉 | 错误：None | cheatsheet/模型空响应错误 |
| 26 | 错误：Content Exists Risk | 错误：Content Exists Risk | 错误：Content Exists Risk | 网页/上下文触发模型风控 |
| 28 | 错误：哈维·阿隆索 | error | 错误：哈维·阿隆索 | 题意误读：常规时间进球 vs 点球大战最后进球 |
| 29 | 错误：5月2日 | error | 错误：5月2日 | 航班日期边界和时刻表检索错误 |
| 30 | 空/失败 | error | 最终输出错误 | 高约束实体匹配和长链搜索失败 |

Task 21 显示的是评测抽取问题。`lightweight_memory` 的最终回答已经写出平均值为 81.9、四舍五入为 82，但 `extracted_answer` 为空，导致 score=0。这个 case 与 Task 19 类似，说明数字答案即使出现在文本中，judge 仍可能漏抽；对这类题应增加确定性数字抽取兜底。

Task 22 和 Task 25 是运行时空响应错误。Task 22 中 `agent_kb` 返回 `None` 并出现 `'NoneType' object has no attribute 'reasoning_content'`；Task 25 中 `dynamic_cheatsheet` 也出现同类空响应，最终无答案。它们不直接说明 memory 逻辑错误，而说明当前 harness 对模型空响应/异常响应的容错不足。更好的做法是把空响应作为可重试状态，而不是直接进入评分。

Task 26 三套系统全部失败，错误均为 DeepSeek `Content Exists Risk`。这不是某一种 memory system 的单点问题，而是网页内容、检索上下文或 memory 注入内容触发了模型侧风控。它提示 memory/harness 需要做输入净化、上下文裁剪和失败 fallback：例如去除不必要网页原文、只保留结构化事实、遇到 400 风控时改用更短上下文重试。

Task 28 三套系统没有得到正确答案。`lightweight_memory` 和 `dynamic_cheatsheet` 都回答哈维·阿隆索，但标准答案是弗拉基米尔·斯米切尔。错误来自题意拆解：agent 把“获胜球队最后一位进球”理解为 2005 欧冠决赛常规时间/加时中帮助利物浦追平的最后进球者，于是选了阿隆索；标准答案对应的是点球大战中利物浦最后一位罚进点球的球员斯米切尔。这里 memory 没有保存“体育题要区分常规时间进球、加时、点球大战、最终获胜条件”的 procedural checklist。

Task 29 三套系统都没有得到正确日期。`lightweight_memory` 和 `dynamic_cheatsheet` 都回答 5月2日凌晨，而标准答案是 5月1日凌晨 1:40。这个 case 的关键是日期范围、时区、出发地机场、直飞条件和航班时刻表版本。Memory 可以帮助保留检索线索，但如果没有可靠时刻表工具或严格的日期边界校验，容易把“最早查到的直飞航班”当成“五一期间最早直飞航班”。

Task 30 三套系统均失败或未输出有效答案。该题要求找出两位诺贝尔奖得主：一位物理奖、一位化学奖，年龄差 6799 天、first name 相同、曾就职于同一所美国西岸大学。标准答案是 David Baker 和 David Jeffrey Wineland。这个任务需要高约束实体匹配，适合用结构化检索/约束求解，而不是纯网页搜索加自然语言 memory。现有 memory 没有把“候选实体表、出生日期差、共同任职机构、奖项类别”组织成可验证表格，因此长链搜索容易耗尽步骤或失败。

追加批次强化了前 20 题的结论：memory 的主要收益来自保存可验证事实和流程，但失败往往来自三类非记忆因素：第一，judge/extraction 漏抽正确答案；第二，模型空响应、风控、网络错误等 harness 稳定性问题；第三，需要专用工具或结构化约束求解的任务，memory 只能辅助，不能替代工具。

### 追加 31-50 题的错误信息

按照“每个 memory system 再多跑 20 条”的要求，我继续运行了 xBench Task 31-50，输出前缀为 `extra31_50_*`。这一批三套系统均完整覆盖 31-50，`status` 全部为 `success`。汇总文件为 `xbench_output/formal_summary/extra31_50_results_summary.md`，逐题 review 文件在 `xbench_output/case_reviews_extra31_50/`。

追加 31-50 批次结果如下：

| Memory system | 正确数 | Accuracy | 错误 task |
| --- | ---: | ---: | --- |
| `lightweight_memory` | 11/20 | 55.0% | 31、32、33、35、41、43、45、48、49 |
| `agent_kb` | 11/20 | 55.0% | 31、33、35、39、41、42、44、45、47 |
| `dynamic_cheatsheet` | 14/20 | 70.0% | 32、33、35、41、45、48 |

逐题差异和新增错误类型：

| Task | `lightweight_memory` | `agent_kb` | `dynamic_cheatsheet` | 新增错误类型 |
| ---: | --- | --- | --- | --- |
| 31 | 错误：空响应 | 错误：多答两个作者 | 正确：何恺明 | 问题问“第一作者”，不能列多个候选 |
| 32 | 错误：空响应 | 正确：7年 | 错误：11年 | 年份区间/起止口径错误 |
| 33 | 错误：5.95-6.05m | 错误：无 | 错误：无 | 近似数值与评测答案口径不一致 |
| 35 | 错误：39 | 错误：39 | 错误：55 | 复杂统计表/名单边界错误 |
| 39 | 正确 | 错误：空响应 | 正确 | `agent_kb` 空响应导致丢分 |
| 41 | 错误：空/无 | 错误：无 | 错误：无 | 经济指标同比/环比口径与数值提取失败 |
| 42 | 正确 | 错误：36集 | 正确 | 同 IP 电视剧版本/正片集数口径错误 |
| 43 | 错误：陈宝香 | 正确：王神医 | 正确 | 小说章节中第 N 位人物定位错误 |
| 44 | 正确 | 错误：空响应 | 正确 | `agent_kb` 空响应导致丢分 |
| 45 | 错误：品牌/地点 | 错误：空 | 错误：地点 | 品牌 logo 谜题链条断裂 |
| 47 | 正确 | 错误：空 | 正确 | `agent_kb` 未输出有效答案 |
| 48 | 错误：可以 | 正确：不能 | 错误：可以直达 | 轨交换乘/站台直达语义判断错误 |
| 49 | 错误：工具输出异常 | 正确：杜可风 | 正确 | harness 工具输出类型未处理 |

Task 31 是“单答案约束”问题。题目问人工智能领域引用量超过 25 万次 paper 的第一作者，标准答案是何恺明。`agent_kb` 输出“Kaiming He 和 Diederik P. Kingma”，把多个高引论文作者并列给出，违背“the paper 的第一作者”这一单答案约束；`lightweight_memory` 则因空响应漏答。`dynamic_cheatsheet` 在这题答对，说明短 checklist 能帮助 agent 最后收敛到单一候选。

Task 32 暴露了年份区间口径问题。`agent_kb` 得到 7 年，`dynamic_cheatsheet` 得到 11 年，说明它把起止年份或事件边界扩宽了。对于时间跨度题，memory 需要保存的不只是年份数字，还要保存“从哪一年算到哪一年、是否含首尾年份、事件发生日是否已经跨过”等口径。

Task 33 是数值近似和评测口径问题。`lightweight_memory` 输出 5.95-6.05m，标准答案是 5.9m；另外两套未给出有效答案。这类题如果答案要求一个标量，范围表达可能被判错。Memory 应帮助记录“最终答案格式只输出一个数值”，而不是只记录估算范围。

Task 35 是复杂名单/表格统计失败。三套都错，但错法不同：`lightweight_memory` 和 `agent_kb` 得到 39，`dynamic_cheatsheet` 得到 55，标准答案是 45。这说明任务依赖完整表格抽取和筛选边界，普通文本 memory 不足以保证集合完整性。更合适的是 structured table memory 或中间 CSV/表格工具。

Task 41 是经济指标口径失败。标准答案为增长 1.7%，三套都未抽出有效答案。此类题通常涉及同比、环比、百分点、百分比、月份/季度口径，memory 如果只保存文本片段而不保存指标公式，很容易无法计算或抽取。

Task 42 是影视/综艺 IP 版本口径问题。`agent_kb` 回答 36 集，而标准是 32 集；另外两套正确。错误原因类似 Task 16：实体找到了，但版本、正片集数、精华剪辑版排除等约束没有保留。说明 episodic/semantic memory 需要把“排除条件”作为一等信息，而不只是记住实体名。

Task 43 是小说章节人物定位错误。`lightweight_memory` 回答陈宝香，标准是王神医；另外两套正确。该题要求定位第九章中出现的第三位人物，属于顺序敏感任务。`lightweight_memory` 的短期记忆能保存人物，但如果没有保存“出现顺序”和“计数规则”，就会把候选人物混淆。

Task 45 是品牌 logo 谜题链条断裂。三套都错：有的回答品牌/公司地点，有的无答案，标准是天空之城。它需要从两个品牌的成立时间、logo 字母对称关系、再到另一个引用/联想对象层层跳转。此类谜题要求 memory 保存多条假设及其验证状态，而不是每步只保留当前最像答案的实体。

Task 48 是交通可达性语义判断错误。`agent_kb` 正确回答不能，另外两套答可以。它要求理解“从长亭站能否直达花桥地下站台”，不是简单判断线路网络是否连通；还要区分换乘、出站、同站台/地下站台等条件。这里 memory 的收益来自 tool-use/procedural checklist：路线题应明确“直达”的定义。

Task 49 是 harness 工具输出类型问题。`lightweight_memory` 出现 `Unsupported step output: <class 'list'>`，而另外两套答对杜可风。这不是 memory 推理错误，而是 Flash-Searcher 对某个工具返回 list 的处理不健壮，导致流程中断或最终抽取失败。

31-50 批次进一步补充了两个观察。第一，`dynamic_cheatsheet` 在 31-50 上达到 14/20，是三套中最好，说明当任务依赖短 checklist、实体跳转和常见网页搜索策略时，轻量 cheatsheet 可能比更重的 memory 更有效。第二，三套系统共同失败的 Task 33、35、41、45 说明，memory 不能替代表格解析、数值计算、指标公式和结构化约束求解；这些任务更需要 tool-use memory 与专用工具配合。

## 5. 哪种 memory 形态更有效

Episodic memory 记录的是“某一次任务中 agent 具体经历了什么”：原问题、搜索路径、访问过的 URL、尝试过的错误方向、最后答案和成功/失败原因。它的优点是能复用完整轨迹；缺点是冗长、昂贵，并且容易把只适用于旧任务的细节带到新任务里，同时不一定相似的任务能够迁移。

适合的任务例子：如果任务要求查找某个综艺节目中“最后出场演员”的姓名，历史上已经做过类似“先定位节目/队伍，再定位某一轮作品，最后核对出场顺序”的任务，episodic memory 可以直接复用过去的检索路径，例如先找节目百科页，再找视频片段或分集信息，最后用多源交叉验证人物顺序。它适合这种原因是：任务表面实体不同，但信息获取路径高度相似，过去轨迹中的 URL 类型、关键词组合和失败路线都有参考价值。

不适合的任务：不适合简单事实问答、强时效问题，或只是表面相似但约束不同的题。例如“某大学 2025 QS 排名比 2020 提高多少名”看起来也可以复用旧的排名检索轨迹，但如果 episodic memory 直接保留旧题中的年份、排名或网站快照，就可能把过时事实注入当前任务。对于这类任务，episodic memory 最好只作为原始经验库，用来生成更短的 semantic/procedural 记忆，而不是完整塞进上下文。

Semantic memory 记录的是“稳定知识和事实”：实体别名、年份-排名对应关系、标准条款、术语定义、可靠来源规则等。它不保留完整过程，而是把多次经验压缩成可复用知识片段。优点是短、密度高、泛化好；缺点是如果抽象时丢掉来源、日期和适用边界，就会形成错误但看似可靠的常识。

适合的任务例子：Task 20 这类排名对比题适合 semantic memory。有效记忆可以写成：“香港中文大学 QS 2025 排名为 36，QS 2020 排名为 46；排名提升按旧排名减新排名计算，因此提升 10 位；来源应优先使用 QS 官方或学校新闻稿。”这里 memory 的价值不是保存一整条搜索轨迹，而是保存已经核对过的关键事实和计算口径。

不适合的任务：不适合需要实时网页状态、强上下文证据或复杂操作链的任务。例如航班时刻、比赛最后进球者、网页 revision 记录等，事实会随日期、版本或定义变化。若 semantic memory 只写“某航线最早直飞为某日某时”，却没有日期范围、时区、查询来源和航班号，下一次任务很可能误用。semantic memory 必须带上证据边界：来源、时间、版本、适用条件。

Procedural memory 记录的是“怎么做”：公式、计算模板、搜索步骤、验证 checklist、失败后 fallback 流程。它不直接记答案，而是记可执行的方法。优点是能稳定改善 agent 行为，减少重复试错；缺点是如果流程来自错误假设，会稳定地产生错误答案。

适合的任务例子：Task 2、Task 5 这类“先找规则再计算”的题最适合 procedural memory。比如羽毛球数量题，好的 procedural memory 不是只记“鹅能提供约 20 根羽毛”，而是记流程：“先确认标准规定每个球需要多少根羽毛；再确认每只动物可用羽毛数量是否有限制；如果左右翼或刀翎有限制，要按合格羽毛数计算；最后向上取整。”它适合的原因是：题目答案依赖约束识别和计算步骤，记住流程比记住某个旧答案更可靠。

不适合的任务：不适合主要依赖开放事实发现、实体集合不完整或定义模糊的任务。例如 Task 17 历任校长姓氏统计，如果 procedural memory 过早固化成“抓取一个名单后直接按姓统计”，但没有加入“检查名单是否覆盖完整任期、是否存在遗漏人物、是否有同名/别名”的步骤，就会把不完整列表稳定地算错。procedural memory 应该包含验证条件，而不只是操作步骤。

Tool-use memory 记录的是“什么工具在什么情况下有效，以及如何调用”：搜索 query 模板、站点限定、PDF/表格解析方式、地图坐标获取、MediaWiki revisions、视频平台检索、失败重试策略等。它可以看作 procedural memory 的工具专项版本。优点是对长链 agent 很直接，因为很多失败不是推理失败，而是工具选错、参数错或没有 fallback；缺点是强依赖当前工具、API、网络和模型环境。

适合的任务例子：Task 10 这类地理等距点计算题适合 tool-use memory。有效记忆应提示：“先用地图/地理编码工具获取三个地点坐标；不要仅凭自然语言搜索结果估算；用几何或脚本计算外心/等距点；再反查该点到三地的距离并统一单位。”它适合的原因是：关键瓶颈不是记住事实，而是调用正确工具链。如果没有坐标获取和数值计算工具，memory 再多也不能替代计算。

不适合的任务：不适合纯文本概念解释、简单常识问答，或工具环境频繁变化的任务。例如某个 provider 从 Jina 切到 crawl4ai 后，旧的“Jina 摘要字段在哪里、失败码如何处理”的记忆可能不再适用。tool-use memory 必须记录工具版本、失败条件和 fallback，而不能只记录“用某工具查”。

本次 20 条 xBench 中，最有效的是 procedural + semantic 的混合形态：semantic memory 保留关键事实、实体、年份和来源边界；procedural memory 保留计算模板、检索步骤和验证 checklist。episodic memory 更适合作为原始轨迹材料，供后续压缩和反思，而不是直接大量注入上下文。tool-use memory 在地理、视频、PDF、表格、网页历史版本等任务中尤其重要，因为这些题的瓶颈常常不是“知道什么”，而是“该用什么工具、按什么参数查证”。单纯 episodic 容易太重；单纯 cheatsheet 在常识化压缩后容易误导；更可靠的方向是把轨迹压缩成带证据边界的 semantic memory、带校验条件的 procedural memory，以及可 fallback 的 tool-use memory。

## 6. Meta-evolution 与 harness 自进化的关系

Memory meta-evolution 是在固定或近似固定 harness 内，进化 memory 的 `Encode/Store/Retrieve/Manage` 架构。它回答的是：agent 应该怎样从轨迹中学习、存什么、何时取出、如何遗忘。

Harness 自进化则是进化 agent 框架本身，例如 planner、tool router、并行调度、反思器、子 agent 分工、工具集和控制流。它回答的是：agent 应该怎样行动、调用什么工具、如何组织推理。

二者关系：

- memory evolution 依赖 harness 产生轨迹和反馈；harness 越稳定，memory 的 fitness signal 越干净。
- harness evolution 会改变 action space 和错误分布，从而改变最优 memory 架构。
- 更理想的方向是协同进化，但必须分离验证集，否则容易把 harness 改动和 memory 改动混在一起，无法判断收益来自哪里。

## 7. 对 MemEvolve 的不足与改进建议

代码层面：

- `lightweight_memory` 默认 `enable_longterm_provision=False`，本次日志也显示长期记忆不参与检索。这会削弱跨任务收益，建议改成可学习/可门控的长期检索，而不是直接关闭。
- 评测报告原本只认 `judgement`，不认 xBench 的 `score`，会误报准确率；已在本地修复。
- Windows 下输出编码容易 mojibake，建议统一设置 `PYTHONIOENCODING=utf-8`，并在 README 的 Windows 部分写明。
- Jina 402 后没有自动 fallback 到 crawl4ai，建议在工具层提供 provider fallback。
- Memory provider 每题重新初始化，虽然能通过磁盘持久化跨任务，但短期状态和长程状态边界不清晰；建议显式区分 run-level memory、task-level working memory、provider checkpoint。

方法层面：

- Pareto 目标应加入“retrieval precision / harmful memory rate”，不仅看 success、cost、latency。
- 需要更严格的 train/validation split。meta-evolution 不应在同一批任务上反复优化后直接报告，否则容易过拟合搜索习惯。
- 诊断阶段应区分“agent 原本不会解”和“memory 误导 agent”，否则架构进化可能错误归因。
- Tool-use memory 应成为一等公民。深度搜索任务中，复用工具调用策略通常比复用自然语言经验更可靠。

## 8. 运行过程复盘与改进

结果完整性检查：

- 三个正式结果文件均包含 20 条记录，task id 覆盖 1-20，无缺失、无重复。
- 三个 provider 的 `status` 均为 `success`，聚合分数来自 xBench 的 `score` 字段。
- 已新增 `xbench_output/formal_summary/results_summary.md`，固定输出完整性、总分、逐题 score 和异常标记，且不写入解密后的 benchmark 题面与标准答案。
- 为补充错误分析，追加运行 Task 21-30，结果写入 `xbench_output/formal_summary/extra_results_summary.md`；该批次沿用前 20 题后的 memory storage，且 `agent_kb` 的 Task 28-30 为运行 error，因此只作为补充失败样本。
- 按后续要求继续追加运行 Task 31-50，结果写入 `xbench_output/formal_summary/extra31_50_results_summary.md`；该批次三套系统均完整覆盖 20 题，`status` 全部为 `success`。

运行过程中的不足：

- 本次只跑了 xBench-DeepSearch，没有跑 GAIA；原因是本地已准备好 xBench 数据，GAIA 数据未在仓库中直接可用。
- `max_steps=12` 低于 Flash-Searcher README 中较常见的长链设置。这保证三套 memory system 公平比较并控制成本，但可能压低 Task 10 这类困难题的表现。
- Judge 也使用 `deepseek-v4-flash`，Task 19 显示 final answer 中有正确数字但 `extracted_answer` 为空，说明评测抽取存在误伤。
- Web 抓取不完全稳定：Jina 因 quota 返回 402 后切到 `crawl4ai`；`agent_kb` 曾在 Task 20 因网页连接问题中断，需要续跑补齐。
- `dynamic_cheatsheet` 日志中出现过 DeepSeek `Content Exists Risk` 400，提示长 memory/网页内容注入可能触发模型侧风控。
- 初版运行脚本默认清空输出和 provider storage，不利于断点续跑；中断后需要人工拼接结果。
- `lightweight_memory` 配置中长期记忆 provision 未启用，本次更多体现 task-level working memory，而不是完整跨任务长期记忆收益。
- 追加批次暴露出新的 harness 问题：`NoneType.reasoning_content`、`NoneType.content`、DeepSeek `Content Exists Risk`、网络 timeout 和 Windows/GBK 编码读取问题。这些会把“系统运行失败”混入“memory 解题失败”，需要在分析时单独标注。

已做的改进：

- 修复 `eval_utils.py`：统一报告现在兼容 xBench 的 `score` 字段，不再把 xBench 准确率误报为 0。
- 改进 `scripts/run_formal_xbench_eval.py`：默认支持 resume，只跑缺失 task；显式 `--fresh` 才清空旧输出和 memory storage；跑后校验缺失、重复、越界 task id 和非 success 状态。
- 新增 `scripts/summarize_xbench_memory_results.py`：一键生成正式结果摘要，并标记 `score=0` 但答案抽取为空的可疑 case。
- 改进 `scripts/export_case_review.py`：支持 `--output-prefix`，可分别导出正式批次和追加批次的逐题 review 文件。
- 复验命令已通过：`py_compile` 成功；runner 在已有结果上识别三套系统均已完成并跳过重跑；正式批次和追加批次汇总文件均已生成。

后续若继续提高严谨性，建议增加两项：其一，换用更强或更稳定的独立 judge 做二次评分；其二，用 `--fresh` 重跑完整 20 题并保留原始日志，避免“续跑拼接”给实验过程留下疑点。

## 9. 结论

在本次 Flash-Searcher + DeepSeek-V4-Flash + xBench 20 题实验中，三种 memory 都能跑通。`dynamic_cheatsheet` 以 18/20 最好，`lightweight_memory` 和 `agent_kb` 均为 17/20。`lightweight_memory` 展示了 MemEvolve 自动进化系统的工作记忆能力，但成本最高；`dynamic_cheatsheet` 在这批任务上更快、更准，但也出现错误 procedural rule 和内容风控风险。整体看，memory 的收益主要来自保存可验证中间事实、工具策略和计算模板；在网页证据不稳定、空间计算、实体集合统计边界模糊时，memory 不能替代可靠检索和验证，甚至会放大错误假设。
