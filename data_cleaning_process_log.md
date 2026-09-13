# AI Personification Corpus — 数据清洗流程存档

本文档记录全年语料处理中，在 `run_conservative_and_stitched_batch()` 跑完之后、
进入动词分级（Level 2/3标注）之前，所做的完整数据清洗步骤，按实际执行顺序排列。

适用对象：`df_conservative`（保守版全量数据）、`review_set`（缝合候选待审核数据）

---

## 前提：两份原始数据已经跑出

```python
df_conservative, review_set = run_conservative_and_stitched_batch()
```

此时两份数据都还是"未清洗"状态，包含之后要处理的各类问题。

---

## Step A：句子去重

### 目的
排除完全相同的句子文本在不同文章间重复出现的情况（多为标题、"Read more"推荐链接等
模板化文字被机械复制导致），避免重复计数扭曲Level分布统计。

注意：这里按 `sentence_text` 纯文本去重（不看是否来自同一篇文章），
因为目标是消除"同一段文字被复制到多处"，而不仅仅是同一文章内部的偶然重复。

同一句话内部AI提及多次的情况（如 "AI is trained using AI-generated data"）
不受此步影响，属于同一句话的不同参与者角色实例，予以保留。

### 代码

```python
conservative_included_raw = df_conservative[df_conservative["exclusion_reason"].isna()].copy()
conservative_included_dedup = conservative_included_raw.drop_duplicates(
    subset=["sentence_text"], keep="first"
)

review_set_dedup_raw = review_set[
    (review_set["exclusion_reason"].isna()) & (review_set["was_stitched"] == True)
].copy()
review_set_dedup_raw = review_set_dedup_raw.drop_duplicates(
    subset=["sentence_text"], keep="first"
)

print(f"保守版：去重前 {len(conservative_included_raw)} -> 去重后 {len(conservative_included_dedup)}")
print(f"缝合候选：去重前 {len(review_set_dedup_raw)} -> 去重后 {len(review_set_dedup_raw)}")
```

---

## Step B：动词提取有效性校验（invalid_verb_extraction）

### 目的
排除因产品型号（如包含"+"符号、数字后缀的芯片名 "Ryzen AI Max+ 395"）导致
依存句法解析错乱，从而提取出不像正常英语单词的"动词"（如"max+"、"395"、"'s"）
的记录。这类记录若不处理，会以无意义的伪动词混入动词分级表。

### 判断标准
提取出的 governing_verb 必须：纯字母组成、长度在2-20个字符之间。
不满足则视为提取失败，标记排除，不参与后续Level分级。

### 代码

```python
def valid_verb_lemma(lemma):
    return bool(lemma) and str(lemma).isalpha() and 1 < len(str(lemma)) <= 20

def apply_invalid_verb_check(df):
    mask_invalid = df["governing_verb"].notna() & ~df["governing_verb"].apply(valid_verb_lemma)
    print(f"  无效动词提取排除: {mask_invalid.sum()} 条")
    df.loc[mask_invalid, "exclusion_reason"] = "invalid_verb_extraction"
    df.loc[mask_invalid, "governing_verb"] = None
    return df

conservative_included_dedup = apply_invalid_verb_check(conservative_included_dedup)
review_set_dedup_raw = apply_invalid_verb_check(review_set_dedup_raw)
```

---

## Step C：关系动词排除（relational_process）

### 目的
按理论框架 Appendix C 的规定，Relational Process（关系过程，如 be / become / seem）
不赋予AI主动参与者角色（AI在此类结构中是Carrier/Attribute，而非Actor/Senser/Sayer），
理应从一开始就不进入Level 2/3的候选池。此前脚本仅依据句法位置（dep_==nsubj）判断，
未检查动词本身的过程类型，导致这类动词被错误地送入了待分级队列
（其中"be"以236次的高频占据了动词表首位，是目前发现的规模最大的一处系统性偏差）。

### 判断标准：分两组处理

**第一组 ALWAYS_RELATIONAL_VERBS —— 无条件排除**
在科技新闻语境下，这几个词几乎总是作关系动词使用：
be, become, remain, constitute, represent, mean, signify, equal

**第二组 AMBIGUOUS_PERCEPTION_VERBS —— 按句法结构动态判断，不能无脑拉黑**
look, sound, feel, taste, smell, seem, appear

这组词是典型的"两用动词"：
- 接形容词补语(acomp)时是关系动词用法（"AI **looks** smart" —— smart是acomp，排除）
- 无形容词补语、作实义动词使用时，是真实的Material/Mental过程，不该排除
  （"AI is **looking for** patterns" —— 认知/调查类动作；"AI **feels** the vibration" —— 感知动作）

判断方法：重新对该句子跑一次spaCy解析，检查该动词token的子节点里
是否存在 dep_ == "acomp"；存在则判定为关系动词用法，排除；不存在则保留，
正常进入Level 2/3候选池。

### 尚待人工核查的灰色地带
"have"（27次）未列入排除清单——因为"AI has learned patterns"这类用法可能偏向
Material，而"AI has a feature"这类又偏向Relational，两种语义混杂，建议单独抽出
"have"对应的全部句子人工核查后再决定是否补充排除。

### 代码

```python
ALWAYS_RELATIONAL_VERBS = {
    "be", "become", "remain", "constitute", "represent",
    "mean", "signify", "equal",
}
AMBIGUOUS_PERCEPTION_VERBS = {"look", "sound", "feel", "taste", "smell", "seem", "appear"}

def apply_relational_verb_check(df):
    # 第一组：无条件排除
    mask_always = df["governing_verb"].notna() & df["governing_verb"].isin(ALWAYS_RELATIONAL_VERBS)
    print(f"  无条件关系动词排除: {mask_always.sum()} 条")
    df.loc[mask_always, "exclusion_reason"] = "relational_process"
    df.loc[mask_always, "governing_verb"] = None
    return df

conservative_included_dedup = apply_relational_verb_check(conservative_included_dedup)
review_set_dedup_raw = apply_relational_verb_check(review_set_dedup_raw)


# 第二组：感官/歧义动词，需要重新解析句子，检查是否带acomp形容词补语
def recheck_relational(sentence_text, verb_lemma, nlp):
    doc = nlp(sentence_text)
    for tok in doc:
        if tok.lemma_.lower() == verb_lemma:
            has_acomp = any(child.dep_ == "acomp" for child in tok.children)
            return has_acomp  # True = 关系动词用法，该排除；False = 实义动词，保留
    return None  # 没找到匹配token，保险起见标记为待人工看

def apply_ambiguous_perception_check(df, nlp):
    mask_ambiguous = df["governing_verb"].isin(AMBIGUOUS_PERCEPTION_VERBS)
    print(f"  涉及感官/seem-appear类动词的记录数: {mask_ambiguous.sum()}")
    affected_rows = df[mask_ambiguous]

    for idx, row in affected_rows.iterrows():
        is_relational = recheck_relational(row["sentence_text"], row["governing_verb"], nlp)
        if is_relational is True:
            df.loc[idx, "exclusion_reason"] = "relational_process"
            df.loc[idx, "governing_verb"] = None
        elif is_relational is None:
            df.loc[idx, "exclusion_reason"] = "manual_review_ambiguous_verb"
        # is_relational is False -> 保留，不做任何改动
    return df

conservative_included_dedup = apply_ambiguous_perception_check(conservative_included_dedup, nlp)
review_set_dedup_raw = apply_ambiguous_perception_check(review_set_dedup_raw, nlp)
```

---

## Step D：清洗后重新收口（去掉Step B/C新产生的排除记录，重新过滤）

Step B、C 会把部分原本"exclusion_reason为空"的记录重新标记为有排除原因，
所以清洗完成后需要重新筛一次"真正纳入统计"的子集。

```python
conservative_final = conservative_included_dedup[
    conservative_included_dedup["exclusion_reason"].isna()
].copy()

review_set_final = review_set_dedup_raw[
    review_set_dedup_raw["exclusion_reason"].isna()
].copy()
review_set_final["manual_keep"] = ""  # 留空，供人工审核填 Y/N

print(f"保守版最终纳入统计数: {len(conservative_final)}")
print(f"缝合候选最终待人工审核数: {len(review_set_final)}")
```

---

## Step E：导出清洗后的文件

```python
conservative_final.to_excel(OUT_DIR / "step2_conservative_included_dedup.xlsx", index=False)
review_set_final.to_excel(OUT_DIR / "manual_review_stitched_all_year.xlsx", index=False)

verbs_all = (
    conservative_final[conservative_final["governing_verb"].notna()]["governing_verb"]
    .value_counts()
    .reset_index()
)
verbs_all.columns = ["verb_lemma", "frequency"]
verbs_all.to_excel(OUT_DIR / "step3_unique_verbs_all_year.xlsx", index=False)

print(f"独立动词数: {len(verbs_all)}")
print("已导出：step2_conservative_included_dedup.xlsx / manual_review_stitched_all_year.xlsx / step3_unique_verbs_all_year.xlsx")
```

---

## 汇总：本轮清洗涉及的所有排除类别

| exclusion_reason | 说明 | 引入阶段 |
|---|---|---|
| modifier | AI作compound/amod修饰语（如"AI-powered"）| 原始pipeline |
| possessive | AI作属格（如"AI's decision"）| 原始pipeline |
| prepositional instrument | AI作介词宾语，工具类介词（如"using AI"）| 原始pipeline |
| subordinate clause | AI所在小句为从句/转述引语 | 原始pipeline |
| incomplete_fragment_start/end | 省略号导致片段首尾不完整 | 数据清洗迭代 |
| other_unclassified | 其余未归类句法结构（如同位语、并列结构）| 原始pipeline |
| invalid_verb_extraction | 因符号/型号导致依存解析错乱，提取出无效"动词" | 本轮清洗 Step B |
| relational_process | 关系动词（be/become/seem等，含acomp判定后的look/sound/feel等），按Appendix C排除 | 本轮清洗 Step C |
| manual_review_ambiguous_verb | 感官/歧义动词经重新解析后仍无法自动判断，需人工逐句核查 | 本轮清洗 Step C |

---

## 写入论文方法论时可参考的措辞

> Following initial syntactic classification, several additional data-cleaning steps were
> applied. Duplicate sentences (arising from repeated headlines or "read more" links across
> articles) were removed based on exact text matches. Records where verb extraction failed
> due to parser confusion caused by unconventional product naming (e.g., model numbers
> containing symbols) were identified via a validity check on the extracted lemma and
> excluded. Consistent with the exclusion of Relational Processes specified in the
> classification framework (Appendix C), instances where AI functioned as the subject of a
> copular or relational verb (e.g., be, become) were excluded from the Level 2/3 candidate
> pool, as these do not assign AI an active participant role (Actor, Senser, or Sayer) under
> Halliday and Matthiessen's (2014) transitivity system. For verbs with inherently ambiguous
> usage between relational and lexical functions (e.g., look, sound, feel, seem, appear),
> each instance was re-examined for the presence of an adjectival complement (acomp): where
> present (e.g., "AI looks smart"), the instance was treated as relational and excluded;
> where absent (e.g., "AI looks for patterns"), the verb was retained as a genuine
> Material/Mental process instance.

## 附注：与"attributive personification"（属性性拟人化）的边界区分

在数据核查过程中注意到，形如 "AI **is** clever" 这类结构，虽然按概念隐喻理论(CMT)
（Lakoff & Johnson, 1980）的定义，属于把人类特质(human trait)映射到AI这一非人类实体上，
理应算作一种拟人化现象；但在及物性(Transitivity)框架下，这类Relational Process结构
并未赋予AI任何主动参与者角色(Actor/Senser/Sayer)，因此不落入本研究Level 1-3的统计范围。

这提示本研究的及物性方法测量的是一种特定类型的拟人化——**"施事性拟人化"
(agentive personification)**：AI是否被赋予"做事/思考/言说"的主动能力；
而"AI is clever"代表的是**"属性性拟人化"(attributive personification)**：
AI是否被赋予人类的性格/品质标签。两者是不同的拟人化实现机制，本研究方法
天然只捕捉前者，这一点建议在Limitations或Discussion中明确说明，作为
方法论边界的自觉交代，而非疏漏：

> While the transitivity-based framework effectively captures agentive personification
> (AI occupying Actor/Senser/Sayer roles), it does not capture attributive personification
> realised through Relational Processes (e.g., "AI is clever"), where human traits are
> ascribed to AI without assigning it an active participant role. Future research could
> incorporate a complementary analysis of adjectival/attributive constructions to capture
> this distinct mechanism of personification.

---

## Step F：与导师沟通后的两项精细化调整（数据处理最后阶段）

### F1：全年缝合候选人工审核完成 + 与保守版合并

在完成"疑难动词"（have/exist/behave等）核查、修复"feature被误判为动词"这一系列
问题后（详见前述Step B/C），进行了全年缝合候选（fragment-repair candidate set）
的完整人工通顺性审核，共审核约690条，通过（manual_keep=Y）321条。审核过程中同步
记录了部分句子涉及的"have"动词的keep/exclude及Level判断，这批判断先于导师关于
轻动词的正式建议提出，后续在F3中被更精细的规则重新覆盖。

审核结果与保守版数据集（1540条，已完成relational/invalid verb清洗及"feature"
修正）合并、去重，得到合并后数据集：1557条。核查发现其中17条为此前已标记但未
真正移除的"feature"相关记录（`exclusion_reason`非空但仍残留在表中），清理后
得到干净的最终数据集：**1540条**。

**技术备注**：此前用 `.loc[mask, "level"] = None` 的方式"软删除"记录（只清空
level字段、不物理删除行）在统计计算时不会有误（因为统计代码用
`df[df["level"].notna()]`筛选），但在跨数据集`concat`合并时，这些"软删除"的
行会被一并带入合并结果，需要在合并后额外执行一次
`df[df["exclusion_reason"].isna()]`才能得到真正干净的数据集。后续所有导出
最终数据集前，应固定执行这一步清理，避免同类问题重复出现。

### F2：词频分组（导师建议）

按导师建议，以频次≥5次为界，将动词分为"高频组"（纳入正文逐一讨论）与"低频组"
（列入附录，不逐一展开）。这是呈现层面的调整，不改变任何Level判定结果。

进一步地，导师建议对高频组动词做**语义子分类**——考察哪些动词的拟人化"标记性"
更强（即较少用于非人类主语，如remember/know/try），哪些较弱（日常英语中本就
常见于非人类主语，如help/work/create）。此举对应`discussion_insights_log.md`
第4点提出的"拟人化强度分层"观察，将其从即兴讨论正式转化为分析步骤（详见该文档
新增第8点）。

### F3：轻动词结构精细化重新分类（导师建议）

在此前"所有轻动词结构统一归Level2"的粗略处理基础上，按导师建议进行精细化重分类，
识别两种句法模式：

**模式1（deverbal noun，名词化动词宾语）**：have/take/make/give/do + 一个由动词
名词化而来的宾语（如solution←solve, decision←decide, thought←think），
按该名词对应的原动词查询Level（已建立`DEVERBAL_NOUN_MAP`初步映射表，覆盖约
20个常见名词化名词，需要根据实际匹配情况持续补充）。

**模式2（abstract noun + infinitive，抽象名词+不定式）**：have + ability/
potential/capacity等抽象名词 + to不定式（如"has the ability **to identify**"），
真正应该追溯的动作动词是不定式部分（这里是identify），而非"ability"本身——
这是在核对F1阶段对"have"的人工判断时发现的遗漏模式：F1阶段将"has the ability
to identify"整体判为Level2，但若按追溯真实动作的原则，"identify"本身在动词
分级表中属于Level3（认知/调查类），因此更精确的判断应为Level3。

未匹配上述两种模式的轻动词记录，暂时维持"统一归Level2"的兜底规则处理，并标记
`note_light_verb_reclassified = "no_deverbal_match_default_L2"`以便后续追溯。

**处理原则**：F1阶段句子层面已存在的人工判断（`manual_level_override`，即
针对8条"have"句子的keep/exclude/Level判断）在本轮精细化重分类中被重置，改由
F3统一规则重新判断，理由是F1阶段的判断早于导师关于轻动词的正式建议提出、标准
尚不完全一致，重新统一处理可以保证全数据集口径一致，避免新旧标准混用。

### 待办事项

- `DEVERBAL_NOUN_MAP` 需要根据实际匹配情况持续补充（每次跑批后"未匹配"部分
  需要抽查，判断是否需要新增词条，还是维持默认Level2兜底）
- 完成F1-F3后，需要重新生成动词表、重新统计Level 1/2/3最终分布；此前基于
  1557/1560条数据得出的图表（Level分布图、RQ1图、Top动词图）需要用最终版本
  数据（预计约1540条附近，具体以F3跑批结果为准）重新生成一次

---

## Step G：最终定稿——重新生成统计图表 + 高频动词语义子分类（导师建议F2的落实）

### G1：最终数据集确认

经Step F全部处理完成后（缝合候选审核合并、17条feature残留清理、轻动词精细化
重分类、mode/rag/rear等新发现的解析错误修正），最终纳入统计的干净数据集为
**1538条**，动词种类共**330个**（含最终新增的45个此前未覆盖词，如operate/
learn/guide/rage等，已逐一人工核实分级）。

### G2：最终统计结果

**RQ1（句法角色分布）**：
- Active Subject: 885条 (57.54%)
- Object: 549条 (35.70%)
- Passive Subject: 78条 (5.07%)
- Passive Subject (by-agent): 26条 (1.69%)

**RQ2 / Level分布**：
- Level 1: 627条 (40.77%)
- Level 2: 690条 (44.86%)
- Level 3: 221条 (14.37%)

与Step F之前（保守版1540条阶段）的初步结果（40.69/43.59/15.71及RQ1
56.94/36.19/5.07/1.80）相比，最终数字变化幅度很小，核心结论保持稳定：
Level 2+3合计59.23%，显著高于Level 1，支持Hypothesis 1；Active Subject
(57.54%)显著高于Object和Passive Subject之和，同样支持H1。这也印证了缝合候选
人工审核、轻动词精细化重分类等后续精修步骤，主要作用是提升数据可靠性与颗粒度，
未对核心研究结论产生实质性影响——可作为结果稳健性的间接佐证。

### G3：高频动词语义子分类（对应导师建议F2，落实discussion_insights_log.md第8点）

以频次≥5为界，识别出51个高频动词（累计500次实例，占纳入统计总数的32.5%），
其余279个低频动词（累计411次）建议列入附录、不逐一展开。

对51个高频动词，按"拟人化标记性"（即该动词在日常英语中是否专属/近乎专属于
有意识主体，还是同样常见于非人类、机构、系统等主语）进行三档分类：

- **Level 2 - Weak（26个）**：help, do, make, offer, take, get, change,
  create, add, work, use, play, come, go, improve, write, start, turn,
  allow, include, launch, end, continue, give, bring, require——日常英语中
  本就常见于非人类/无生命主语
- **Level 2 - Medium（12个）**：generate, transform, power, replace,
  enhance, reshape, deliver, enable, act, drive, solve, emerge——带有更强的
  主动施加影响的语义色彩
- **Level 3 - Medium（7个）**：say, suggest, see, respond, identify, look,
  lack——具备认知/言语功能，但也见于仪器、系统等技术语境
- **Level 3 - Strong（6个）**：know, remember, think, feel, seem, promise
  ——近乎专属于有意识主体，是数据中拟人化程度最集中的动词群

**关键发现**：Level 2高频词中70%（26/37）落在Weak组，而Level 3中Strong+Medium
合计占100%、其中46%（6/13）为Strong——这一语义标记性沿Level 1→2→3递增的模式，
与句法层面的三层分类结果相互印证，为三层框架的理论有效性提供了独立的交叉验证
证据，建议作为Findings部分的一个重点呈现内容。

详细分类见 `high_freq_verbs_semantic_classification.xlsx`。

### 产出文件清单（最终版）

- `step2_FINAL_complete_dataset.xlsx` — 最终完整数据集（1538条）
- `step3_unique_verbs_FINAL.xlsx` — 最终动词表（330个）
- `light_verb_reclassification_summary.xlsx` — 轻动词重分类明细（89条）
- `verbs_high_frequency_FINAL.xlsx` — 高频动词表（51个，≥5次）
- `high_freq_verbs_semantic_classification.xlsx` — 高频动词语义子分类表
- `FINAL_level_summary_v2.xlsx` / `level_distribution_chart_FINAL.png` — 最终Level分布
- `RQ1_FINAL.xlsx` — 最终RQ1句法角色分布
- `top_verbs_by_level_FINAL.png` — 最终Top动词图

---

## Step H：最终一轮修正——get/go/come的become-equivalent用法排除

### 背景

在撰写方法论中关于seem/appear/become的动态判断段落时，回溯确认了acomp
（形容词补语）规则的实际含义：接形容词补语的linking verb判定为Relational
Process予以排除，理由是这类结构只赋予AI Attribute（属性），未赋予其
Actor/Senser等主动参与者角色——即便某些表达（如"AI seems capable"）在语用
层面仍可能带有拟人化印象，这类"属性性拟人化"与本研究测量的"施事性拟人化"
是不同的机制，前者不纳入Level 1-3统计（该区分在方法论正文中说明，不在
Discussion中展开，以保持论文聚焦于句法分类这一核心议题）。

在确认这一标准后，进一步排查发现get/go/come三个动词存在同样性质的
"become-equivalent"用法（如"AI **gets** more sophisticated"、"AI **goes**
rogue"、"AI doesn't **come** cheap"），语义功能等同于"becomes X"，此前未被
纳入排查范围。按与become一致的acomp检测规则重新排查，共排除7条记录。

### 排查过程

在系统性排查"是否存在其他遗漏的轻动词/关系动词候选"时，对照英语文献中
常见轻动词清单（have, take, make, give, do, get, put, keep, set, hold,
pay, run, come, go, turn, bring等），逐一核查其在最终动词表中的出现频次
及实际搭配的宾语/补语类型。确认：
- offer, turn, bring, give：搭配的均为具体名词性宾语，非名词化动词，
  维持原Material Process判断，无需调整
- get, go, come：除大部分为正常Material用法外，各自存在若干"动词+形容词
  补语"的become-equivalent用法，按acomp规则排除（get: 18→14, go: 9→7,
  come: 10→9，共排除7条）

### 最终统计结果（本研究报告的最终版本）

**RQ1（句法角色分布）**：
- Active Subject: 878条 (57.35%)
- Object: 549条 (35.86%)
- Passive Subject: 78条 (5.09%)
- Passive Subject (by-agent): 26条 (1.70%)

**RQ2 / Level分布**：
- Level 1: 627条 (40.95%)
- Level 2: 683条 (44.61%)
- Level 3: 221条 (14.44%)

最终纳入统计句子数：**1531条**；独立动词数：**330个**。

与Step G阶段（1538条）相比，数字变化幅度低于0.2个百分点，核心结论（H1:
Level2+3合计59.05%显著高于Level1；Active Subject显著高于Object与Passive
Subject之和）保持完全稳定，进一步印证方法论后期精修对结果稳健性的正面
支持。

**高频动词语义子分类表**（`high_freq_verbs_semantic_classification.xlsx`）
无需更新：受影响的get/go/come三词频次略有下降，但Level归类（均为Level2-
Weak）未发生变化。

### 产出文件清单（本研究最终定稿版本，供论文写作直接引用）

- `step2_FINAL_complete_dataset.xlsx` — 最终完整数据集（1531条）
- `step2_FINAL_complete_dataset_BACKUP_1538.xlsx` — Step H修正前的备份版本
- `step3_unique_verbs_FINAL.xlsx` — 最终动词表（330个）
- `light_verb_reclassification_summary.xlsx` — 轻动词重分类明细
- `verbs_high_frequency_FINAL.xlsx` — 高频动词表（51个，≥5次）
- `high_freq_verbs_semantic_classification.xlsx` — 高频动词语义子分类表
- `FINAL_level_summary_v2.xlsx` / `level_distribution_chart_FINAL.png` — 最终Level分布
- `RQ1_FINAL.xlsx` — 最终RQ1句法角色分布
- `top_verbs_by_level_FINAL.png` — Top动词图（基于1538条版本生成，数值极小
  幅度过时但不影响图形整体形态，如需完全精确可重新生成）
