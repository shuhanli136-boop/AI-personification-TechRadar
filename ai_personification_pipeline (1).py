# -*- coding: utf-8 -*-
"""
TechRadar AI Personification Corpus Pipeline
Step 2 (句子筛选) -> Step 3 (排除标准) -> Step 4 (句法角色) -> Step 5 (语义分级)

使用前准备：
    pip install spacy pandas openpyxl
    python -m spacy download en_core_web_sm
    (可选，若要自动语义打标) pip install pymusas
        参考 https://ucrel.github.io/pymusas/ 下载英文 lexicon 组件

目录结构假设：
    corpus_folder/
        2025_Jan_001.txt
        2025_Jan_002.txt
        ...
    每个 txt 文件名 = metadata 表里的 File ID + ".txt"，内容为纯正文（无多余元信息）。

运行：
    python ai_personification_pipeline.py --corpus_dir ./JAN_cleaned --metadata_xlsx ./2025_Corpus_Metadata_Full_Year_1.xlsx --sheet Jan --out_dir ./output_Jan
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import spacy

# ---------------------------------------------------------------------------
# 配置区：如需调整关键词/排除逻辑，改这里即可
# ---------------------------------------------------------------------------

# Step 2 — 显性 AI 指称词（大小写不敏感）。按需增删。
# 注意：允许可选的复数 s（AIs / LLMs / models 等）
AI_PATTERN = re.compile(
    r"\b(AIs?|artificial intelligence|chatgpts?|generative AIs?|large language models?|llms?)\b",
    re.IGNORECASE,
)

# Step 3 — 主句判断：AI token 沿 head 链往上走，若经过这些依存关系，判定为从句/转述句，排除
SUBORDINATE_DEPS = {"ccomp", "advcl", "acl", "relcl", "xcomp"}

# Step 4 — 句法角色：决定是否进入"及物性主动施事"统计，还是直接归 Level 1
ACTIVE_DEPS = {"nsubj", "agent"}       # AI 作主动主语 / 被动句里的by-agent -> 需要看动词
PASSIVE_OR_OBJECT_DEPS = {"nsubjpass", "dobj"}  # 直接归 Level 1（如你的模型是新版UD，注意可能是 "nsubj:pass"）

# 排除类别：modifier / possessive / prepositional instrument
EXCLUDE_MODIFIER_DEPS = {"compound", "amod"}
EXCLUDE_POSSESSIVE_DEPS = {"poss"}
INSTRUMENT_PREPS = {"with", "using", "via", "through"}

# 关系动词（Relational Process）——按Appendix C，这类过程不赋予AI主动参与者角色
# （AI是Carrier/Attribute而非Actor/Senser/Sayer），需要在动词提取阶段就排除。
#
# 分两组处理：
# 1) ALWAYS_RELATIONAL_VERBS —— 在这类科技新闻语境下，几乎总是作关系动词使用，
#    直接无条件排除。
# 2) AMBIGUOUS_PERCEPTION_VERBS —— look/sound/feel/taste/smell 这类感官动词是
#    "两用词"：接形容词补语(acomp)时是关系动词（"AI looks smart"，该排除）；
#    但也常作实义动词使用（"AI looks for patterns"、"AI feels the vibration"，
#    这时候是真实的Material/Mental过程，不该排除）。这组词不能无脑拉黑，
#    要在实际句子里检查有没有形容词补语，动态判断。
ALWAYS_RELATIONAL_VERBS = {
    "be", "become", "remain", "constitute", "signify", "equal",
}
AMBIGUOUS_PERCEPTION_VERBS = {"look", "sound", "feel", "taste", "smell", "seem", "appear"}

# represent / mean 这两个词，"关系性用法"（等同于/象征）和"非关系性用法"
# （代表/为...发声；打算/意图）在句法结构上长得几乎一样（都是 动词+名词宾语），
# 没有像acomp那样的可靠句法信号能自动区分，只能强制标记为人工核查，
# 不做自动判定，避免用规则强行分类反而引入新的系统性偏差。
FORCE_MANUAL_REVIEW_VERBS = {"represent", "mean"}


def _is_relational_usage(verb_token) -> bool:
    """检查这个动词token是否带形容词补语(acomp)——带了就是关系动词用法。"""
    return any(child.dep_ == "acomp" for child in verb_token.children)

# Lexis+ 关键词检索导出时，未被保留的句子/换行会被标成省略号，
# 但同时也会误吞正常换行——用它切片段，片段首尾不完整的句子单独标记排除。
ELLIPSIS_SPLIT_PATTERN = re.compile(r"\.\.\.\s*\.\.\.|\.\.\.")


# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------

# 项目符号（listicle类文章常见）不会被spaCy当作句子边界，需要预先替换成句号
BULLET_PATTERN = re.compile(r"\s*[•·▪]\s*")


def preprocess_bullets(text: str) -> str:
    """把列表符号替换成句号+空格，帮助spaCy正确切分要点列表。"""
    return BULLET_PATTERN.sub(". ", text)


def split_into_fragments(raw_text: str, enable_stitching: bool = False):
    """
    按 Lexis+ 导出遗留的省略号把原文切成片段列表。

    enable_stitching=False（默认，推荐）：
        不做任何缝合，每个省略号都当作真实边界处理。

    enable_stitching=True：
        尝试把"前一片段无句末标点 + 后一片段小写开头"的情况缝合。

    返回 [(fragment_text, join_positions), ...]
    join_positions 是这个片段内所有"拼接点"在合并后文本里的字符位置列表
    （用于后续精确判断：只有真正跨越拼接点的那一句，才算被缝合过，
    而不是把整个片段里所有句子都算作缝合——避免把片段内本来就完整、
    跟缝合毫无关系的句子也误标记）。
    """
    raw_fragments = [f.strip() for f in ELLIPSIS_SPLIT_PATTERN.split(raw_text)]
    raw_fragments = [f for f in raw_fragments if f]

    if not raw_fragments:
        return []

    if not enable_stitching:
        return [(f, []) for f in raw_fragments]

    merged = [raw_fragments[0]]
    join_positions = [[]]
    for frag in raw_fragments[1:]:
        prev = merged[-1]
        prev_ends_incomplete = len(prev) > 0 and prev[-1] not in ".!?\"'”’"
        next_starts_lowercase = len(frag) > 0 and frag[0].islower()

        if prev_ends_incomplete and next_starts_lowercase:
            join_pos = len(prev.rstrip())  # 拼接发生的字符位置（拼接空格所在处）
            merged[-1] = prev.rstrip() + " " + frag.lstrip()
            join_positions[-1].append(join_pos)
        else:
            merged.append(frag)
            join_positions.append([])

    return list(zip(merged, join_positions))


def sentence_is_fragment_start(sent_text: str) -> bool:
    """片段第一句若不以大写字母开头（真正的引号开头除外），判定为被截断的句子后半部分。
    注意：像 "'s not ignoring AI" 这种撇号+小写字母（缩写残留，如 he's / it's 被截断）
    不算合法的引号开头，也应判定为不完整。"""
    stripped = sent_text.strip()
    if not stripped:
        return True
    first_char = stripped[0]
    if first_char.isupper():
        return False
    if first_char in "\"'“‘" and len(stripped) > 1 and stripped[1].isupper():
        return False  # 真正的引号开头，后面紧跟大写字母，视为合法
    return True


def sentence_is_fragment_end(sent_text: str) -> bool:
    """片段最后一句若不以句末标点结尾，判定为被截断的句子前半部分。"""
    stripped = sent_text.strip()
    if not stripped:
        return True
    return stripped[-1] not in ".!?\"'”’"

def is_main_clause(token) -> bool:
    """沿着 head 链往上走，检查是否经过从句标记。True = 在主句里。"""
    node = token
    visited = set()
    while node.head != node and node.i not in visited:
        visited.add(node.i)
        if node.dep_ in SUBORDINATE_DEPS:
            return False
        node = node.head
    return True


def _valid_verb_lemma(lemma: str) -> bool:
    """校验提取出来的'动词'是否像一个正常的英语单词（纯字母、长度合理）。
    像"max+"、"395"这种因为产品型号(Ryzen AI Max+ 395)把依存句法搞乱后
    提取出来的东西，会在这里被拦截，标记为提取失败，不参与Level分级。"""
    return bool(lemma) and lemma.isalpha() and 1 < len(lemma) <= 20


def classify_ai_token(tok):
    """
    对句子里匹配到 AI 的 token，判断其句法角色 / 排除原因。
    返回 dict：{ai_dep, exclusion_reason, governing_verb}
    """
    result = {"ai_dep": tok.dep_, "exclusion_reason": None, "governing_verb": None}

    # 排除：modifier（AI-powered / AI-driven 这类，AI是compound/amod修饰语）
    if tok.dep_ in EXCLUDE_MODIFIER_DEPS:
        result["exclusion_reason"] = "modifier"
        return result

    # 排除：possessive（AI's decision）
    if tok.dep_ in EXCLUDE_POSSESSIVE_DEPS:
        result["exclusion_reason"] = "possessive"
        return result

    # 排除：prepositional instrument（using AI / with AI）
    if tok.dep_ == "pobj" and tok.head.dep_ == "prep" and tok.head.text.lower() in INSTRUMENT_PREPS:
        result["exclusion_reason"] = "prepositional instrument"
        return result

    # 排除：不在主句里（从句/转述引语）
    if not is_main_clause(tok):
        result["exclusion_reason"] = "subordinate clause"
        return result

    # 主动施事（nsubj）-> 需要提取动词，后续做语义分级（Level 2/3）
    if tok.dep_ == "nsubj":
        verb_token = tok.head
        verb = verb_token.lemma_.lower()
        if verb in FORCE_MANUAL_REVIEW_VERBS:
            result["exclusion_reason"] = "manual_review_ambiguous_verb"
            return result
        if verb in ALWAYS_RELATIONAL_VERBS:
            result["exclusion_reason"] = "relational_process"
            return result
        if verb in AMBIGUOUS_PERCEPTION_VERBS and _is_relational_usage(verb_token):
            result["exclusion_reason"] = "relational_process"
            return result
        if not _valid_verb_lemma(verb):
            result["exclusion_reason"] = "invalid_verb_extraction"
            return result
        result["governing_verb"] = verb
        return result

    # 被动句 by-agent：AI 是 "by" 的宾语(pobj)，"by" 本身的 dep_ 才是 agent
    # 例："jobs are threatened by AI" -> AI.dep_=="pobj", AI.head.text=="by", AI.head.dep_=="agent"
    if tok.dep_ == "pobj" and tok.head.dep_ == "agent":
        verb_token = tok.head.head
        verb = verb_token.lemma_.lower()
        if verb in FORCE_MANUAL_REVIEW_VERBS:
            result["exclusion_reason"] = "manual_review_ambiguous_verb"
            return result
        if verb in ALWAYS_RELATIONAL_VERBS:
            result["exclusion_reason"] = "relational_process"
            return result
        if verb in AMBIGUOUS_PERCEPTION_VERBS and _is_relational_usage(verb_token):
            result["exclusion_reason"] = "relational_process"
            return result
        if not _valid_verb_lemma(verb):
            result["exclusion_reason"] = "invalid_verb_extraction"
            return result
        result["governing_verb"] = verb
        return result

    # 被动主语 / 直接宾语 -> 直接 Level 1
    if tok.dep_ in PASSIVE_OR_OBJECT_DEPS or tok.dep_ == "nsubj:pass":
        result["governing_verb"] = None  # Level 1 不需要动词分级
        return result

    # 其余情况（如表语、同位语等），暂标记为 other，供人工检查
    result["exclusion_reason"] = "other_unclassified"
    return result


def find_ai_tokens(sent, pattern):
    """
    在句子里找出所有AI相关指称的token。

    不再要求"整个token"完全等于关键词（那样会漏掉"artificial intelligence"、
    "large language model"这种多词短语，因为spaCy会把它们拆成好几个token，
    没有哪个token能单独完整匹配整个短语）。

    改用正则在句子原文里定位匹配到的字符范围，再映射回对应的token span，
    取这个span里**最后一个token**作为分析对象——英语里像"artificial intelligence"
    "large language model"这类名词短语，核心词(head)在最后一个词
    （intelligence / model），用它做后续依存句法判断更准确。
    """
    matches = []
    doc = sent.doc
    for m in pattern.finditer(sent.text):
        abs_start = sent.start_char + m.start()
        abs_end = sent.start_char + m.end()
        span = doc.char_span(abs_start, abs_end, alignment_mode="expand")
        if span is not None and len(span) > 0:
            matches.append(span[-1])
    return matches


def process_corpus(corpus_dir: Path, nlp, enable_stitching: bool = False):
    records = []
    txt_files = sorted(corpus_dir.glob("*.txt"))
    print(f"共发现 {len(txt_files)} 个txt文件")

    for fp in txt_files:
        file_id = fp.stem  # 文件名（不含扩展名）作为 File ID
        raw_text = fp.read_text(encoding="utf-8", errors="ignore")
        raw_text = preprocess_bullets(raw_text)  # 先处理列表符号，避免要点被误粘连

        # 先按省略号切成片段
        fragments = split_into_fragments(raw_text, enable_stitching=enable_stitching)

        global_sent_idx = 0
        for frag_idx, (fragment, join_positions) in enumerate(fragments):
            doc = nlp(fragment)
            sents = list(doc.sents)
            n_sents = len(sents)

            for local_idx, sent in enumerate(sents):
                if not AI_PATTERN.search(sent.text):
                    global_sent_idx += 1
                    continue  # Step 2: 不含AI指称词，直接跳过

                # 只有这句话的字符范围真正跨越了某个拼接点，才算被缝合过
                was_stitched = any(
                    sent.start_char <= jp < sent.end_char for jp in join_positions
                )

                # 片段边界不完整性检测：只对片段的第一句/最后一句做判断
                incomplete_reason = None
                if local_idx == 0 and sentence_is_fragment_start(sent.text):
                    incomplete_reason = "incomplete_fragment_start"
                elif local_idx == n_sents - 1 and sentence_is_fragment_end(sent.text):
                    incomplete_reason = "incomplete_fragment_end"

                ai_tokens = find_ai_tokens(sent, AI_PATTERN)
                if not ai_tokens:
                    global_sent_idx += 1
                    continue

                for tok in ai_tokens:
                    if incomplete_reason:
                        cls = {"ai_dep": tok.dep_, "exclusion_reason": incomplete_reason, "governing_verb": None}
                    else:
                        cls = classify_ai_token(tok)
                    records.append({
                        "sent_id": f"{file_id}_frag{frag_idx}_s{global_sent_idx}",
                        "file_id": file_id,
                        "sentence_text": sent.text.strip(),
                        "ai_span_text": tok.text,
                        "ai_dep": cls["ai_dep"],
                        "exclusion_reason": cls["exclusion_reason"],
                        "governing_verb": cls["governing_verb"],
                        "was_stitched": was_stitched,  # 精确到这一句话本身是否跨越拼接点
                    })
                global_sent_idx += 1

    return pd.DataFrame(records)


def assign_level(row, verb_level_map: dict):
    """
    Step 5: 根据 exclusion_reason / ai_dep / governing_verb 分配最终 level。
    verb_level_map: {verb_lemma: 2 or 3}，需要你先跑一遍去重动词列表、
                    用 PyMUSAS 或人工判断后手动/自动填好这个映射表。
    """
    if row["exclusion_reason"] is not None:
        return None  # 被排除，不参与统计
    if row["ai_dep"] in PASSIVE_OR_OBJECT_DEPS or row["ai_dep"] == "nsubj:pass":
        return 1
    if row["governing_verb"] is not None:
        return verb_level_map.get(row["governing_verb"], None)  # None = 待人工确认的未知动词
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", required=True, help="存放 txt 文件的文件夹路径")
    parser.add_argument("--metadata_xlsx", required=False, help="metadata xlsx 路径（可选，用于join日期/来源）")
    parser.add_argument("--sheet", required=False, help="metadata 对应的月份 sheet 名，如 'Jan'")
    parser.add_argument("--model", default="en_core_web_sm", help="spaCy 模型名")
    parser.add_argument("--out_dir", default="./output", help="输出文件夹")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载 spaCy 模型: {args.model} ...")
    nlp = spacy.load(args.model)

    print("开始处理语料 ...")
    df = process_corpus(Path(args.corpus_dir), nlp)
    print(f"共提取 {len(df)} 条 AI 相关记录（未筛选前）")

    # 如果提供了 metadata，join 上日期/来源信息，方便溯源
    if args.metadata_xlsx and args.sheet:
        meta = pd.read_excel(args.metadata_xlsx, sheet_name=args.sheet)
        meta = meta.rename(columns={"File ID": "file_id", "Date": "article_date", "Source": "source"})
        df = df.merge(meta[["file_id", "article_date", "source"]], on="file_id", how="left")

    # ---- Step 1 快照：全部候选句（筛选前）----
    df.to_excel(out_dir / "step1_raw_ai_sentences.xlsx", index=False)

    # ---- Step 2 快照：应用排除标准后的情况一览 ----
    df.to_excel(out_dir / "step2_with_exclusions.xlsx", index=False)

    # ---- 去重动词列表，供你人工/PyMUSAS 打语义标签 ----
    verbs = (
        df[df["governing_verb"].notna()]["governing_verb"]
        .value_counts()
        .reset_index()
    )
    verbs.columns = ["verb_lemma", "frequency"]
    verbs.to_excel(out_dir / "step3_unique_verbs_for_tagging.xlsx", index=False)
    print(f"共发现 {len(verbs)} 个独立动词，已导出至 step3_unique_verbs_for_tagging.xlsx")
    print("请在该表里新增一列 'level'（填 2 或 3），或用 PyMUSAS/Wmatrix 自动打标后合并。")

    print("\n=== 完成 Step 1-4，等待动词分级表填好后运行 Step 5（见脚本内 assign_level 函数）===")


if __name__ == "__main__":
    main()
