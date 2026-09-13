# -*- coding: utf-8 -*-
"""
全年批处理 + 缝合句子人工审核合并流程

使用前提：
    已经在同一个 Colab / Python 环境里加载了 ai_personification_pipeline.py
    里的所有函数（process_corpus, classify_ai_token, split_into_fragments 等）。

整体流程分三个阶段，按顺序运行本文件里的对应函数：
    阶段A：run_conservative_and_stitched_batch()  -> 产出保守版 + 缝合候选审核表
    [手动步骤：打开 manual_review_stitched_all_year.xlsx，逐条填 manual_keep 列 Y/N]
    阶段B：merge_reviewed_dataset()                -> 合并成最终数据集 + 全年去重动词表
    [手动步骤：打开 step3_unique_verbs_all_year.xlsx，逐个动词填 level 列 2/3]
    阶段C：finalize_levels_and_summary()           -> 产出最终Level分布统计（全年+分月）
"""

from pathlib import Path
import pandas as pd
import spacy


# ---------------------------------------------------------------------------
# 配置区：把下面的路径改成你实际的文件夹路径
# ---------------------------------------------------------------------------

MONTH_FOLDERS = {
    "Jan": "/content/drive/MyDrive/full_year_corpus/Jan_Cleaned_Corpus_Ultimate",
    "Feb": "/content/drive/MyDrive/full_year_corpus/Feb_Cleaned_Corpus_Ultimate",
    "Mar": "/content/drive/MyDrive/full_year_corpus/Mar_Cleaned_Corpus_Ultimate",
    "Apr": "/content/drive/MyDrive/full_year_corpus/Apr_Cleaned_Corpus_Ultimate",
    "May": "/content/drive/MyDrive/full_year_corpus/May_Cleaned_Corpus_Ultimate",
    "Jun": "/content/drive/MyDrive/full_year_corpus/Jun_Cleaned_Corpus_Ultimate",
    "Jul": "/content/drive/MyDrive/full_year_corpus/Jul_Cleaned_Corpus_Ultimate",
    "Aug": "/content/drive/MyDrive/full_year_corpus/Aug_Cleaned_Corpus_Ultimate",
    "Sep": "/content/drive/MyDrive/full_year_corpus/Sep_Cleaned_Corpus_Ultimate",
    "Oct": "/content/drive/MyDrive/full_year_corpus/Oct_Cleaned_Corpus_Ultimate",
    "Nov": "/content/drive/MyDrive/full_year_corpus/Nov_Cleaned_Corpus_Ultimate",
    "Dec_Jan": "/content/drive/MyDrive/full_year_corpus/25DEC-26JAN1ST_Cleaned_Corpus_Ultimate",
}

OUT_DIR = Path("/content/drive/MyDrive/full_year_corpus/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "en_core_web_sm"


# ---------------------------------------------------------------------------
# 阶段 A：跑保守版 + 缝合版，导出缝合候选审核表
# ---------------------------------------------------------------------------

def run_conservative_and_stitched_batch(month_folders: dict = MONTH_FOLDERS,
                                          model_name: str = MODEL_NAME):
    nlp = spacy.load(model_name)

    conservative_dfs = []
    stitched_dfs = []

    for month, folder in month_folders.items():
        print(f"--- 处理 {month} ---")
        df_cons = process_corpus(Path(folder), nlp, enable_stitching=False)
        df_cons["month"] = month
        conservative_dfs.append(df_cons)

        df_stitch = process_corpus(Path(folder), nlp, enable_stitching=True)
        df_stitch["month"] = month
        stitched_dfs.append(df_stitch)

    df_conservative = pd.concat(conservative_dfs, ignore_index=True)
    df_stitched = pd.concat(stitched_dfs, ignore_index=True)

    print(f"\n保守版总记录数: {len(df_conservative)}")
    print(f"保守版纳入统计数: {df_conservative['exclusion_reason'].isna().sum()}")

    df_conservative.to_excel(OUT_DIR / "step2_conservative_all_year.xlsx", index=False)

    # 只导出"靠缝合才纳入统计"的候选句子，供人工审核
    review_set = df_stitched[
        (df_stitched["exclusion_reason"].isna()) & (df_stitched["was_stitched"] == True)
    ].copy()
    review_set["manual_keep"] = ""  # 留空，人工填 Y 或 N
    review_set.to_excel(OUT_DIR / "manual_review_stitched_all_year.xlsx", index=False)

    print(f"待人工审核的缝合候选句子数: {len(review_set)}")
    print("\n=== 阶段A完成 ===")
    print("请下载 manual_review_stitched_all_year.xlsx，逐条在 manual_keep 列填 Y 或 N，")
    print("填好后重新上传（或在Drive里直接编辑保存），再运行阶段B。")

    return df_conservative, review_set


# ---------------------------------------------------------------------------
# 阶段 B：合并人工审核结果，产出最终数据集 + 全年去重动词表
# ---------------------------------------------------------------------------

def merge_reviewed_dataset(df_conservative: pd.DataFrame,
                             reviewed_xlsx_path: str = None):
    if reviewed_xlsx_path is None:
        reviewed_xlsx_path = OUT_DIR / "manual_review_stitched_all_year.xlsx"

    reviewed = pd.read_excel(reviewed_xlsx_path)
    if "manual_keep" not in reviewed.columns:
        raise ValueError("找不到 manual_keep 列，请确认已经填好审核表再运行本函数。")

    approved = reviewed[reviewed["manual_keep"].astype(str).str.strip().str.upper() == "Y"]
    print(f"人工审核通过的缝合句子数: {len(approved)} / {len(reviewed)}")

    baseline_included = df_conservative[df_conservative["exclusion_reason"].isna()]

    final_dataset = pd.concat([baseline_included, approved], ignore_index=True)
    # 按纯句子文本去重（不看file_id），去掉跨文章重复出现的标题/推荐链接类模板文字
    # 注意：keep="first" 会保留先出现的那一条，baseline_included在前，approved在后，
    # 因此如果同一句话在两边都出现，优先保留baseline版本的记录
    final_dataset = final_dataset.drop_duplicates(subset=["sentence_text"], keep="first")

    print(f"最终纳入统计的句子总数（已按文本去重）: {len(final_dataset)}")
    final_dataset.to_excel(OUT_DIR / "step2_final_dataset.xlsx", index=False)

    # 全年去重动词表，供人工/PyMUSAS 打 level 标签
    verbs = (
        final_dataset[final_dataset["governing_verb"].notna()]["governing_verb"]
        .value_counts()
        .reset_index()
    )
    verbs.columns = ["verb_lemma", "frequency"]
    verbs.to_excel(OUT_DIR / "step3_unique_verbs_all_year.xlsx", index=False)
    print(f"全年独立动词数: {len(verbs)}")
    print("请打开 step3_unique_verbs_all_year.xlsx，新增 level 列（填2或3），填好后运行阶段C。")

    return final_dataset


# ---------------------------------------------------------------------------
# 阶段 C：合并动词分级，产出最终统计（全年 + 分月）
# ---------------------------------------------------------------------------

PASSIVE_OR_OBJECT_DEPS = {"nsubjpass", "dobj"}


def assign_level(row, verb_level_map: dict):
    if pd.notna(row["exclusion_reason"]):
        return None
    if row["ai_dep"] in PASSIVE_OR_OBJECT_DEPS or row["ai_dep"] == "nsubj:pass":
        return 1
    if pd.notna(row["governing_verb"]):
        return verb_level_map.get(row["governing_verb"])
    return None


def finalize_levels_and_summary(final_dataset: pd.DataFrame,
                                  verbs_xlsx_path: str = None):
    if verbs_xlsx_path is None:
        verbs_xlsx_path = OUT_DIR / "step3_unique_verbs_all_year.xlsx"

    verbs = pd.read_excel(verbs_xlsx_path)
    if "level" not in verbs.columns:
        raise ValueError("动词表里还没有 level 列，请先填好再运行本函数。")

    verb_level_map = dict(zip(verbs["verb_lemma"], verbs["level"]))

    final_dataset = final_dataset.copy()
    final_dataset["level"] = final_dataset.apply(lambda r: assign_level(r, verb_level_map), axis=1)

    final_dataset.to_excel(OUT_DIR / "step4_final_with_levels.xlsx", index=False)

    included = final_dataset[final_dataset["level"].notna()]

    overall = included["level"].value_counts().sort_index().reset_index()
    overall.columns = ["level", "count"]
    overall["percentage"] = (overall["count"] / overall["count"].sum() * 100).round(2)

    monthly = included.groupby(["month", "level"]).size().unstack(fill_value=0)

    with pd.ExcelWriter(OUT_DIR / "step6_summary_all_year.xlsx") as writer:
        overall.to_excel(writer, sheet_name="overall_level_distribution", index=False)
        monthly.to_excel(writer, sheet_name="monthly_level_distribution")

    print("=== 全年 Level 分布 ===")
    print(overall.to_string(index=False))
    print("\n=== 分月 Level 分布 ===")
    print(monthly)
    print("\n已输出：step4_final_with_levels.xlsx / step6_summary_all_year.xlsx")

    return included, overall, monthly


# ---------------------------------------------------------------------------
# 使用示例（在 Colab 里分阶段运行，不要一次性跑完 —— 阶段A和B之间需要你手动填审核表）
# ---------------------------------------------------------------------------
#
# 阶段A：
#   df_conservative, review_set = run_conservative_and_stitched_batch()
#
# [手动去 Excel 里填 manual_keep 列]
#
# 阶段B：
#   final_dataset = merge_reviewed_dataset(df_conservative)
#
# [手动去 Excel 里填 level 列]
#
# 阶段C：
#   included, overall, monthly = finalize_levels_and_summary(final_dataset)
