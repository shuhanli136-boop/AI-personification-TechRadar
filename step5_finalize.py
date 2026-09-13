# -*- coding: utf-8 -*-
"""
Step 5: 合并动词分级表 -> 计算每句最终 Level -> 输出统计分布

前提：已经运行过 ai_personification_pipeline.py，
      并且已经在 step3_unique_verbs_for_tagging.xlsx 里手动/自动填好了 'level' 列（2 或 3）。

运行：
    python step5_finalize.py --out_dir ./output_Jan
"""

import argparse
from pathlib import Path

import pandas as pd

PASSIVE_OR_OBJECT_DEPS = {"nsubjpass", "dobj"}


def assign_level(row, verb_level_map: dict):
    if pd.notna(row["exclusion_reason"]):
        return None  # 被排除，不参与统计
    if row["ai_dep"] in PASSIVE_OR_OBJECT_DEPS or row["ai_dep"] == "nsubj:pass":
        return 1
    if pd.notna(row["governing_verb"]):
        return verb_level_map.get(row["governing_verb"])  # 找不到 -> None，需人工确认
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True, help="Step1-4 的输出文件夹（与pipeline脚本一致）")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    df = pd.read_excel(out_dir / "step2_with_exclusions.xlsx")
    verbs = pd.read_excel(out_dir / "step3_unique_verbs_for_tagging.xlsx")

    if "level" not in verbs.columns:
        raise ValueError("step3_unique_verbs_for_tagging.xlsx 里还没有 'level' 列，请先手动填好再运行本脚本。")

    verb_level_map = dict(zip(verbs["verb_lemma"], verbs["level"]))

    df["level"] = df.apply(lambda r: assign_level(r, verb_level_map), axis=1)

    # 完整明细表（含所有句子，含被排除的、含最终level）
    df.to_excel(out_dir / "step4_final_with_levels.xlsx", index=False)

    # 只看被纳入统计的句子
    included = df[df["level"].notna()].copy()
    included.to_excel(out_dir / "step5_included_only.xlsx", index=False)

    # 汇总统计：各level数量与占比
    summary = (
        included["level"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    summary.columns = ["level", "count"]
    summary["percentage"] = (summary["count"] / summary["count"].sum() * 100).round(2)

    # 排除原因分布，方便写方法论/核对
    exclusion_summary = (
        df[df["exclusion_reason"].notna()]["exclusion_reason"]
        .value_counts()
        .reset_index()
    )
    exclusion_summary.columns = ["exclusion_reason", "count"]

    with pd.ExcelWriter(out_dir / "step6_summary.xlsx") as writer:
        summary.to_excel(writer, sheet_name="level_distribution", index=False)
        exclusion_summary.to_excel(writer, sheet_name="exclusion_distribution", index=False)

    print("=== Level 分布 ===")
    print(summary.to_string(index=False))
    print("\n=== 排除原因分布 ===")
    print(exclusion_summary.to_string(index=False))
    print(f"\n找不到level映射的动词数（governing_verb非空但level为None）: "
          f"{df[(df['exclusion_reason'].isna()) & (df['governing_verb'].notna()) & (df['level'].isna())].shape[0]}")
    print("已输出：step4_final_with_levels.xlsx / step5_included_only.xlsx / step6_summary.xlsx")


if __name__ == "__main__":
    main()
