#!/usr/bin/env python3
"""
Generate standard academic three-line tables (LaTeX booktabs, Markdown, and CSV)
for the downstream classification experiments in experiments_mini.
"""

import os
import json
from pathlib import Path
import pandas as pd

EXP_ROOT = "/dataset/downstream_classification/experiments_mini"
OUT_DIRS = [
    Path("/dataset/downstream_classification/result"),
    Path("/workspace/class/result"),
]

MODEL_MAP = {
    "dinov2s_mil": "DINOv2-S",
    "deit_mil": "DeiT-S",
    "resnet50_mil": "ResNet-50",
    "resnet18_mil": "ResNet-18",
}

GROUP_ORDER = ["cect_real", "cect_syn", "cect_real_syn", "cect_real_aug"]
MODEL_ORDER = ["DINOv2-S", "DeiT-S", "ResNet-50", "ResNet-18"]


def collect_metrics(mode="val"):
    records = []
    base = Path(EXP_ROOT)
    if not base.exists():
        return records

    for m_dir in base.iterdir():
        if not m_dir.is_dir() or m_dir.name not in MODEL_MAP:
            continue
        model_name = MODEL_MAP[m_dir.name]

        for g_dir in m_dir.iterdir():
            if not g_dir.is_dir() or g_dir.name not in GROUP_ORDER:
                continue
            group = g_dir.name

            target_file = g_dir / ("best_val.json" if mode == "val" else "test_results.json")
            if not target_file.exists():
                continue

            with open(target_file) as f:
                d = json.load(f)

            m = d if mode == "val" else d.get("metrics", {})
            t3 = m.get("three_class", {})

            escaped_group = group.replace('_', r'\_')
            regime_tex = f"\\texttt{{{escaped_group}}}"
            records.append({
                "model_key": m_dir.name,
                "Backbone": model_name,
                "Training Regime": regime_tex,
                "Regime_raw": group,
                "Macro-AUC": m.get("macro_auc", 0.0),
                "3-Class AUC": t3.get("macro_auc", 0.0),
                "Accuracy": m.get("accuracy", 0.0),
                "Balanced Acc": m.get("balanced_accuracy", 0.0),
                "Macro-F1": m.get("macro_f1", 0.0),
                "3-Class Acc": t3.get("accuracy", 0.0),
            })

    # Sort according to MODEL_ORDER and GROUP_ORDER
    records.sort(key=lambda r: (
        MODEL_ORDER.index(r["Backbone"]) if r["Backbone"] in MODEL_ORDER else 99,
        GROUP_ORDER.index(r["Regime_raw"]) if r["Regime_raw"] in GROUP_ORDER else 99
    ))
    return records


def generate_latex(records, split_name="Validation", sample_count=35):
    caption = f"{split_name} performance comparison across backbones and multi-phase CECT data regimes ($N={sample_count}$, Balanced Mini cohort)."
    label = f"tab:{split_name.lower()}_cect_results_mini"

    # Group by Backbone to handle multirow and find best per backbone
    by_backbone = {}
    for r in records:
        by_backbone.setdefault(r["Backbone"], []).append(r)

    lines = [
        "% --- Standard Academic Three-line Table (LaTeX booktabs format) ---",
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "\\textbf{Backbone} & \\textbf{Training Regime} & \\textbf{Macro-AUC} & \\textbf{3-Class AUC} & \\textbf{Accuracy} & \\textbf{Balanced Acc} & \\textbf{Macro-F1} \\\\",
        "\\midrule",
    ]

    backbones = list(by_backbone.keys())
    for idx, bname in enumerate(backbones):
        group_rows = by_backbone[bname]
        nrows = len(group_rows)

        # Find best values within this backbone for bolding
        best_auc5 = max(r["Macro-AUC"] for r in group_rows)
        best_auc3 = max(r["3-Class AUC"] for r in group_rows)
        best_acc = max(r["Accuracy"] for r in group_rows)
        best_bacc = max(r["Balanced Acc"] for r in group_rows)
        best_f1 = max(r["Macro-F1"] for r in group_rows)

        for row_idx, r in enumerate(group_rows):
            auc5_str = f"\\textbf{{{r['Macro-AUC']:.4f}}}" if r["Macro-AUC"] == best_auc5 else f"{r['Macro-AUC']:.4f}"
            auc3_str = f"\\textbf{{{r['3-Class AUC']:.4f}}}" if r["3-Class AUC"] == best_auc3 else f"{r['3-Class AUC']:.4f}"
            acc_str = f"\\textbf{{{r['Accuracy']:.4f}}}" if r["Accuracy"] == best_acc else f"{r['Accuracy']:.4f}"
            bacc_str = f"\\textbf{{{r['Balanced Acc']:.4f}}}" if r["Balanced Acc"] == best_bacc else f"{r['Balanced Acc']:.4f}"
            f1_str = f"\\textbf{{{r['Macro-F1']:.4f}}}" if r["Macro-F1"] == best_f1 else f"{r['Macro-F1']:.4f}"

            prefix = f"\\multirow{{{nrows}}}{{*}}{{{bname}}}" if row_idx == 0 else ""
            line = f"{prefix:25} & {r['Training Regime']:25} & {auc5_str} & {auc3_str} & {acc_str} & {bacc_str} & {f1_str} \\\\"
            lines.append(line)

        if idx < len(backbones) - 1:
            lines.append("\\midrule")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])
    return "\n".join(lines)


def generate_markdown(records, split_name="Validation", sample_count=35):
    lines = [
        f"# {split_name} 集（{split_name} Set）多期相 CECT 实验结果三线表",
        "",
        f"> **数据队列**：{sample_count} 例平衡子集患者（Balanced Mini，每类 7 例，完全评测真实 CECT 影像）  ",
        "> **表格规范**：学术论文三线表格式（顶线 / 栏目线 / 底线）",
        "",
        "| 骨干网络 (Backbone) | 实验分组 (Regime) | Macro-AUC (5分类) | 3-Class AUC (良恶性) | Accuracy (5分类) | 3-Class Acc | Macro-F1 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    by_backbone = {}
    for r in records:
        by_backbone.setdefault(r["Backbone"], []).append(r)

    for idx, (bname, group_rows) in enumerate(by_backbone.items()):
        best_auc5 = max(r["Macro-AUC"] for r in group_rows)
        best_auc3 = max(r["3-Class AUC"] for r in group_rows)
        best_acc = max(r["Accuracy"] for r in group_rows)
        best_acc3 = max(r["3-Class Acc"] for r in group_rows)
        best_f1 = max(r["Macro-F1"] for r in group_rows)

        for row_idx, r in enumerate(group_rows):
            b_cell = f"**{bname}**" if row_idx == 0 else ""
            auc5 = f"**{r['Macro-AUC']:.4f}**" if r["Macro-AUC"] == best_auc5 else f"{r['Macro-AUC']:.4f}"
            auc3 = f"**{r['3-Class AUC']:.4f}**" if r["3-Class AUC"] == best_auc3 else f"{r['3-Class AUC']:.4f}"
            acc = f"**{r['Accuracy']:.4f}**" if r["Accuracy"] == best_acc else f"{r['Accuracy']:.4f}"
            acc3 = f"**{r['3-Class Acc']:.4f}**" if r["3-Class Acc"] == best_acc3 else f"{r['3-Class Acc']:.4f}"
            f1 = f"**{r['Macro-F1']:.4f}**" if r["Macro-F1"] == best_f1 else f"{r['Macro-F1']:.4f}"
            lines.append(f"| {b_cell} | `{r['Regime_raw']}` | {auc5} | {auc3} | {acc} | {acc3} | {f1} |")

        if idx < len(by_backbone) - 1:
            lines.append("| | | | | | | |")

    return "\n".join(lines)


def generate_csv(records):
    csv_rows = []
    for r in records:
        csv_rows.append({
            "Backbone": r["Backbone"],
            "Training Regime": r["Regime_raw"],
            "Macro-AUC": round(r["Macro-AUC"], 4),
            "3-Class AUC": round(r["3-Class AUC"], 4),
            "Accuracy": round(r["Accuracy"], 4),
            "Balanced Acc": round(r["Balanced Acc"], 4),
            "Macro-F1": round(r["Macro-F1"], 4),
            "3-Class Acc": round(r["3-Class Acc"], 4),
        })
    return pd.DataFrame(csv_rows)


def main():
    for mode in ["val", "test"]:
        records = collect_metrics(mode)
        if not records:
            print(f"No records found for mode={mode}")
            continue

        split_label = "Validation" if mode == "val" else "Test"
        tex_content = generate_latex(records, split_name=split_label)
        md_content = generate_markdown(records, split_name=split_label)
        df_csv = generate_csv(records)

        file_prefix = f"{mode}_three_line_table_mini"

        for out_dir in OUT_DIRS:
            out_dir.mkdir(parents=True, exist_ok=True)
            tex_path = out_dir / f"{file_prefix}.tex"
            md_path = out_dir / f"{file_prefix}.md"
            csv_path = out_dir / f"{file_prefix}.csv"

            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(tex_content + "\n")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content + "\n")
            df_csv.to_csv(csv_path, index=False)

            print(f"Saved: {tex_path}")
            print(f"Saved: {md_path}")
            print(f"Saved: {csv_path}")

        # Also print to stdout
        print("\n" + "=" * 80)
        print(f">> LaTeX Three-Line Table ({split_label} Set)")
        print("=" * 80)
        print(tex_content)
        print("\n")


if __name__ == "__main__":
    main()
