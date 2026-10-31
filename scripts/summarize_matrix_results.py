#!/usr/bin/env python3
"""
Summarize Matrix Results from experiments_mini.
Reads test_results.json from each (model, group) and displays a 3-line Markdown table.
"""

import json
from pathlib import Path
import pandas as pd

def summarize_matrix(work_dir="/dataset/downstream_classification/experiments_mini"):
    base = Path(work_dir)
    if not base.exists():
        print(f"Directory {base} does not exist yet.")
        return

    records = []
    for test_json in base.rglob("test_results.json"):
        try:
            with open(test_json) as f:
                data = json.load(f)
            rel_parts = test_json.relative_to(base).parts
            model = rel_parts[0] if len(rel_parts) > 1 else "unknown"
            group = rel_parts[1] if len(rel_parts) > 2 else "unknown"
            metrics = data.get("metrics", {})
            t3 = metrics.get("three_class", {})
            records.append({
                "Model": model,
                "Group": group,
                "Test Acc (5-Class)": round(metrics.get("accuracy", 0.0) * 100, 2),
                "Test Acc (3-Class)": round(t3.get("accuracy", 0.0) * 100, 2) if t3.get("accuracy") is not None else "N/A",
                "Macro-AUC (5-Class)": round(metrics.get("macro_auc", 0.0), 4) if metrics.get("macro_auc") is not None else "N/A",
                "3-Class AUC": round(t3.get("macro_auc", 0.0), 4) if t3.get("macro_auc") is not None else "N/A",
            })
        except Exception as e:
            pass

    if not records:
        print("No test_results.json found yet.")
        return

    df = pd.DataFrame(records).sort_values(by=["Model", "Group"])
    print("\n" + "=" * 80)
    print(">> Matrix Experiment Test Results Summary Table")
    print("=" * 80)
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df.to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    import sys
    wdir = sys.argv[1] if len(sys.argv) > 1 else "/dataset/downstream_classification/experiments_mini"
    summarize_matrix(wdir)
