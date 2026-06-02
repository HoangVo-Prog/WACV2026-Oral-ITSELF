#!/usr/bin/env python3
"""Compare margin diagnostics between no-prototype and prototype settings.

Example:

python tools/diagnostics/compare_margin_significance.py \
  --dataset_root /path/to/RSTPReid \
  --dataset_name RSTPReid \
  --split test \
  --base_checkpoint results/no_proto/best.pth \
  --proto_checkpoint results/proto/best.pth \
  --out_dir results/margin_significance \
  --n_boot 10000 \
  --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


THRESHOLDS = (0.0, 0.01, 0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Positive--Hard Negative Score Margin diagnostics between "
            "baseline/no-prototype and prototype TBPS settings."
        )
    )
    parser.add_argument("--dataset_root", required=True, help="Dataset folder or parent folder containing the dataset.")
    parser.add_argument("--dataset_name", required=True, help="Dataset name, e.g. RSTPReid, CUHK-PEDES, ICFG-PEDES, PAB.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--base_checkpoint", required=True, help="No-prototype model checkpoint.")
    parser.add_argument("--proto_checkpoint", required=True, help="With-prototype model checkpoint.")
    parser.add_argument("--out_dir", required=True, help="Directory for report, CSVs, and plots.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for feature extraction.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--device", default="cuda", help='Device, e.g. "cuda" or "cpu".')
    parser.add_argument("--max_queries", type=int, default=None, help="Optional query limit for debugging.")
    parser.add_argument("--n_boot", type=int, default=10000, help="Number of paired bootstrap samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def require_runtime_dependencies() -> Tuple[Any, Any, Any]:
    try:
        import numpy as np
        import pandas as pd
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "This script requires pandas, numpy, and matplotlib. "
            "Install the project analysis dependencies before running it."
        ) from exc
    return np, pd, plt


def maybe_import_scipy() -> Optional[Any]:
    try:
        from scipy import stats
    except ImportError:
        return None
    return stats


def load_margin_diagnostic_runtime() -> Any:
    try:
        from scripts import plot_margin_diagnostic as margin_diag
    except ImportError as exc:
        raise RuntimeError(
            "Could not import scripts.plot_margin_diagnostic. Run this script from the repository root "
            "with the project dependencies installed."
        ) from exc
    return margin_diag


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def fmt_margin(value: float) -> str:
    return f"{value:.6f}"


def fmt_percent(ratio: float) -> str:
    return f"{ratio * 100.0:.2f}%"


def threshold_label(threshold: float) -> str:
    return "0" if threshold == 0 else f"{threshold:.2f}"


def make_margin_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_root=str(resolve_path(args.dataset_root)),
        checkpoint="",
        dataset_name=args.dataset_name,
        split=args.split,
        output_dir=str(resolve_path(args.out_dir)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        config=None,
        max_queries=args.max_queries,
        bins=35,
        xmin=-0.25,
        xmax=0.65,
    )


def rows_to_margin_dataframe(rows: Sequence[Mapping[str, Any]], pd: Any) -> Any:
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No usable margin rows were produced.")
    required = {"query_index", "margin"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Internal margin rows are missing required columns: {sorted(missing)}")
    return pd.DataFrame(
        {
            "query_id": df["query_index"].astype(str),
            "margin": df["margin"].astype(float),
        }
    )


def compute_checkpoint_margins(
    label: str,
    checkpoint_path: Path,
    args: argparse.Namespace,
    split_data: Any,
    device: Any,
    margin_diag: Any,
) -> Tuple[List[Dict[str, Any]], int, Mapping[str, int]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint_path}")

    model_cli_args = make_margin_args(args)
    model_args = margin_diag.build_model_args(model_cli_args)
    num_classes = max(int(split_data.num_train_ids), 1)

    print(f"\n[{label}] Building model and loading checkpoint: {checkpoint_path}")
    model = margin_diag.build_model(model_args, num_classes=num_classes)
    load_stats = margin_diag.load_checkpoint_for_inference(model, checkpoint_path)
    print(
        f"[{label}] Loaded checkpoint tensors: "
        f"{load_stats['loaded']} loaded, {load_stats['skipped_missing']} missing, "
        f"{load_stats['skipped_shape']} shape-mismatch, {load_stats['skipped_non_tensor']} non-tensor skipped."
    )

    model.to(device)
    if device.type == "cpu":
        model.float()
    model.eval()

    text_features, query_pids = margin_diag.extract_text_features(
        model,
        split_data,
        text_length=int(model_args.text_length),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    image_features, gallery_pids = margin_diag.extract_image_features(
        model,
        split_data,
        img_size=margin_diag.parse_img_size(model_args.img_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    sim = text_features @ image_features.t()
    rows, skipped = margin_diag.compute_margin_rows(sim, query_pids, gallery_pids)
    if skipped:
        print(
            f"[{label}] Warning: skipped {skipped} queries because they had no positive gallery image "
            "or no negative gallery image in the selected split."
        )

    del model, text_features, image_features, query_pids, gallery_pids, sim
    if device.type == "cuda":
        margin_diag.torch.cuda.empty_cache()

    return rows, skipped, load_stats


def read_margin_csv(path: Path, label: str, pd: Any) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"{label} CSV not found: {path}")

    df = pd.read_csv(path, dtype={"query_id": "string"})
    if "margin" not in df.columns:
        raise ValueError(f"{label} CSV must contain a 'margin' column: {path}")

    try:
        df["margin"] = pd.to_numeric(df["margin"], errors="raise")
    except Exception as exc:
        raise ValueError(f"{label} CSV has non-numeric values in the 'margin' column: {path}") from exc

    if df["margin"].isna().any():
        count = int(df["margin"].isna().sum())
        raise ValueError(f"{label} CSV contains {count} NaN margin values: {path}")

    if "query_id" in df.columns:
        if df["query_id"].isna().any():
            count = int(df["query_id"].isna().sum())
            raise ValueError(f"{label} CSV contains {count} missing query_id values: {path}")
        duplicates = df["query_id"].duplicated(keep=False)
        if duplicates.any():
            examples = df.loc[duplicates, "query_id"].astype(str).head(5).tolist()
            raise ValueError(f"{label} CSV contains duplicate query_id values, e.g. {examples}: {path}")

    return df


def ensure_same_query_ids(base_df: Any, proto_df: Any) -> None:
    base_ids = set(base_df["query_id"].astype(str).tolist())
    proto_ids = set(proto_df["query_id"].astype(str).tolist())
    if base_ids == proto_ids:
        return

    base_only = sorted(base_ids - proto_ids)[:5]
    proto_only = sorted(proto_ids - base_ids)[:5]
    raise ValueError(
        "Both CSVs contain query_id, but their query sets differ. "
        f"base-only examples={base_only}; proto-only examples={proto_only}."
    )


def make_paired_dataframe(base_df: Any, proto_df: Any, pd: Any) -> Tuple[Any, str]:
    base_has_query_id = "query_id" in base_df.columns
    proto_has_query_id = "query_id" in proto_df.columns

    if base_has_query_id and proto_has_query_id:
        ensure_same_query_ids(base_df, proto_df)
        base_pair = base_df[["query_id", "margin"]].rename(columns={"margin": "base_margin"})
        base_pair = base_pair.assign(_base_order=range(len(base_pair)))
        proto_pair = proto_df[["query_id", "margin"]].rename(columns={"margin": "proto_margin"})
        paired = base_pair.merge(proto_pair, on="query_id", how="inner", validate="one_to_one")
        paired = paired.sort_values("_base_order").drop(columns=["_base_order"]).reset_index(drop=True)
        pairing_note = "Paired by query_id."
    else:
        if len(base_df) != len(proto_df):
            raise ValueError(
                "CSV row counts differ and both files do not contain query_id. "
                f"base rows={len(base_df)}, proto rows={len(proto_df)}."
            )
        paired = pd.DataFrame(
            {
                "base_margin": base_df["margin"].to_numpy(),
                "proto_margin": proto_df["margin"].to_numpy(),
            }
        )
        if base_has_query_id:
            paired.insert(0, "query_id", base_df["query_id"].astype(str).to_numpy())
            pairing_note = "Paired by row order; query_id came only from the baseline margin table."
        elif proto_has_query_id:
            paired.insert(0, "query_id", proto_df["query_id"].astype(str).to_numpy())
            pairing_note = "Paired by row order; query_id came only from the prototype margin table."
        else:
            pairing_note = "Paired by row order because no query_id column was available."

    paired["diff"] = paired["proto_margin"] - paired["base_margin"]
    if paired[["base_margin", "proto_margin", "diff"]].isna().any().any():
        raise ValueError("Paired margin table contains NaN values after alignment.")
    return paired, pairing_note


def setting_metrics(values: Any, np: Any) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "num_used_queries": int(values.size),
        "mean_margin": float(np.mean(values)),
        "median_margin": float(np.median(values)),
        "negative_margin_ratio": float(np.mean(values < 0.0)),
        "margin_below_0_01_ratio": float(np.mean(values < 0.01)),
        "margin_below_0_05_ratio": float(np.mean(values < 0.05)),
    }


def paired_metrics(diff: Any, np: Any) -> Dict[str, float]:
    diff = np.asarray(diff, dtype=float)
    return {
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
        "win_ratio": float(np.mean(diff > 0.0)),
        "tie_ratio": float(np.mean(diff == 0.0)),
        "loss_ratio": float(np.mean(diff < 0.0)),
    }


def bootstrap_mean_ci(diff: Any, n_boot: int, seed: int, np: Any) -> Tuple[float, float]:
    if n_boot <= 0:
        raise ValueError("--n_boot must be a positive integer.")

    diff = np.asarray(diff, dtype=float)
    n = diff.size
    if n == 0:
        raise ValueError("No paired queries are available for bootstrap.")

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    chunk_size = max(1, min(1024, 5_000_000 // max(n, 1)))
    write_at = 0

    while write_at < n_boot:
        current = min(chunk_size, n_boot - write_at)
        indices = rng.integers(0, n, size=(current, n))
        boot_means[write_at:write_at + current] = diff[indices].mean(axis=1)
        write_at += current

    low, high = np.percentile(boot_means, [2.5, 97.5])
    return float(low), float(high)


def wilcoxon_test(diff: Any, stats: Optional[Any], np: Any) -> Dict[str, Any]:
    if stats is None:
        return {"available": False, "statistic": None, "pvalue": None, "note": "scipy unavailable"}
    if np.all(diff == 0.0):
        return {"available": False, "statistic": None, "pvalue": None, "note": "all differences are zero"}
    try:
        result = stats.wilcoxon(diff)
    except Exception as exc:
        return {"available": False, "statistic": None, "pvalue": None, "note": str(exc)}
    return {
        "available": True,
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "note": "",
    }


def paired_t_test(diff: Any, stats: Optional[Any]) -> Dict[str, Any]:
    if stats is None:
        return {"available": False, "statistic": None, "pvalue": None, "note": "scipy unavailable"}
    try:
        result = stats.ttest_1samp(diff, popmean=0.0)
    except Exception as exc:
        return {"available": False, "statistic": None, "pvalue": None, "note": str(exc)}
    return {
        "available": True,
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "note": "Reference test; bootstrap/Wilcoxon are more robust when margins are skewed.",
    }


def binomial_test(improved: int, worsened: int, stats: Optional[Any]) -> Dict[str, Any]:
    discordant = improved + worsened
    if stats is None:
        return {"available": False, "pvalue": None, "note": "scipy unavailable"}
    if discordant == 0:
        return {"available": False, "pvalue": None, "note": "no discordant threshold transitions"}

    try:
        if hasattr(stats, "binomtest"):
            result = stats.binomtest(improved, discordant, p=0.5, alternative="two-sided")
            pvalue = result.pvalue
        else:
            pvalue = stats.binom_test(improved, discordant, p=0.5, alternative="two-sided")
    except Exception as exc:
        return {"available": False, "pvalue": None, "note": str(exc)}

    return {"available": True, "pvalue": float(pvalue), "note": "exact binomial/McNemar-style test"}


def threshold_comparisons(base: Any, proto: Any, thresholds: Sequence[float], stats: Optional[Any], np: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n = len(base)
    for threshold in thresholds:
        base_bad = base < threshold
        proto_bad = proto < threshold

        improved_count = int(np.sum(base_bad & ~proto_bad))
        worsened_count = int(np.sum(~base_bad & proto_bad))
        unchanged_bad = int(np.sum(base_bad & proto_bad))
        unchanged_good = int(np.sum(~base_bad & ~proto_bad))
        net_improvement = improved_count - worsened_count
        test = binomial_test(improved_count, worsened_count, stats)

        rows.append(
            {
                "threshold": float(threshold),
                "improved_count": improved_count,
                "worsened_count": worsened_count,
                "unchanged_bad": unchanged_bad,
                "unchanged_good": unchanged_good,
                "net_improvement": net_improvement,
                "net_improvement_ratio": float(net_improvement / n),
                "pvalue": test["pvalue"],
                "test_available": test["available"],
                "test_note": test["note"],
            }
        )
    return rows


def format_setting_block(name: str, metrics: Mapping[str, float]) -> List[str]:
    return [
        f"{name}:",
        f"  used queries: {int(metrics['num_used_queries'])}",
        f"  mean margin: {fmt_margin(metrics['mean_margin'])}",
        f"  median margin: {fmt_margin(metrics['median_margin'])}",
        f"  negative margin ratio: {fmt_percent(metrics['negative_margin_ratio'])}",
        f"  margin below 0.01 ratio: {fmt_percent(metrics['margin_below_0_01_ratio'])}",
        f"  margin below 0.05 ratio: {fmt_percent(metrics['margin_below_0_05_ratio'])}",
    ]


def interpretation(
    paired: Mapping[str, float],
    ci_low: float,
    ci_high: float,
    wilcoxon: Mapping[str, Any],
    threshold_rows: Sequence[Mapping[str, Any]],
) -> List[str]:
    lines: List[str] = []
    wilcoxon_sig = bool(wilcoxon.get("available")) and float(wilcoxon["pvalue"]) < 0.05

    if ci_low > 0.0 and wilcoxon_sig:
        lines.append("The prototype variant yields a statistically significant positive shift in per-query margins.")
    elif paired["mean_diff"] > 0.0 and ci_low <= 0.0 <= ci_high:
        lines.append(
            "The prototype variant shows a numerical improvement, but the evidence is not sufficient to claim statistical significance."
        )
    elif paired["mean_diff"] <= 0.0:
        lines.append("The prototype variant does not show a positive mean margin shift in this paired comparison.")
    else:
        lines.append(
            "The prototype variant shows a positive mean margin shift, but the robust significance criteria are not both satisfied."
        )

    max_net = max(float(row["net_improvement_ratio"]) for row in threshold_rows)
    if 0.0 < max_net < 0.02:
        lines.append(
            "The reduction in negative or near-boundary margins is directionally positive but small in effect size."
        )
    return lines


def build_report(
    dataset_root: Path,
    dataset_name: str,
    split: str,
    base_checkpoint: Path,
    proto_checkpoint: Path,
    out_dir: Path,
    pairing_note: str,
    base_metrics: Mapping[str, float],
    proto_metrics: Mapping[str, float],
    paired: Mapping[str, float],
    ci_low: float,
    ci_high: float,
    wilcoxon: Mapping[str, Any],
    ttest: Mapping[str, Any],
    threshold_rows: Sequence[Mapping[str, Any]],
    interpretation_lines: Sequence[str],
) -> str:
    lines: List[str] = []
    lines.append("Margin Significance Report")
    lines.append("=" * 28)
    lines.append(f"dataset_root: {dataset_root}")
    lines.append(f"dataset_name: {dataset_name}")
    lines.append(f"split: {split}")
    lines.append(f"base_checkpoint: {base_checkpoint}")
    lines.append(f"proto_checkpoint: {proto_checkpoint}")
    lines.append(f"out_dir: {out_dir}")
    lines.append(f"pairing: {pairing_note}")
    lines.append("")
    lines.extend(format_setting_block("Baseline / no prototype", base_metrics))
    lines.append("")
    lines.extend(format_setting_block("With prototype", proto_metrics))
    lines.append("")
    lines.append("Paired improvement (prototype - baseline):")
    lines.append(f"  mean diff: {fmt_margin(paired['mean_diff'])}")
    lines.append(f"  median diff: {fmt_margin(paired['median_diff'])}")
    lines.append(f"  win ratio: {fmt_percent(paired['win_ratio'])}")
    lines.append(f"  tie ratio: {fmt_percent(paired['tie_ratio'])}")
    lines.append(f"  loss ratio: {fmt_percent(paired['loss_ratio'])}")
    lines.append("")
    lines.append("Statistical tests:")
    lines.append(f"  paired bootstrap 95% CI for mean diff: [{fmt_margin(ci_low)}, {fmt_margin(ci_high)}]")
    if ci_low > 0.0:
        lines.append("  bootstrap conclusion: significant positive mean margin improvement (CI entirely above 0)")
    else:
        lines.append("  bootstrap conclusion: CI is not entirely above 0")

    if wilcoxon["available"]:
        lines.append(
            f"  Wilcoxon signed-rank: statistic={fmt_margin(wilcoxon['statistic'])}, "
            f"p={wilcoxon['pvalue']:.6g}"
        )
    else:
        lines.append(f"  Wilcoxon signed-rank: unavailable ({wilcoxon['note']})")

    if ttest["available"]:
        lines.append(
            f"  paired t-test: statistic={fmt_margin(ttest['statistic'])}, "
            f"p={ttest['pvalue']:.6g}"
        )
        lines.append(f"  paired t-test note: {ttest['note']}")
    else:
        lines.append(f"  paired t-test: unavailable ({ttest['note']})")

    lines.append("")
    lines.append("Threshold transition counts:")
    for row in threshold_rows:
        lines.append(
            f"  threshold {threshold_label(row['threshold'])}: "
            f"improved={row['improved_count']}, worsened={row['worsened_count']}, "
            f"unchanged_bad={row['unchanged_bad']}, unchanged_good={row['unchanged_good']}, "
            f"net={row['net_improvement']} ({fmt_percent(row['net_improvement_ratio'])})"
        )
        if row["test_available"]:
            lines.append(f"    exact binomial/McNemar-style p={row['pvalue']:.6g}")
        else:
            lines.append(f"    exact test unavailable ({row['test_note']})")

    lines.append("")
    lines.append("Interpretation:")
    for item in interpretation_lines:
        lines.append(f"  {item}")
    lines.append("")
    return "\n".join(lines)


def add_summary_row(rows: List[Dict[str, Any]], section: str, metric: str, value: Any, formatted: str = "") -> None:
    rows.append({"section": section, "metric": metric, "value": value, "formatted": formatted})


def save_summary_csv(
    path: Path,
    base_metrics: Mapping[str, float],
    proto_metrics: Mapping[str, float],
    paired: Mapping[str, float],
    ci_low: float,
    ci_high: float,
    wilcoxon: Mapping[str, Any],
    ttest: Mapping[str, Any],
    threshold_rows: Sequence[Mapping[str, Any]],
    pd: Any,
) -> None:
    rows: List[Dict[str, Any]] = []

    for section, metrics in (("base", base_metrics), ("prototype", proto_metrics)):
        add_summary_row(rows, section, "num_used_queries", int(metrics["num_used_queries"]), str(int(metrics["num_used_queries"])))
        add_summary_row(rows, section, "mean_margin", metrics["mean_margin"], fmt_margin(metrics["mean_margin"]))
        add_summary_row(rows, section, "median_margin", metrics["median_margin"], fmt_margin(metrics["median_margin"]))
        add_summary_row(rows, section, "negative_margin_ratio", metrics["negative_margin_ratio"], fmt_percent(metrics["negative_margin_ratio"]))
        add_summary_row(rows, section, "margin_below_0_01_ratio", metrics["margin_below_0_01_ratio"], fmt_percent(metrics["margin_below_0_01_ratio"]))
        add_summary_row(rows, section, "margin_below_0_05_ratio", metrics["margin_below_0_05_ratio"], fmt_percent(metrics["margin_below_0_05_ratio"]))

    for metric, value in paired.items():
        formatted = fmt_percent(value) if metric.endswith("_ratio") else fmt_margin(value)
        add_summary_row(rows, "paired", metric, value, formatted)

    add_summary_row(rows, "bootstrap", "ci_low_95", ci_low, fmt_margin(ci_low))
    add_summary_row(rows, "bootstrap", "ci_high_95", ci_high, fmt_margin(ci_high))
    add_summary_row(rows, "bootstrap", "ci_entirely_above_zero", ci_low > 0.0, str(ci_low > 0.0))

    add_summary_row(rows, "wilcoxon", "available", wilcoxon["available"], str(wilcoxon["available"]))
    add_summary_row(rows, "wilcoxon", "statistic", wilcoxon["statistic"], "" if wilcoxon["statistic"] is None else fmt_margin(wilcoxon["statistic"]))
    add_summary_row(rows, "wilcoxon", "pvalue", wilcoxon["pvalue"], "" if wilcoxon["pvalue"] is None else f"{wilcoxon['pvalue']:.6g}")
    add_summary_row(rows, "wilcoxon", "note", wilcoxon["note"], str(wilcoxon["note"]))

    add_summary_row(rows, "paired_t_test", "available", ttest["available"], str(ttest["available"]))
    add_summary_row(rows, "paired_t_test", "statistic", ttest["statistic"], "" if ttest["statistic"] is None else fmt_margin(ttest["statistic"]))
    add_summary_row(rows, "paired_t_test", "pvalue", ttest["pvalue"], "" if ttest["pvalue"] is None else f"{ttest['pvalue']:.6g}")
    add_summary_row(rows, "paired_t_test", "note", ttest["note"], str(ttest["note"]))

    for row in threshold_rows:
        section = f"threshold_{threshold_label(row['threshold'])}"
        for key, value in row.items():
            if key == "threshold":
                continue
            if key == "net_improvement_ratio":
                formatted = fmt_percent(value)
            elif key == "pvalue" and value is not None:
                formatted = f"{value:.6g}"
            else:
                formatted = "" if value is None else str(value)
            add_summary_row(rows, section, key, value, formatted)

    pd.DataFrame(rows).to_csv(path, index=False)


def plot_margin_distribution(base: Any, proto: Any, path: Path, plt: Any, np: Any) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    combined = np.concatenate([base, proto])
    bins = np.histogram_bin_edges(combined, bins=40)
    ax.hist(base, bins=bins, density=True, alpha=0.55, label="No prototype", color="#4C78A8")
    ax.hist(proto, bins=bins, density=True, alpha=0.55, label="With prototype", color="#F58518")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Positive--hard negative score margin")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_diff_histogram(diff: Any, path: Path, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(diff, bins=40, density=True, alpha=0.85, color="#54A24B", edgecolor="white", linewidth=0.4)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Per-query margin difference (prototype - baseline)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_threshold_counts(threshold_rows: Sequence[Mapping[str, Any]], path: Path, plt: Any, np: Any) -> None:
    labels = [threshold_label(row["threshold"]) for row in threshold_rows]
    improved = np.array([row["improved_count"] for row in threshold_rows], dtype=float)
    worsened = np.array([row["worsened_count"] for row in threshold_rows], dtype=float)
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(x - width / 2, improved, width, label="Improved", color="#4C78A8")
    ax.bar(x + width / 2, worsened, width, label="Worsened", color="#E45756")
    ax.set_xlabel("Margin threshold")
    ax.set_ylabel("Query count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    np, pd, plt = require_runtime_dependencies()
    stats = maybe_import_scipy()
    margin_diag = load_margin_diagnostic_runtime()

    dataset_root = resolve_path(args.dataset_root)
    base_checkpoint = resolve_path(args.base_checkpoint)
    proto_checkpoint = resolve_path(args.proto_checkpoint)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = margin_diag.resolve_device(args.device)
    split_data = margin_diag.load_split_data(args.dataset_name, dataset_root, args.split)
    split_data = margin_diag.limit_queries(split_data, args.max_queries)
    margin_diag.validate_split_data(split_data, args.split)

    print(
        f"[{args.split}] images={len(split_data.img_paths)} "
        f"queries={len(split_data.captions)} identities={len(set(split_data.image_pids))}"
    )

    base_rows, base_skipped, _base_load_stats = compute_checkpoint_margins(
        "base/no prototype",
        base_checkpoint,
        args,
        split_data,
        device,
        margin_diag,
    )
    proto_rows, proto_skipped, _proto_load_stats = compute_checkpoint_margins(
        "prototype",
        proto_checkpoint,
        args,
        split_data,
        device,
        margin_diag,
    )

    base_margin_csv_path = out_dir / "base_margin_host.csv"
    proto_margin_csv_path = out_dir / "proto_margin_host.csv"
    pd.DataFrame(base_rows).to_csv(base_margin_csv_path, index=False)
    pd.DataFrame(proto_rows).to_csv(proto_margin_csv_path, index=False)

    if base_skipped != proto_skipped:
        raise RuntimeError(
            "The two checkpoint runs skipped different numbers of queries. "
            f"base skipped={base_skipped}, proto skipped={proto_skipped}."
        )

    base_df = rows_to_margin_dataframe(base_rows, pd)
    proto_df = rows_to_margin_dataframe(proto_rows, pd)
    paired_df, pairing_note = make_paired_dataframe(base_df, proto_df, pd)

    base = paired_df["base_margin"].to_numpy(dtype=float)
    proto = paired_df["proto_margin"].to_numpy(dtype=float)
    diff = paired_df["diff"].to_numpy(dtype=float)

    base_metrics = setting_metrics(base, np)
    proto_metrics = setting_metrics(proto, np)
    paired = paired_metrics(diff, np)
    ci_low, ci_high = bootstrap_mean_ci(diff, args.n_boot, args.seed, np)
    wilcoxon = wilcoxon_test(diff, stats, np)
    ttest = paired_t_test(diff, stats)
    threshold_rows = threshold_comparisons(base, proto, THRESHOLDS, stats, np)
    interpretation_lines = interpretation(paired, ci_low, ci_high, wilcoxon, threshold_rows)

    paired_csv_path = out_dir / "paired_margin_diff.csv"
    summary_csv_path = out_dir / "margin_significance_summary.csv"
    report_path = out_dir / "margin_significance_report.txt"
    dist_plot_path = out_dir / "margin_distribution_base_vs_proto.png"
    diff_plot_path = out_dir / "paired_diff_histogram.png"
    transition_plot_path = out_dir / "threshold_transition_counts.png"

    columns = ["base_margin", "proto_margin", "diff"]
    if "query_id" in paired_df.columns:
        columns.insert(0, "query_id")
    paired_df[columns].to_csv(paired_csv_path, index=False)

    save_summary_csv(
        summary_csv_path,
        base_metrics,
        proto_metrics,
        paired,
        ci_low,
        ci_high,
        wilcoxon,
        ttest,
        threshold_rows,
        pd,
    )

    plot_margin_distribution(base, proto, dist_plot_path, plt, np)
    plot_diff_histogram(diff, diff_plot_path, plt)
    plot_threshold_counts(threshold_rows, transition_plot_path, plt, np)

    report = build_report(
        dataset_root,
        args.dataset_name,
        args.split,
        base_checkpoint,
        proto_checkpoint,
        out_dir,
        pairing_note,
        base_metrics,
        proto_metrics,
        paired,
        ci_low,
        ci_high,
        wilcoxon,
        ttest,
        threshold_rows,
        interpretation_lines,
    )
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print("Saved outputs:")
    print(f"  report: {report_path}")
    print(f"  base margin CSV: {base_margin_csv_path}")
    print(f"  prototype margin CSV: {proto_margin_csv_path}")
    print(f"  summary CSV: {summary_csv_path}")
    print(f"  paired diff CSV: {paired_csv_path}")
    print(f"  distribution plot: {dist_plot_path}")
    print(f"  diff histogram: {diff_plot_path}")
    print(f"  transition counts plot: {transition_plot_path}")


if __name__ == "__main__":
    main()
