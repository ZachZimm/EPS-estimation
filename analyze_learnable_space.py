from __future__ import annotations

import argparse
import json
import re
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import umap

from external_data import ExternalSeriesCache


TARGET_SURFACE_FEATURES = [
    "last_prior_eps",
    "baseline_trailing_mean",
    "prev_eps_yoy_change",
    "revenue_growth_yoy",
    "operating_margin",
    "vix_close",
]

TARGET_SURFACE_LABELS = {
    "last_prior_eps": "Last Prior EPS",
    "baseline_trailing_mean": "Trailing Mean EPS (4Q)",
    "prev_eps_yoy_change": "Prior EPS YoY Change",
    "revenue_growth_yoy": "Revenue Growth YoY",
    "operating_margin": "Operating Margin",
    "vix_close": "VIX Close",
}

REPORT_PLOT_FILES = [
    "pca_separability.png",
    "umap_separability.png",
    "pca_faceted_by_delta_bucket.png",
    "umap_faceted_by_delta_bucket.png",
    "pca_separability_company_specific_only.png",
    "umap_separability_company_specific_only.png",
    "pca_faceted_by_delta_bucket_company_specific_only.png",
    "umap_faceted_by_delta_bucket_company_specific_only.png",
    "target_surfaces_actual_eps.png",
    "target_surfaces_delta_last.png",
]

REPORT_TABLE_FILES = ["summary.json", "analysis_frame.csv", "embedding_coordinates.csv"]

REPORT_LABELS = {
    "pca_separability.png": "PCA separability",
    "umap_separability.png": "UMAP separability",
    "pca_faceted_by_delta_bucket.png": "PCA by delta bucket",
    "umap_faceted_by_delta_bucket.png": "UMAP by delta bucket",
    "pca_separability_company_specific_only.png": "PCA separability, company-specific only",
    "umap_separability_company_specific_only.png": "UMAP separability, company-specific only",
    "pca_faceted_by_delta_bucket_company_specific_only.png": "PCA by delta bucket, company-specific only",
    "umap_faceted_by_delta_bucket_company_specific_only.png": "UMAP by delta bucket, company-specific only",
    "target_surfaces_actual_eps.png": "Target surfaces vs actual EPS",
    "target_surfaces_delta_last.png": "Target surfaces vs EPS delta",
}


def _sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return slug or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize learnable structure in EPS feature-target space.")
    parser.add_argument("--dataset-dir", default="artifacts/phase_two_sector_dataset")
    parser.add_argument("--training-dir", default="artifacts/phase_two_sector_training")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output-dir", default="artifacts/learnable_space_analysis")
    parser.add_argument(
        "--render-index-only",
        action="store_true",
        help="Skip analysis and regenerate index.html from the existing report artifacts.",
    )
    return parser.parse_args()


def load_bundle(dataset_dir: str | Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    dataset_path = Path(dataset_dir)
    metadata = pd.read_csv(dataset_path / "event_metadata.csv")
    arrays = np.load(dataset_path / "dataset_arrays.npz")
    normalization = json.loads((dataset_path / "normalization.json").read_text())
    return metadata, arrays["sequences"].astype(np.float32), arrays["static"].astype(np.float32), normalization


def load_test_predictions(training_dir: str | Path) -> pd.DataFrame:
    path = Path(training_dir) / "test_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_feature_frame(
    metadata: pd.DataFrame,
    sequences: np.ndarray,
    static: np.ndarray,
    normalization: dict[str, Any],
    env_file: str | Path,
    predictions: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    static_columns = normalization["static_feature_columns"]
    sequence_columns = normalization["sequence_feature_columns"]

    static_df = pd.DataFrame(static, columns=static_columns)
    sequence_last = pd.DataFrame(sequences[:, -1, :], columns=[f"lastseq_{col}" for col in sequence_columns])
    feature_frame = pd.concat([metadata.reset_index(drop=True), static_df, sequence_last], axis=1)

    feature_frame["target_delta_last"] = (
        pd.to_numeric(feature_frame["target_basic_eps"], errors="coerce")
        - pd.to_numeric(feature_frame["last_prior_eps"], errors="coerce")
    )

    latest_revenue = pd.to_numeric(feature_frame.get("fund_income_totalrevenue_latest"), errors="coerce")
    delta4_revenue = pd.to_numeric(feature_frame.get("fund_income_totalrevenue_delta4"), errors="coerce")
    lag4_revenue = latest_revenue - delta4_revenue
    revenue_growth = delta4_revenue / lag4_revenue.abs().replace(0.0, np.nan)
    feature_frame["revenue_growth_yoy"] = revenue_growth.replace([np.inf, -np.inf], np.nan)

    feature_frame["operating_margin"] = pd.to_numeric(feature_frame.get("ratio_operating_margin"), errors="coerce")
    feature_frame["prev_eps_yoy_change"] = pd.to_numeric(feature_frame.get("prev_eps_yoy_change"), errors="coerce")
    feature_frame["baseline_trailing_mean"] = pd.to_numeric(feature_frame.get("baseline_trailing_mean"), errors="coerce")
    feature_frame["last_prior_eps"] = pd.to_numeric(feature_frame.get("last_prior_eps"), errors="coerce")

    if not predictions.empty:
        merge_columns = ["ticker", "target_published_date", "actual", "prediction"]
        available_columns = [column for column in merge_columns if column in predictions.columns]
        pred_frame = predictions[available_columns].copy()
        pred_frame = pred_frame.rename(columns={"actual": "prediction_actual", "prediction": "prediction_value"})
        feature_frame = feature_frame.merge(pred_frame, on=["ticker", "target_published_date"], how="left")
        feature_frame["abs_error"] = (
            pd.to_numeric(feature_frame["prediction_value"], errors="coerce")
            - pd.to_numeric(feature_frame["prediction_actual"], errors="coerce")
        ).abs()
    else:
        feature_frame["prediction_actual"] = np.nan
        feature_frame["prediction_value"] = np.nan
        feature_frame["abs_error"] = np.nan

    cache = ExternalSeriesCache(env_file=env_file)
    vix = cache.get_market_series("^VIX", refresh=False)[["Date", "Close"]].copy()
    vix = vix.rename(columns={"Date": "vix_date", "Close": "vix_close"})
    vix["vix_date"] = pd.to_datetime(vix["vix_date"], errors="coerce").dt.normalize()
    vix = vix.dropna(subset=["vix_date"]).sort_values("vix_date")
    feature_frame["last_observed_market_date"] = pd.to_datetime(feature_frame["last_observed_market_date"], errors="coerce")
    feature_frame["merge_market_date"] = feature_frame["last_observed_market_date"].dt.normalize()
    feature_frame = pd.merge_asof(
        feature_frame.sort_values("merge_market_date"),
        vix,
        left_on="merge_market_date",
        right_on="vix_date",
        direction="backward",
    ).sort_index()

    train_delta_std = feature_frame.loc[feature_frame["split"] == "train", "target_delta_last"].std()
    if not np.isfinite(train_delta_std) or train_delta_std <= 0:
        train_delta_std = float(feature_frame["target_delta_last"].std()) or 1.0
    delta = feature_frame["target_delta_last"]
    feature_frame["delta_bucket"] = pd.cut(
        delta,
        bins=[-np.inf, -1.0 * train_delta_std, -0.25 * train_delta_std, 0.25 * train_delta_std, 1.0 * train_delta_std, np.inf],
        labels=["sharp_drop", "mild_drop", "flat", "mild_rise", "sharp_rise"],
    ).astype(str)

    feature_frame.to_csv(output_dir / "analysis_frame.csv", index=False)
    return feature_frame


def get_embedding_columns(normalization: dict[str, Any], mode: str) -> list[str]:
    static_columns = list(normalization["static_feature_columns"])
    sequence_columns = [f"lastseq_{col}" for col in normalization["sequence_feature_columns"]]
    if mode == "full":
        return static_columns + sequence_columns
    if mode == "company_specific_only":
        sequence_columns = [column for column in sequence_columns if not column.startswith("lastseq_ctx_")]
        return static_columns + sequence_columns
    raise ValueError(f"Unsupported embedding mode: {mode}")


def build_embedding_matrix(feature_frame: pd.DataFrame, normalization: dict[str, Any], mode: str) -> np.ndarray:
    embed_columns = get_embedding_columns(normalization, mode)
    matrix_df = feature_frame[embed_columns].apply(pd.to_numeric, errors="coerce")
    matrix_df = matrix_df.loc[:, matrix_df.notna().any(axis=0)]
    matrix = matrix_df.to_numpy(dtype=np.float32)
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return pipeline.fit_transform(matrix)


def plot_embedding_figure(
    coords: np.ndarray,
    feature_frame: pd.DataFrame,
    output_path: Path,
    title_prefix: str,
    pca_explained: tuple[float, float] | None = None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

    bucket_palette = {
        "sharp_drop": "#b22222",
        "mild_drop": "#f28e2b",
        "flat": "#9c9c9c",
        "mild_rise": "#59a14f",
        "sharp_rise": "#1f77b4",
    }
    for bucket, color in bucket_palette.items():
        mask = feature_frame["delta_bucket"] == bucket
        axes[0].scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.55, c=color, label=bucket)
    axes[0].set_title(f"{title_prefix}: Target Buckets")
    axes[0].legend(markerscale=2, fontsize=8)

    sectors = feature_frame["sector_bucket"].fillna("UNKNOWN").astype(str)
    sector_codes = pd.Categorical(sectors)
    scatter = axes[1].scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.55, c=sector_codes.codes, cmap="tab20")
    axes[1].set_title(f"{title_prefix}: Sector Buckets")
    legend_handles, _ = scatter.legend_elements(num=min(len(sector_codes.categories), 12))
    axes[1].legend(legend_handles[: min(len(sector_codes.categories), 12)], sector_codes.categories[: min(len(sector_codes.categories), 12)], fontsize=7, loc="best")

    test_mask = feature_frame["abs_error"].notna().to_numpy()
    axes[2].scatter(coords[:, 0], coords[:, 1], s=6, alpha=0.08, c="#808080")
    if test_mask.any():
        err = feature_frame.loc[test_mask, "abs_error"].to_numpy(dtype=float)
        vmax = np.nanquantile(err, 0.95) if np.isfinite(err).any() else 1.0
        norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-6))
        err_scatter = axes[2].scatter(coords[test_mask, 0], coords[test_mask, 1], s=14, alpha=0.7, c=err, cmap="viridis", norm=norm)
        fig.colorbar(err_scatter, ax=axes[2], shrink=0.8, label="Test Abs Error")
    axes[2].set_title(f"{title_prefix}: Test Error Overlay")

    xlabel = "Component 1"
    ylabel = "Component 2"
    if pca_explained is not None:
        xlabel = f"PC1 ({pca_explained[0]:.1%})"
        ylabel = f"PC2 ({pca_explained[1]:.1%})"
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_surface_panel(ax: plt.Axes, x: pd.Series, y: pd.Series, x_label: str, y_label: str) -> None:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if frame.empty:
        ax.set_title(f"{x_label}\n(no data)")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        return
    ax.hexbin(frame["x"], frame["y"], gridsize=35, cmap="Blues", mincnt=1)
    try:
        frame["bin"] = pd.qcut(frame["x"], q=min(24, frame["x"].nunique()), duplicates="drop")
        summary = frame.groupby("bin", observed=False).agg(x_mid=("x", "median"), y_mid=("y", "median"), count=("y", "size")).reset_index(drop=True)
        summary = summary[summary["count"] >= 5]
        if not summary.empty:
            ax.plot(summary["x_mid"], summary["y_mid"], color="#d62728", linewidth=2.0)
    except ValueError:
        pass
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.15)


def plot_target_surfaces(feature_frame: pd.DataFrame, output_dir: Path) -> None:
    targets = {
        "target_basic_eps": "Actual EPS",
        "target_delta_last": "EPS Delta vs Last",
    }
    for target_column, y_label in targets.items():
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
        for ax, feature_name in zip(axes.flat, TARGET_SURFACE_FEATURES):
            _plot_surface_panel(ax, feature_frame[feature_name], feature_frame[target_column], TARGET_SURFACE_LABELS[feature_name], y_label)
            ax.set_title(TARGET_SURFACE_LABELS[feature_name])
        suffix = "actual_eps" if target_column == "target_basic_eps" else "delta_last"
        fig.savefig(output_dir / f"target_surfaces_{suffix}.png", dpi=180)
        plt.close(fig)


def plot_faceted_embedding_by_bucket(
    coords: np.ndarray,
    feature_frame: pd.DataFrame,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    order = ["sharp_drop", "mild_drop", "flat", "mild_rise", "sharp_rise"]
    palette = {
        "sharp_drop": "#b22222",
        "mild_drop": "#f28e2b",
        "flat": "#9c9c9c",
        "mild_rise": "#59a14f",
        "sharp_rise": "#1f77b4",
    }
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    axes_flat = axes.flat
    error_values = pd.to_numeric(feature_frame["abs_error"], errors="coerce")
    finite_errors = error_values[np.isfinite(error_values)]
    vmax = float(np.nanquantile(finite_errors, 0.95)) if len(finite_errors) else 1.0
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-6))

    for idx, bucket in enumerate(order):
        ax = axes_flat[idx]
        bucket_mask = feature_frame["delta_bucket"] == bucket
        bucket_count = int(bucket_mask.sum())
        bucket_share = bucket_count / max(len(feature_frame), 1)
        ax.scatter(coords[:, 0], coords[:, 1], s=6, alpha=0.08, c="#7a7a7a")
        bucket_error = pd.to_numeric(feature_frame.loc[bucket_mask, "abs_error"], errors="coerce")
        has_error = bucket_mask.to_numpy() & feature_frame["abs_error"].notna().to_numpy()
        no_error = bucket_mask.to_numpy() & ~feature_frame["abs_error"].notna().to_numpy()
        if no_error.any():
            ax.scatter(coords[no_error, 0], coords[no_error, 1], s=18, alpha=0.55, c=palette[bucket], edgecolors="none")
        if has_error.any():
            scatter = ax.scatter(
                coords[has_error, 0],
                coords[has_error, 1],
                s=18,
                alpha=0.8,
                c=error_values[has_error],
                cmap="viridis",
                norm=norm,
                edgecolors="none",
            )
        ax.set_title(f"{bucket}\nn={bucket_count} ({bucket_share:.1%})")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.12)

    summary_ax = axes_flat[5]
    summary_ax.axis("off")
    lines = ["Delta bucket coverage"]
    for bucket in order:
        count = int((feature_frame["delta_bucket"] == bucket).sum())
        share = count / max(len(feature_frame), 1)
        mean_error = pd.to_numeric(feature_frame.loc[feature_frame["delta_bucket"] == bucket, "abs_error"], errors="coerce").mean()
        mean_text = "n/a" if pd.isna(mean_error) else f"{mean_error:.3f}"
        lines.append(f"{bucket}: n={count}, share={share:.1%}, mean_abs_err={mean_text}")
    summary_ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
        ax=axes[:, :2],
        shrink=0.85,
        label="Test Abs Error",
    )
    colorbar.ax.tick_params(labelsize=9)
    fig.suptitle(title, fontsize=16)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_summary(feature_frame: pd.DataFrame, pca: PCA, output_dir: Path) -> None:
    summary = {
        "num_samples": int(len(feature_frame)),
        "split_counts": {key: int(val) for key, val in feature_frame["split"].value_counts().to_dict().items()},
        "sector_counts": {key: int(val) for key, val in feature_frame["sector_bucket"].value_counts().to_dict().items()},
        "delta_bucket_counts": {key: int(val) for key, val in feature_frame["delta_bucket"].value_counts().to_dict().items()},
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_[:2]],
        "feature_missingness": {
            feature: float(feature_frame[feature].isna().mean()) for feature in TARGET_SURFACE_FEATURES
        },
        "target_surface_correlations": {
            feature: {
                "corr_actual_eps": None if pd.isna(pd.to_numeric(feature_frame[feature], errors="coerce").corr(pd.to_numeric(feature_frame["target_basic_eps"], errors="coerce"))) else float(pd.to_numeric(feature_frame[feature], errors="coerce").corr(pd.to_numeric(feature_frame["target_basic_eps"], errors="coerce"))),
                "corr_delta_last": None if pd.isna(pd.to_numeric(feature_frame[feature], errors="coerce").corr(pd.to_numeric(feature_frame["target_delta_last"], errors="coerce"))) else float(pd.to_numeric(feature_frame[feature], errors="coerce").corr(pd.to_numeric(feature_frame["target_delta_last"], errors="coerce"))),
            }
            for feature in TARGET_SURFACE_FEATURES
        },
    }
    if feature_frame["abs_error"].notna().any():
        summary["test_error_by_bucket"] = {
            key: float(val)
            for key, val in feature_frame.dropna(subset=["abs_error"]).groupby("delta_bucket")["abs_error"].mean().to_dict().items()
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def _read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _format_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "n/a"


def _format_pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}%}"
    except (TypeError, ValueError):
        return "n/a"


def _format_float(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _label_for_plot(filename: str) -> str:
    return REPORT_LABELS.get(filename, Path(filename).stem.replace("_", " ").title())


def _label_for_sector(slug: str) -> str:
    return slug.replace("_", " ").title()


def _render_table_panel(title: str, rows: list[tuple[str, str]], compact: bool = False) -> str:
    if not rows:
        rows = [("Status", "Not available")]
    body = "\n".join(
        f"<tr><th scope=\"row\">{escape(label)}</th><td>{escape(value)}</td></tr>"
        for label, value in rows
    )
    compact_class = " panel--compact" if compact else ""
    return (
        f"<section class=\"panel{compact_class}\">\n"
        f"  <h3>{escape(title)}</h3>\n"
        f"  <table class=\"stats-table\"><tbody>\n{body}\n  </tbody></table>\n"
        f"</section>"
    )


def _render_file_panel(base_path: str, table_files: list[str]) -> str:
    items = "\n".join(
        f"    <li><a href=\"{escape(base_path + filename)}\">{escape(filename)}</a></li>"
        for filename in table_files
    )
    return (
        "<section class=\"panel panel--compact\">\n"
        "  <h3>Files</h3>\n"
        "  <ul class=\"link-list\">\n"
        f"{items}\n"
        "  </ul>\n"
        "</section>"
    )


def _render_figure_grid(base_path: str, plot_files: list[str]) -> str:
    cards: list[str] = []
    for filename in plot_files:
        href = f"{base_path}{filename}"
        label = _label_for_plot(filename)
        cards.append(
            "<figure class=\"figure-card\">"
            f"<a class=\"figure-image\" href=\"{escape(href)}\">"
            f"<img src=\"{escape(href)}\" alt=\"{escape(label)}\" loading=\"lazy\"></a>"
            f"<figcaption><a href=\"{escape(href)}\">{escape(label)}</a></figcaption>"
            "</figure>"
        )
    return "\n".join(cards)


def _build_overview_panels(summary: dict[str, Any]) -> str:
    split_counts = summary.get("split_counts", {})
    delta_counts = summary.get("delta_bucket_counts", {})
    sector_counts = summary.get("sector_counts", {})
    variance = summary.get("pca_explained_variance_ratio", [None, None])
    missingness = summary.get("feature_missingness", {})
    feature_corr = summary.get("target_surface_correlations", {})
    error_by_bucket = summary.get("test_error_by_bucket", {})

    missing_row = ("Worst missingness", "n/a")
    if missingness:
        feature_name, missing_value = max(missingness.items(), key=lambda item: item[1])
        missing_row = (feature_name.replace("_", " ").title(), _format_pct(missing_value))

    corr_rows: list[tuple[str, str]] = []
    for feature_name in TARGET_SURFACE_FEATURES:
        actual_corr = feature_corr.get(feature_name, {}).get("corr_actual_eps")
        corr_rows.append((TARGET_SURFACE_LABELS[feature_name], _format_float(actual_corr)))

    panels = [
        _render_file_panel("", REPORT_TABLE_FILES),
        _render_table_panel(
            "Dataset",
            [
                ("Samples", _format_int(summary.get("num_samples"))),
                ("Sector groups", _format_int(len(sector_counts))),
                ("PC1 variance", _format_pct(variance[0] if len(variance) > 0 else None)),
                ("PC2 variance", _format_pct(variance[1] if len(variance) > 1 else None)),
                missing_row,
            ],
        ),
        _render_table_panel(
            "Splits",
            [(str(key).title(), _format_int(value)) for key, value in split_counts.items()],
            compact=True,
        ),
        _render_table_panel(
            "Delta buckets",
            [(str(key).replace("_", " ").title(), _format_int(value)) for key, value in delta_counts.items()],
            compact=True,
        ),
        _render_table_panel(
            "Mean test error",
            [(str(key).replace("_", " ").title(), _format_float(value)) for key, value in error_by_bucket.items()],
            compact=True,
        ),
        _render_table_panel("Correlation to actual EPS", corr_rows),
    ]
    return "\n".join(panels)


def _build_sector_sections(sectors_root: Path) -> tuple[str, str]:
    sector_dirs = sorted(path for path in sectors_root.iterdir() if path.is_dir()) if sectors_root.exists() else []
    toc_items: list[str] = []
    sections: list[str] = []
    for sector_dir in sector_dirs:
        sector_slug = sector_dir.name
        sector_label = _label_for_sector(sector_slug)
        sector_summary = _read_summary(sector_dir / "summary.json")
        split_counts = sector_summary.get("split_counts", {})
        meta_parts = [f"{_format_int(sector_summary.get('num_samples'))} samples"]
        if split_counts:
            meta_parts.extend(f"{str(key).title()} {_format_int(value)}" for key, value in split_counts.items())
        meta_text = " | ".join(meta_parts)

        toc_items.append(f"<li><a href=\"#{escape(sector_slug)}\">{escape(sector_label)}</a></li>")
        sections.append(
            f"""
    <section class="section sector-section" id="{escape(sector_slug)}">
      <div class="section-header">
        <div>
          <h2>{escape(sector_label)}</h2>
          <p>{escape(meta_text)}</p>
        </div>
        <div class="section-files">
          <a href="sectors/{escape(sector_slug)}/summary.json">summary.json</a>
          <a href="sectors/{escape(sector_slug)}/analysis_frame.csv">analysis_frame.csv</a>
          <a href="sectors/{escape(sector_slug)}/embedding_coordinates.csv">embedding_coordinates.csv</a>
        </div>
      </div>
      <div class="figure-grid">
        {_render_figure_grid(f"sectors/{sector_slug}/", REPORT_PLOT_FILES)}
      </div>
    </section>""".strip()
        )

    toc_html = "\n".join(toc_items)
    sections_html = "\n\n".join(sections)
    return toc_html, sections_html


def build_report_index(output_dir: Path) -> None:
    summary = _read_summary(output_dir / "summary.json")
    sector_toc, sector_sections = _build_sector_sections(output_dir / "sectors")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Learnable Space Analysis</title>
  <style>
    :root {{
      --bg: #f6f3ed;
      --surface: #fffdf9;
      --surface-alt: #f1ece3;
      --text: #201d18;
      --muted: #5b564d;
      --line: #d6cec1;
      --accent: #5a4630;
      --accent-soft: #ece4d8;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "IBM Plex Sans", "Helvetica Neue", sans-serif;
      line-height: 1.45;
    }}
    a {{
      color: var(--accent);
      text-decoration: underline;
      text-underline-offset: 0.14em;
    }}
    a:hover {{
      color: #3d2f1f;
    }}
    img {{
      display: block;
      max-width: 100%;
    }}
    .page {{
      max-width: 1380px;
      margin: 0 auto;
      padding: 24px;
    }}
    .page-header {{
      display: grid;
      gap: 20px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--line);
    }}
    .page-header h1 {{
      margin: 0;
      font-size: 32px;
      line-height: 1.1;
    }}
    .page-header p {{
      margin: 8px 0 0;
      max-width: 760px;
      color: var(--muted);
    }}
    .top-links,
    .section-files {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px 18px;
    }}
    .toc {{
      display: grid;
      gap: 10px;
      padding: 14px 16px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .toc h2 {{
      margin: 0;
      font-size: 16px;
    }}
    .toc ul {{
      list-style: none;
      margin: 0;
      padding: 0;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 8px 16px;
    }}
    .section {{
      margin-top: 28px;
    }}
    .section h2 {{
      margin: 0 0 12px;
      font-size: 22px;
    }}
    .section-header {{
      display: grid;
      gap: 10px;
      margin-bottom: 16px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--line);
    }}
    .section-header h2 {{
      margin-bottom: 4px;
    }}
    .section-header p {{
      margin: 0;
      color: var(--muted);
    }}
    .overview-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 14px;
    }}
    .panel {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .panel--compact {{
      min-height: 0;
    }}
    .panel h3 {{
      margin: 0 0 10px;
      font-size: 15px;
    }}
    .link-list {{
      list-style: none;
      margin: 0;
      padding: 0;
      display: grid;
      gap: 8px;
    }}
    .stats-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    .stats-table th,
    .stats-table td {{
      padding: 7px 0;
      border-top: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    .stats-table tbody tr:first-child th,
    .stats-table tbody tr:first-child td {{
      border-top: 0;
      padding-top: 0;
    }}
    .stats-table th {{
      width: 56%;
      font-weight: 500;
      color: var(--muted);
    }}
    .figure-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
    }}
    .figure-card {{
      margin: 0;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .figure-image {{
      display: block;
      background: var(--surface-alt);
      border-bottom: 1px solid var(--line);
    }}
    .figure-card img {{
      width: 100%;
      height: auto;
    }}
    .figure-card figcaption {{
      padding: 10px 12px;
      font-size: 14px;
    }}
    .figure-card figcaption a {{
      text-decoration: none;
    }}
    .figure-card figcaption a:hover {{
      text-decoration: underline;
    }}
    @media (max-width: 720px) {{
      .page {{
        padding: 16px;
      }}
      .page-header h1 {{
        font-size: 28px;
      }}
      .toc ul {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header class="page-header">
      <div>
        <h1>Learnable Space Analysis</h1>
        <p>Embedding plots, bucketed views, and target-surface diagnostics for the full EPS dataset and each sector slice.</p>
      </div>
      <div class="top-links">
        <a href="summary.json">summary.json</a>
        <a href="analysis_frame.csv">analysis_frame.csv</a>
        <a href="embedding_coordinates.csv">embedding_coordinates.csv</a>
      </div>
      <nav class="toc" aria-label="Sector navigation">
        <h2>Sectors</h2>
        <ul>
          {sector_toc}
        </ul>
      </nav>
    </header>

    <section class="section">
      <h2>Overview</h2>
      <div class="overview-grid">
        {_build_overview_panels(summary)}
      </div>
    </section>

    <section class="section" id="global">
      <h2>Global figures</h2>
      <div class="figure-grid">
        {_render_figure_grid("", REPORT_PLOT_FILES)}
      </div>
    </section>

    <section class="section">
      <h2>Sector figures</h2>
      {sector_sections}
    </section>
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html)


def analyze_feature_frame(feature_frame: pd.DataFrame, normalization: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_frame.to_csv(output_dir / "analysis_frame.csv", index=False)

    coords_frames: list[pd.DataFrame] = []
    pca_for_summary: PCA | None = None
    for mode in ["full", "company_specific_only"]:
        embedding_matrix = build_embedding_matrix(feature_frame, normalization, mode)
        suffix = "" if mode == "full" else "_company_specific_only"
        title_suffix = "" if mode == "full" else " Company-Specific Only"

        pca = PCA(n_components=2, random_state=42)
        pca_coords = pca.fit_transform(embedding_matrix)
        plot_embedding_figure(
            pca_coords,
            feature_frame,
            output_dir / f"pca_separability{suffix}.png",
            title_prefix=f"PCA{title_suffix}",
            pca_explained=(pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]),
        )

        n_neighbors = min(40, max(2, len(feature_frame) - 1))
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.15, random_state=42)
        umap_coords = reducer.fit_transform(embedding_matrix)
        plot_embedding_figure(
            umap_coords,
            feature_frame,
            output_dir / f"umap_separability{suffix}.png",
            title_prefix=f"UMAP{title_suffix}",
        )

        plot_faceted_embedding_by_bucket(
            umap_coords,
            feature_frame,
            output_dir / f"umap_faceted_by_delta_bucket{suffix}.png",
            title=f"UMAP Facets by EPS Delta Bucket{title_suffix}",
            x_label="UMAP 1",
            y_label="UMAP 2",
        )
        plot_faceted_embedding_by_bucket(
            pca_coords,
            feature_frame,
            output_dir / f"pca_faceted_by_delta_bucket{suffix}.png",
            title=f"PCA Facets by EPS Delta Bucket{title_suffix}",
            x_label=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
            y_label=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
        )

        coords_frame = feature_frame[["ticker", "target_published_date", "split", "sector_bucket", "delta_bucket", "abs_error"]].copy()
        coords_frame["embedding_mode"] = mode
        coords_frame[["pca_x", "pca_y"]] = pca_coords
        coords_frame[["umap_x", "umap_y"]] = umap_coords
        coords_frames.append(coords_frame)
        if mode == "full":
            pca_for_summary = pca

    pd.concat(coords_frames, ignore_index=True).to_csv(output_dir / "embedding_coordinates.csv", index=False)
    plot_target_surfaces(feature_frame, output_dir)
    build_summary(feature_frame, pca_for_summary or PCA(n_components=2), output_dir)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.render_index_only:
        build_report_index(output_dir)
        print(json.dumps({"output_dir": str(output_dir), "index_path": str(output_dir / "index.html")}, indent=2))
        return

    metadata, sequences, static, normalization = load_bundle(args.dataset_dir)
    predictions = load_test_predictions(args.training_dir)
    feature_frame = build_feature_frame(metadata, sequences, static, normalization, args.env_file, predictions, output_dir)
    analyze_feature_frame(feature_frame, normalization, output_dir)

    sectors_root = output_dir / "sectors"
    sectors_root.mkdir(parents=True, exist_ok=True)
    sector_dirs: list[str] = []
    for sector_bucket in sorted(feature_frame["sector_bucket"].dropna().astype(str).unique().tolist()):
        sector_frame = feature_frame[feature_frame["sector_bucket"].astype(str) == sector_bucket].copy()
        if len(sector_frame) < 10:
            continue
        sector_dir = sectors_root / _sanitize_slug(sector_bucket)
        analyze_feature_frame(sector_frame, normalization, sector_dir)
        sector_dirs.append(str(sector_dir))

    build_report_index(output_dir)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "plots": REPORT_PLOT_FILES,
                "tables": REPORT_TABLE_FILES,
                "index_path": str(output_dir / "index.html"),
                "sector_output_root": str(sectors_root),
                "sector_dirs": sector_dirs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
