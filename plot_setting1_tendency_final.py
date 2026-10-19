#!/usr/bin/env python3
"""Regenerate the Setting-1 tendency figure with final PSC results."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
TEMPLATE = (
    ROOT
    / "reproduce/code/code_2026年09月08日-22时40分18秒"
    / "experenment/visualise/plot_setting1_tendency_compact.py"
)


def main():
    spec = spec_from_file_location("setting1_tendency_template", TEMPLATE)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    module.SEEN_RESULTS["Ours"] = {
        "mAP": [83.7, 85.5, 79.6, 68.9, 68.3],
        "R1": [92.5, 90.2, 88.4, 82.9, 79.8],
    }
    module.UNSEEN_RESULTS["Ours"] = {
        "mAP": [55.9, 63.9, 67.3, 68.1, 74.6],
        "R1": [47.9, 56.1, 60.9, 61.2, 67.6],
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 6.8,
            "axes.linewidth": 0.55,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(3.55, 3.20), dpi=600)
    legend_handles = legend_labels = None
    for ax, panel in zip(axes.flat, module.PANELS):
        handles, labels = module.draw_panel(ax, *panel)
        if legend_handles is None:
            legend_handles, legend_labels = handles, labels

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        fontsize=7.1,
        columnspacing=0.95,
        handlelength=1.45,
        handletextpad=0.38,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(
        left=0.145, right=0.992, top=0.795, bottom=0.105,
        wspace=0.37, hspace=0.55,
    )

    output_dir = ROOT / "paper_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        output_path = output_dir / f"setting1_tendency_compact.{suffix}"
        fig.savefig(
            output_path, dpi=600, facecolor="white",
            bbox_inches="tight", pad_inches=0.02,
        )
        print(output_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
