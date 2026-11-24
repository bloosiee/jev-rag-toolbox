"""Render README figures from audited RAG result JSON files."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
COLORS = {"fixed": "#64748b", "qwen": "#0f766e", "v1": "#24539a",
          "v2": "#713f9e", "v3": "#a44f18"}
POLICIES = {"95_95": "95/95 full", "95_80": "95/80 full", "80_80": "80/80 full",
            "compact_80_80": "80/80 compact"}
METRICS = [
    ("EM accuracy", "%", "em", "{:.2f}"),
    ("Answer F1", "%", "f1", "{:.2f}"),
    ("Total tokens", "millions / 400 questions", "tokens", "{:.2f}"),
    ("Qwen calls", "calls / 400 questions", "qwen", "{:,.0f}"),
    ("End-to-end", "s / question", "e2e", "{:.2f}"),
    ("Qwen time", "s / question", "qwen_time", "{:.2f}"),
    ("Jev time", "s / question", "jev_time", "{:.2f}"),
    ("Jev calls", "calls / 400 questions", "jev", "{:,.0f}"),
]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(row):
    if "questions" in row:
        n = row["questions"]
        assert row["e2e_n"] == n
        correct, f1 = row["EM_correct"], row["F1_sum"] / n
        counts = row["LLM_calls"], row.get("Jev_calls", 0)
        timings = row["e2e_seconds"], row["LLM_seconds"], row.get("Jev_seconds", 0)
    else:
        n = row["n"]
        correct, f1 = row["correct"], row["F1"]
        counts = row["qwen_calls"], row["jev_calls"]
        timings = row["end_to_end_seconds"], row["qwen_seconds"], row["jev_seconds"]
    if n != 400:
        raise ValueError("These README figures require the full 400-question split")
    return dict(em=100 * correct / n, f1=100 * f1,
                tokens=(row["input_tokens"] + row["output_tokens"]) / 1e6,
                qwen=counts[0], jev=counts[1],
                e2e=timings[0] / n, qwen_time=timings[1] / n, jev_time=timings[2] / n)


def draw(title, filename, rows, directory):
    labels = [label for label, _, _ in rows]
    colors = [COLORS[key] for _, key, _ in rows]
    stats = [normalize(row) for _, _, row in rows]
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.4 if len(rows) > 4 else 4.7), sharey=True)
    fig.patch.set_facecolor("white")
    for i, (ax, (label, unit, key, fmt)) in enumerate(zip(axes.flat, METRICS)):
        values = [s[key] for s in stats]
        ax.barh(range(len(labels)), values, height=.65, color=colors, zorder=3)
        bound = max(values) or 1
        ax.set_xlim(0, bound * 1.34)
        ax.set_title(label, loc="left", fontsize=11, color="#0f172a", weight="bold", pad=7)
        ax.set_xlabel(unit, fontsize=8, labelpad=4)
        ax.set_yticks(range(len(labels)), labels=labels, fontsize=9)
        ax.tick_params(axis="y", length=0, pad=6, labelleft=(i % 4 == 0))
        ax.tick_params(axis="x", length=0, labelsize=7, pad=3)
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=key in {"qwen", "jev"}))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e2e8f0", linewidth=.8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for y, v in enumerate(values):
            ax.text(v + bound * .022, y, fmt.format(v), va="center", fontsize=8, color="#0f172a")
    axes[0, 0].invert_yaxis()
    fig.text(.023, .94, title, size=16, weight="bold", color="#0f172a")
    fig.text(.023, .882, "Same 400 questions within each comparison", size=9, color="#475569")
    fig.subplots_adjust(left=.13, right=.98, top=.81, bottom=.115, wspace=.19, hspace=.62)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    fig.savefig(path, dpi=145, facecolor="white", metadata={"Software":"Jev-RAG Toolbox"})
    plt.close(fig)
    print(path)


def baseline(metrics):
    return [("No gate", "fixed", metrics["fixed"]), ("Qwen gate", "qwen", metrics["qwen"])]


def current_figures(directory):
    hot = read(RESULTS / "hotpotqa400.json")["metrics"]
    v1 = read(RESULTS / "hotpotqa400_gate_ablation.json")["summary"]
    v2 = read(RESULTS / "hotpotqa400_v2_ablation.json")["summary"]
    v2_main = read(RESULTS / "hotpotqa400_v2.json")["summary"]["95_95"]
    for version, main, ablations in (("v1", hot["jev95"], v1), ("v2", v2_main, v2)):
        rows = baseline(hot) + [(POLICIES["95_95"], version, main)]
        rows += [(POLICIES[key], version, ablations[key])
                 for key in ("95_80", "80_80", "compact_80_80")]
        draw(f"HotpotQA · Qwen3.5-4B · Jev {version} ablation",
             f"hotpotqa-qwen4b-{version}-ablation.png", rows, directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "figures")
    parser.add_argument("--extra", type=Path, action="append", default=[],
                        help="Additional verified figure specs (title, filename, rows)")
    args = parser.parse_args()
    current_figures(args.output)
    for path in [RESULTS / "jev_v3_figure_data.json", *args.extra]:
        for fig in read(path):
            filename = fig["filename"]
            if Path(filename).name != filename or not filename.endswith(".png"):
                raise ValueError("Expected a PNG filename")
            draw(fig["title"], filename, fig["rows"], args.output)


if __name__ == "__main__":
    main()
