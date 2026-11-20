import json
import statistics as st

import matplotlib.pyplot as plt
import numpy as np

# exp_prompt_length.py sonuçlarından grafik üretir
BLUE, ORANGE, INK, MUTED, GRID = "#2a6fdb", "#c2571a", "#1f2933", "#6b7785", "#e3e7ec"

records = json.load(open("results/prompt_length_experiment.json"))["records"]
ks = sorted({r["k"] for r in records})

tokens, cold_prefill, decode_per_token, warm_total, cold_total = [], [], [], [], []
for k in ks:
    rows = [r for r in records if r["k"] == k]
    cold, warm = rows[0], rows[1:]            # ilk istek: soğuk, diğerleri: önbellek sıcak
    tokens.append(st.median(r["prompt_eval_count"] for r in rows))
    cold_prefill.append(cold["prompt_eval_duration_ms"] / 1000)
    decode_per_token.append(st.median(r["eval_duration_ms"] / r["eval_count"] for r in warm))
    warm_total.append(st.median(r["total_duration_ms"] for r in warm) / 1000)
    cold_total.append(cold["total_duration_ms"] / 1000)

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def label_k(ax, xs, ys, color):
    for k, x, y in zip(ks, xs, ys):
        # k=0 ve k=1 noktaları yakın: biri altta, biri üstte etiketlenir
        offset = (26, -10) if k == 0 else (0, 7)
        ax.annotate(f"k={k}", (x, y), textcoords="offset points", xytext=offset, ha="center", fontsize=8, color=color)


# 1. Prefill (soğuk): prompt token sayısına karşı
ax = axes[0]
slope, intercept = np.polyfit(tokens, cold_prefill, 1)
quad = np.polyfit(tokens, cold_prefill, 2)
xs = np.linspace(min(tokens), max(tokens), 100)
ax.plot(xs, slope * xs + intercept, color=MUTED, linewidth=1, linestyle="--")
ax.plot(tokens, cold_prefill, color=BLUE, linewidth=2, marker="o", markersize=7)
label_k(ax, tokens, cold_prefill, MUTED)
style(ax, "Prefill (cold first request)", "Prefill time (s)")
speeds = [t / s for t, s in zip(tokens, cold_prefill)]
ax.text(0.04, 0.9, f"linear fit: {slope * 1000:.2f} ms/token\nspeed: {min(speeds):.0f}-{max(speeds):.0f} tokens/s (almost constant)",
        transform=ax.transAxes, fontsize=8.5, color=MUTED, va="top")

# 2. Decode: token başına süre
ax = axes[1]
ax.plot(tokens, decode_per_token, color=BLUE, linewidth=2, marker="o", markersize=7)
label_k(ax, tokens, decode_per_token, MUTED)
style(ax, "Decode (per generated token)", "ms per output token")
ax.set_ylim(0, max(decode_per_token) * 1.4)

# 3. Toplam süre: soğuk ve sıcak
ax = axes[2]
ax.plot(tokens, cold_total, color=BLUE, linewidth=2, marker="o", markersize=7)
ax.plot(tokens, warm_total, color=ORANGE, linewidth=2, marker="o", markersize=7)
ax.text(tokens[-1], cold_total[-1] + 0.35, "cold first request", color=BLUE, ha="right", fontsize=9)
ax.text(tokens[-1], warm_total[-1] + 0.35, "warm (cache hit), median", color=ORANGE, ha="right", fontsize=9)
style(ax, "Total time per request", "Total time (s)")
ax.set_ylim(0, max(cold_total) * 1.15)

fig.suptitle("Few-shot examples vs latency (qwen3:1.7b Q4_K_M, 30 test samples, k = 0, 1, 5, 10, 20, 40)", x=0.01, ha="left", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("results/prompt_length_experiment.png", dpi=150, facecolor="white")
print("saved results/prompt_length_experiment.png")
