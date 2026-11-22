import json
import os
import sys
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Rapordaki görselleri gerçek veri ve sonuç dosyalarından üretir.
# Kullanım (proje kökünden): python report/make_figures.py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import normalize  # noqa: E402

OUT_DIR = "report/figures"
DATA_PATH = "data/absa_120.json"
RESULTS_PATH = "results/test/results_v4_reason.json"

INK = "#1f1f1d"
MUTED = "#6b6a66"
PANEL = "#f4f4f2"
BORDER = "#c9c8c3"
GOOD = "#1b7a3a"
BAD = "#b3261e"
MONO = "DejaVu Sans Mono"

os.makedirs(OUT_DIR, exist_ok=True)

with open(DATA_PATH, encoding="utf-8") as file:
    data = {record["id"]: record for record in json.load(file)}
with open(RESULTS_PATH, encoding="utf-8") as file:
    results = {record["id"]: record for record in json.load(file)}


def wrap(lines, width):
    # Uzun JSON satırlarını girintiyi koruyarak alt satıra kaydırır
    wrapped = []
    for line in lines:
        indent = " " * (len(line) - len(line.lstrip()) + 2)
        wrapped.extend(textwrap.wrap(line, width, subsequent_indent=indent) or [line])
    return wrapped


def panel(ax, x, y, width, height, title, lines):
    # Editör penceresine benzeyen kutu: başlık şeridi + tek aralıklı metin
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0,rounding_size=0.015",
                                facecolor=PANEL, edgecolor=BORDER, linewidth=1))
    ax.add_patch(FancyBboxPatch((x, y + height - 0.13), width, 0.13, boxstyle="round,pad=0,rounding_size=0.015",
                                facecolor="#e4e3df", edgecolor=BORDER, linewidth=1))
    ax.text(x + 0.015, y + height - 0.065, title, fontsize=7.5, color=INK, va="center", family="DejaVu Sans")
    for i, line in enumerate(lines):
        ax.text(x + 0.02, y + height - 0.2 - i * 0.085, line, fontsize=7, color=INK, va="center", family=MONO)


# Şekil 1: veri setinden bir kayıt ve nihai sistemin çıktısı
sample_id = 65
record = data[sample_id]
output = results[sample_id]

input_lines = wrap(json.dumps(record, indent=2, ensure_ascii=False).splitlines(), 40)
output_lines = wrap(output["raw_response"].strip().splitlines(), 42)

fig, ax = plt.subplots(figsize=(7.2, 1.75))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

panel(ax, 0.0, 0.02, 0.43, 0.96, f"Input: {DATA_PATH} (id {sample_id})", input_lines)
ax.annotate("", xy=(0.485, 0.5), xytext=(0.445, 0.5),
            arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1.4))
panel(ax, 0.5, 0.02, 0.5, 0.96, "Output: absa-model-v4-reason, schema (raw text)", output_lines)

fig.savefig(f"{OUT_DIR}/fig1_example.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)


# Şekil 2: eşleştirme kuralı örnekleri (gerçek test çıktıları)
cases = [65, 32, 26, 71, 14]

fig, ax = plt.subplots(figsize=(7.2, 1.95))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

columns = [(0.0, "id"), (0.05, "gold aspect"), (0.33, "predicted aspect"), (0.55, "why"),
           (0.84, "exact"), (0.93, "partial")]
header_y = 0.9
for x, name in columns:
    ax.text(x, header_y, name, fontsize=7.5, color=MUTED, va="center", family="DejaVu Sans", weight="bold")
ax.plot([0.0, 1.0], [header_y - 0.08, header_y - 0.08], color=BORDER, linewidth=1)

for row, case_id in enumerate(cases):
    y = header_y - 0.2 - row * 0.155
    gold = results[case_id]["gold_aspect"]
    predicted = results[case_id]["parsed_response"]["aspect"]
    gold_n, predicted_n = normalize(gold), normalize(predicted)

    exact = gold_n == predicted_n
    partial = predicted_n != "" and (predicted_n in gold_n or gold_n in predicted_n)

    if gold == predicted:
        why = "identical"
    elif exact:
        why = "equal after lower()"
    elif predicted_n in gold_n:
        why = "predicted is inside gold"
    elif gold_n in predicted_n:
        why = "gold is inside predicted"
    else:
        why = "no overlap"

    ax.text(0.0, y, str(case_id), fontsize=7.5, color=INK, va="center", family=MONO)
    ax.text(0.05, y, f'"{gold}"', fontsize=7.5, color=INK, va="center", family=MONO)
    ax.text(0.33, y, f'"{predicted}"', fontsize=7.5, color=INK, va="center", family=MONO)
    ax.text(0.55, y, why, fontsize=7.5, color=MUTED, va="center", family="DejaVu Sans")
    ax.text(0.85, y, "✓" if exact else "✗", fontsize=9, color=GOOD if exact else BAD, va="center", family=MONO)
    ax.text(0.95, y, "✓" if partial else "✗", fontsize=9, color=GOOD if partial else BAD, va="center", family=MONO)

fig.savefig(f"{OUT_DIR}/fig2_matching.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Saved {OUT_DIR}/fig1_example.png and {OUT_DIR}/fig2_matching.png")
