"""
Weather Prediction — Experiment Progress Visualization
Creates a clean chart showing the journey from baseline to best score.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Experiment data from PROGRESS.md and git history
experiments = [
    # (label, composite_score, status, category)
    ("RF baseline\n(200 trees)", 0.8695, "kept", "flat"),
    ("XGBoost\n(500 trees)", 0.8575, "reverted", "flat"),
    ("XGBoost\n(300 trees)", 0.8499, "reverted", "flat"),
    ("LightGBM\n(500 trees)", 0.8624, "reverted", "flat"),
    ("RF + threshold\ntuning", 0.8556, "reverted", "flat"),
    ("Prob avg\nRF+LGB+XGB", 0.8689, "reverted", "flat"),
    ("Per-horizon\nbest model", 0.8671, "reverted", "flat"),
    ("Cascade RF\n(200 trees)", 0.8729, "kept", "cascade"),
    ("Cascade +\nphysics feats", 0.8725, "reverted", "cascade"),
    ("Cascade +\nOOB threshold", 0.8697, "reverted", "cascade"),
    ("Cascade +\nadaptive thresh", 0.8689, "reverted", "cascade"),
    ("Cascade RF\n(300 trees)", 0.8731, "kept", "cascade"),
    ("Cascade RF\n(400 trees)", 0.8733, "kept", "cascade"),
    ("Cascade +\nLGB stage2", 0.8729, "reverted", "cascade"),
    ("Cascade\nadapt. depth", 0.8697, "reverted", "cascade"),
    ("ExtraTrees\n(500 trees)", 0.8571, "reverted", "cascade"),
    ("Cascade+Flat\nblend (BEST)", 0.8738, "kept", "blend"),
]

labels = [e[0] for e in experiments]
scores = [e[1] for e in experiments]
statuses = [e[2] for e in experiments]
categories = [e[3] for e in experiments]

# Colors
cat_colors = {"flat": "#5B9BD5", "cascade": "#ED7D31", "blend": "#70AD47"}
edge_colors = ["#2E7D32" if s == "kept" else "#B0B0B0" for s in statuses]
face_colors = [cat_colors[c] if s == "kept" else "#E0E0E0" for c, s in zip(categories, statuses)]

fig, ax = plt.subplots(figsize=(18, 7))

x = np.arange(len(experiments))
bars = ax.bar(x, scores, color=face_colors, edgecolor=edge_colors, linewidth=1.5, width=0.7)

# Running best line
running_best = []
best_so_far = 0
for s, st in zip(scores, statuses):
    if st == "kept" and s > best_so_far:
        best_so_far = s
    running_best.append(best_so_far)
ax.step(x, running_best, where='mid', color='#2E7D32', linewidth=2.5, linestyle='--',
        label='Running best', zorder=5)

# Annotations for key moments
ax.annotate('Cascade\nbreakthrough!',
            xy=(7, 0.8729), xytext=(7, 0.877),
            fontsize=9, fontweight='bold', color='#D84315',
            ha='center', va='bottom',
            arrowprops=dict(arrowstyle='->', color='#D84315', lw=1.5))

ax.annotate('Best: 0.8738',
            xy=(16, 0.8738), xytext=(14.5, 0.8775),
            fontsize=10, fontweight='bold', color='#2E7D32',
            ha='center', va='bottom',
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2))

# Phase separators
ax.axvline(x=6.5, color='#888', linestyle=':', linewidth=1, alpha=0.5)
ax.axvline(x=15.5, color='#888', linestyle=':', linewidth=1, alpha=0.5)
ax.text(3, 0.878, 'Phase 1: Flat Models', ha='center', fontsize=10, color='#5B9BD5', fontweight='bold')
ax.text(11, 0.878, 'Phase 2: Cascade Architecture', ha='center', fontsize=10, color='#ED7D31', fontweight='bold')
ax.text(16, 0.878, 'Phase 3:\nBlend', ha='center', fontsize=9, color='#70AD47', fontweight='bold')

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7, rotation=0, ha='center')
ax.set_ylabel('Composite Score (weighted F1)', fontsize=12)
ax.set_title('Precipitation Nowcasting — Experiment Journey\n'
             'Galápagos Weather Stations • 3h/6h/12h Forecast Horizons',
             fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0.845, 0.880)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.4f}'))
ax.grid(axis='y', alpha=0.3, linestyle='-')
ax.set_axisbelow(True)

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#5B9BD5', label='Flat classifier'),
    mpatches.Patch(facecolor='#ED7D31', label='Cascade (rain/no-rain → intensity)'),
    mpatches.Patch(facecolor='#70AD47', label='Cascade + Flat blend'),
    mpatches.Patch(facecolor='#E0E0E0', edgecolor='#B0B0B0', label='Reverted (worse)'),
    plt.Line2D([0], [0], color='#2E7D32', linewidth=2, linestyle='--', label='Running best'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)

# Key findings box
findings = (
    "Key Findings:\n"
    "• Cascade architecture = biggest breakthrough (+0.003 over flat RF)\n"
    "• Threshold tuning always hurts on temporal validation split\n"
    "• Probability blending > hard predictions\n"
    "• 12h horizon is the bottleneck (heavy_rain F1 ≈ 0)"
)
ax.text(0.02, 0.02, findings, transform=ax.transAxes, fontsize=8,
        verticalalignment='bottom', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4', alpha=0.9))

plt.tight_layout()
out_path = "weather_experiment_journey.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.close()
