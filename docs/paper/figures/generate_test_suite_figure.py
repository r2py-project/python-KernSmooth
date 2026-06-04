import matplotlib.pyplot as plt
import numpy as np

# 1. Data Extracted from Manuscript Table 3
functions = ["bkde", "bkde2D", "bkfe", "dpih", "dpik", "dpill", "locpoly"]
functions.reverse()  # Reverse to match top-to-bottom reading order

pos = np.array([28, 28, 24, 32, 52, 38, 38])[::-1]
neg = np.array([9, 12, 9, 15, 23, 17, 11])[::-1]
edge = np.array([20, 23, 16, 24, 44, 28, 23])[::-1]
totals = pos + neg + edge

initial_pass = np.array([57, 63, 49, 71, 119, 21, 72])[::-1]
resolved = totals - initial_pass

# 2. Color-Blind Friendly Palette
c_pos = "#77AADD"  # Blue
c_neg = "#EE8866"  # Red/Orange
c_edge = "#44BB99"  # Green
c_resolved = "#BBCC33"  # Yellow-Green

# 3. Global Figure and Font Setup
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.size"] = 20  # Applies to ticks, labels, and text globally
plt.rcParams["axes.titlesize"] = 20  # Slightly larger for titles
plt.rcParams["legend.fontsize"] = 20  # Slightly smaller for the legend

# Increased figure height to accommodate legends at the bottom
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(18, 9), gridspec_kw={"width_ratios": [1.3, 1]}
)

# ==========================================
# Panel A: Test Suite Breakdown
# ==========================================
ax1.barh(functions, pos, color=c_pos, edgecolor="black", height=0.7, label="Positive")
ax1.barh(
    functions,
    neg,
    left=pos,
    color=c_neg,
    edgecolor="black",
    height=0.7,
    label="Negative",
)
ax1.barh(
    functions,
    edge,
    left=pos + neg,
    color=c_edge,
    edgecolor="black",
    height=0.7,
    label="Edge-case",
)

# Annotations for Panel A (Numbers only to prevent overlap)
for i, (p, n, e, t) in enumerate(zip(pos, neg, edge, totals)):
    ax1.text(p / 2, i, str(p), va="center", ha="center", color="black")
    ax1.text(p + n / 2, i, str(n), va="center", ha="center", color="black")
    ax1.text(p + n + e / 2, i, str(e), va="center", ha="center", color="black")
    # Total count at the end of the bar
    ax1.text(t + 2, i, str(t), va="center", ha="left", fontweight="bold")

ax1.set_title("Test Suite Composition", fontweight="bold", pad=15)
ax1.set_xlabel("Number of Tests")
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.set_xlim(0, max(totals) + 15)

# Move legend outside the plot, below the x-axis
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

# ==========================================
# Panel B: Initial vs. Final Pass Rate
# ==========================================
ax2.barh(
    functions,
    initial_pass,
    color=c_pos,
    edgecolor="black",
    height=0.7,
    label="Initial Pass",
)
ax2.barh(
    functions,
    resolved,
    left=initial_pass,
    color=c_resolved,
    edgecolor="black",
    height=0.7,
    label="Resolved in Phase 7",
)

# Annotations for Panel B
for i, (ip, r) in enumerate(zip(initial_pass, resolved)):
    if ip > 0:
        ax2.text(ip / 2, i, str(ip), va="center", ha="center", color="black")
    if r > 0:
        ax2.text(ip + r / 2, i, str(r), va="center", ha="center", color="black")

ax2.set_title("Initial vs. Final Pass Rate", fontweight="bold", pad=15)
ax2.set_xlabel("Number of Tests Passed")
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.set_xlim(0, max(totals) + 10)

# Move legend outside the plot, below the x-axis
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

# ==========================================
# Final Output
# ==========================================
# Use tight_layout or bbox_inches='tight' to ensure the external legends aren't cut off
plt.tight_layout()
plt.savefig("test_suite.pdf", format="pdf", dpi=300, bbox_inches="tight")
print("Saved test_suite.pdf successfully.")
