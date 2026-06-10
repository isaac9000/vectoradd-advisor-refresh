"""
Compare openevolve vs advisor (no refresh) vs advisor-refresh runs.
Marks epoch refresh boundary with a vertical line.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv

# ── Advisor (no refresh) data ─────────────────────────────────────────────────
ADV_TSV = "/workspace/vectoradd-advisor/vectoradd/runs/20260608_210059_vectoradd_starting_point/results.tsv"
adv_iters, adv_times, adv_kinds = [], [], []
with open(ADV_TSV) as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        adv_iters.append(int(row["agent_iteration"]))
        adv_times.append(float(row["time_us"]))
        adv_kinds.append(row["status"])

# ── Advisor-refresh data (epoch 1 + epoch 2, stitched by agent_iteration) ─────
# Epoch 1: agent_iterations 0–15, Epoch 2: agent_iterations 15–25
# Epoch refresh happens after agent_iteration 15.
REFRESH_ITER = 15  # vertical marker position

refresh_iters, refresh_times, refresh_kinds = [], [], []

epoch1_rows = [
    (0,  60.27, "keep"),
    (1,  70.63, "discard"),
    (2,  32.91, "keep"),
    (3,  89.94, "discard"),
    (4,  35.28, "discard"),
    (5,  49.80, "discard"),
    (6,  32.53, "keep"),
    (7,  426.00,"discard"),
    (8,  34.96, "discard"),
    (9,  33.99, "discard"),
    (10, 0.00,  "crash"),
    (11, 47.98, "discard"),
    (12, 32.38, "keep"),
    (13, 32.16, "keep"),
    (14, 31.96, "keep"),
    (15, 32.25, "discard"),
]

epoch2_rows = [
    (15, 32.00, "keep"),
    (16, 54.25, "discard"),
    (17, 54.81, "discard"),
    (18, 73.30, "discard"),
    (19, 39.99, "discard"),
    (20, 35.46, "discard"),
    (21, 33.09, "discard"),
    (22, 33.08, "discard"),
    (23, 33.56, "discard"),
    (24, 33.11, "discard"),
    (25, 32.44, "discard"),
]

for it, t, k in epoch1_rows + epoch2_rows:
    refresh_iters.append(it)
    refresh_times.append(t)
    refresh_kinds.append(k)

# ── OpenEvolve data ───────────────────────────────────────────────────────────
oe_raw = [
    (0,  59.813), (1,  None), (2,  59.853), (3,  59.525), (4,  59.796),
    (5,  59.615), (6,  60.028), (7,  59.864), (8,  59.539), (9,  59.504),
    (10, 60.104), (11, 59.110), (12, 59.670), (13, 59.138), (14, 59.428),
    (15, 59.772), (16, 60.147), (17, 60.107), (18, 59.687), (19, 59.515),
    (20, 59.386), (21, 59.686), (22, 60.639), (23, 60.039), (24, 59.759),
]
oe_iters, oe_times, oe_kinds = [], [], []
best_so_far = float("inf")
for it, t in oe_raw:
    oe_iters.append(it)
    oe_times.append(t if t is not None else 0.0)
    if t is None:
        oe_kinds.append("crash")
    elif t < best_so_far:
        best_so_far = t
        oe_kinds.append("keep")
    else:
        oe_kinds.append("discard")

# ── Best-over-time step lines ─────────────────────────────────────────────────
def best_step(iters, times, kinds):
    bx, by = [], []
    best = float("inf")
    for it, t, k in sorted(zip(iters, times, kinds)):
        if k == "keep" and t > 0:
            best = t
        if best < float("inf"):
            bx.append(it)
            by.append(best)
    return bx, by

adv_bx, adv_by       = best_step(adv_iters,     adv_times,     adv_kinds)
ref_bx, ref_by       = best_step(refresh_iters, refresh_times, refresh_kinds)
oe_bx,  oe_by        = best_step(oe_iters,      oe_times,      oe_kinds)

adv_best = min(t for t, k in zip(adv_times, adv_kinds) if k == "keep")
ref_best = min(t for t, k in zip(refresh_times, refresh_kinds) if k == "keep" and t > 0)
oe_best  = min(oe_by) if oe_by else float("inf")

# ── Y-axis (negative latency, clip outliers) ──────────────────────────────────
CLIP_US = 300.0
all_valid = [t for t in adv_times + refresh_times + oe_times if 0 < t <= CLIP_US]
y_hi = -(min(all_valid) * 0.82)
y_lo = -(CLIP_US * 1.08)

def ny(t):
    return max(-t, y_lo) if t > 0 else y_lo

LLM_CALLS = 171  # advisor-refresh run total

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))
fig.subplots_adjust(top=0.75)

# OpenEvolve — blue
oe_kx = [it for it, k in zip(oe_iters, oe_kinds) if k == "keep"]
oe_ky = [ny(oe_times[i]) for i, k in enumerate(oe_kinds) if k == "keep"]
oe_dx = [it for it, k in zip(oe_iters, oe_kinds) if k == "discard"]
oe_dy = [ny(oe_times[i]) for i, k in enumerate(oe_kinds) if k == "discard"]
oe_cx = [it for it, k in zip(oe_iters, oe_kinds) if k == "crash"]
if oe_kx:
    ax.scatter(oe_kx, oe_ky, c="#3b82f6", s=70, zorder=5, edgecolors="white", linewidths=0.5, label="openevolve keep")
if oe_dx:
    ax.scatter(oe_dx, oe_dy, c="#93c5fd", s=40, zorder=4, edgecolors="white", linewidths=0.3, alpha=0.8, label="openevolve discard")
if oe_bx:
    ax.step(oe_bx, [-t for t in oe_by], where="post", color="#3b82f6", linewidth=2, label="openevolve best", zorder=6)

# Advisor (no refresh) — green
adv_kx = [it for it, k in zip(adv_iters, adv_kinds) if k == "keep"]
adv_ky = [ny(adv_times[i]) for i, k in enumerate(adv_kinds) if k == "keep"]
adv_dx = [it for it, k in zip(adv_iters, adv_kinds) if k == "discard"]
adv_dy = [ny(adv_times[i]) for i, k in enumerate(adv_kinds) if k == "discard"]
adv_cx = [it for it, k in zip(adv_iters, adv_kinds) if k == "crash"]
if adv_kx:
    ax.scatter(adv_kx, adv_ky, c="#22c55e", s=70, zorder=5, edgecolors="white", linewidths=0.5, label="advisor keep")
if adv_dx:
    ax.scatter(adv_dx, adv_dy, c="#ef4444", s=40, zorder=4, edgecolors="white", linewidths=0.3, alpha=0.7, label="advisor discard")
if adv_bx:
    ax.step(adv_bx, [-t for t in adv_by], where="post", color="#22c55e", linewidth=2, label="advisor best", zorder=6)

# Advisor-refresh — purple
ref_kx = [it for it, k in zip(refresh_iters, refresh_kinds) if k == "keep" and refresh_times[refresh_iters.index(it)] > 0]
ref_ky = [ny(refresh_times[i]) for i, k in enumerate(refresh_kinds) if k == "keep" and refresh_times[i] > 0]
ref_dx = [it for it, k in zip(refresh_iters, refresh_kinds) if k == "discard"]
ref_dy = [ny(refresh_times[i]) for i, k in enumerate(refresh_kinds) if k == "discard"]
ref_cx = [it for it, k in zip(refresh_iters, refresh_kinds) if k == "crash"]
if ref_kx:
    ax.scatter(ref_kx, ref_ky, c="#a855f7", s=70, zorder=5, edgecolors="white", linewidths=0.5, label="advisor-refresh keep")
if ref_dx:
    ax.scatter(ref_dx, ref_dy, c="#d8b4fe", s=40, zorder=4, edgecolors="white", linewidths=0.3, alpha=0.7, label="advisor-refresh discard")
if ref_bx:
    ax.step(ref_bx, [-t for t in ref_by], where="post", color="#a855f7", linewidth=2, label="advisor-refresh best", zorder=6)

# Crashes (all series)
all_cx = oe_cx + adv_cx + ref_cx
if all_cx:
    ax.scatter(all_cx, [y_lo] * len(all_cx), c="#fbbf24", s=40, zorder=3,
               marker="x", linewidths=1.5, label=f"crash ({len(all_cx)})", alpha=0.8)

# Epoch refresh marker
ax.axvline(x=REFRESH_ITER, color="#a855f7", linewidth=1.5, linestyle="--", alpha=0.7, zorder=2)
ax.annotate("← epoch refresh", xy=(REFRESH_ITER + 0.2, y_hi * 0.97),
            fontsize=9, color="#7c3aed", va="top")

ax.set_ylim(y_lo * 1.05, y_hi)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
ax.set_xlabel("Iteration #", fontsize=12)
ax.set_ylabel("Negative Latency (-μs)", fontsize=12)
ax.grid(True, alpha=0.3)

# Legend above the plot
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4,
          framealpha=0.9, fontsize=10, borderaxespad=0)

# Best-time records above the plot (figure-level text)
fig.text(0.5, 0.92,
         f"OpenEvolve best: {oe_best:.2f} μs    |    "
         f"Advisor best: {adv_best:.2f} μs    |    "
         f"Advisor-refresh best: {ref_best:.2f} μs",
         ha="center", va="top", fontsize=11, fontweight="bold", color="#1e3a5f",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#a855f7", alpha=0.9))

# Title
fig.text(0.5, 0.995, "openevolve vs advisor vs advisor-refresh — vectoradd",
         ha="center", va="top", fontsize=14, fontweight="bold")

# LLM call counter — bottom right (advisor-refresh only)
ax.annotate(
    f"advisor-refresh LLM calls: {LLM_CALLS}",
    xy=(0.99, 0.02), xycoords="axes fraction",
    ha="right", va="bottom", fontsize=10, color="#6b7280",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d1d5db", alpha=0.9),
)

# Outlier note — bottom left
ax.annotate(
    f"(outliers > {CLIP_US:.0f} μs shown at floor)",
    xy=(0.01, 0.02), xycoords="axes fraction",
    ha="left", va="bottom", fontsize=9, color="#6b7280",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d1d5db", alpha=0.8),
)
out = "/workspace/vectoradd-advisor-refresh/comparison.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved {out}")
