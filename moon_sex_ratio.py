import pandas as pd
import ephem
import matplotlib.pyplot as plt
from datetime import timedelta

# ── 1. Load & clean ──────────────────────────────────────────────────────────

df = pd.read_csv("births.csv")
df = df.dropna(subset=["year", "month", "day"])
df = df[df["day"] != 99]

def safe_ts(r):
    try:
        return pd.Timestamp(int(r["year"]), int(r["month"]), int(r["day"]))
    except ValueError:
        return pd.NaT

df["date"] = df.apply(safe_ts, axis=1)
df = df.dropna(subset=["date"])

# ── 2. Conception date = birth - 38 weeks ────────────────────────────────────

df["conception_date"] = df["date"] - timedelta(weeks=38)

# ── 3. Moon phase at conception ──────────────────────────────────────────────

def moon_info(date):
    moon = ephem.Moon()
    d_str = date.strftime("%Y/%m/%d")
    moon.compute(d_str)
    illum_today = moon.phase

    # Waxing = illumination increasing day over day
    next_day = date + timedelta(days=1)
    moon.compute(next_day.strftime("%Y/%m/%d"))
    illum_next = moon.phase

    is_waxing = illum_next > illum_today

    # Assign one of 8 traditional phases
    if illum_today < 3:
        phase_name = "Nouvelle lune"
    elif illum_today < 50 and is_waxing:
        phase_name = "Croissant montant"
    elif illum_today < 55 and is_waxing:
        phase_name = "Premier quartier"
    elif illum_today < 98 and is_waxing:
        phase_name = "Gibbeuse montante"
    elif illum_today >= 98:
        phase_name = "Pleine lune"
    elif illum_today >= 50 and not is_waxing:
        phase_name = "Gibbeuse descendante"
    elif illum_today >= 45 and not is_waxing:
        phase_name = "Dernier quartier"
    else:
        phase_name = "Croissant descendant"

    return illum_today, is_waxing, phase_name

results = df["conception_date"].apply(lambda d: pd.Series(moon_info(d), index=["moon_phase", "is_waxing", "phase_name"]))
df = pd.concat([df, results], axis=1)

# ── 4. Sex ratio helper ───────────────────────────────────────────────────────

def sex_ratio(subset):
    m = subset.loc[subset["gender"] == "M", "births"].sum()
    f = subset.loc[subset["gender"] == "F", "births"].sum()
    return (m / f) * 100 if f > 0 else 0.0

# ── 5. Waxing vs Waning ───────────────────────────────────────────────────────

ratio_waxing = sex_ratio(df[df["is_waxing"]])
ratio_waning = sex_ratio(df[~df["is_waxing"]])

print("=" * 60)
print("  LUNE MONTANTE / DESCENDANTE — ANALYSE DU SEXE DU BÉBÉ")
print("=" * 60)
print(f"  Ratio lune MONTANTE : {ratio_waxing:.4f} garçons / 100 filles")
print(f"  Ratio lune DESCENDANTE : {ratio_waning:.4f} garçons / 100 filles")
diff = ratio_waxing - ratio_waning
print(f"  Différence : {diff:+.4f}")
print()

# ── 6. Ratio par phase ────────────────────────────────────────────────────────

PHASE_ORDER = [
    "Nouvelle lune",
    "Croissant montant",
    "Premier quartier",
    "Gibbeuse montante",
    "Pleine lune",
    "Gibbeuse descendante",
    "Dernier quartier",
    "Croissant descendant",
]

print("  Ratio par phase lunaire :")
phase_ratios = {}
for phase in PHASE_ORDER:
    subset = df[df["phase_name"] == phase]
    if len(subset) > 0:
        r = sex_ratio(subset)
        phase_ratios[phase] = r
        print(f"    {phase:<25} {r:.4f}")

print()
print("  CONCLUSION : " + (
    "Différences négligeables. Aucun effet détectable de la phase lunaire sur le sexe du bébé."
    if max(phase_ratios.values()) - min(phase_ratios.values()) < 2
    else "Écart notable — à investiguer."
))
print("=" * 60)

# ── 7. Chart ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#0f1117")

# Left: waxing vs waning
ax = axes[0]
ax.set_facecolor("#1a1d27")
bars = ax.bar(["Lune montante", "Lune descendante"],
              [ratio_waxing, ratio_waning],
              color=["#f5c518", "#5b9bd5"], width=0.4, zorder=3)
for bar, val in zip(bars, [ratio_waxing, ratio_waning]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
            f"{val:.2f}", ha="center", va="bottom", color="white",
            fontsize=12, fontweight="bold")
y_vals = [ratio_waxing, ratio_waning]
ax.set_ylim(min(y_vals) - 1.5, max(y_vals) + 1.5)
ax.set_ylabel("Garçons pour 100 filles", color="white", fontsize=10)
ax.set_title("Montante vs Descendante", color="white", fontsize=12, fontweight="bold")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#333")
ax.yaxis.grid(True, color="#333", linestyle="--", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

# Right: all 8 phases
ax2 = axes[1]
ax2.set_facecolor("#1a1d27")
present_phases = [p for p in PHASE_ORDER if p in phase_ratios]
vals = [phase_ratios[p] for p in present_phases]
colors_phases = ["#444"] * 4 + ["#f5c518"] + ["#5b9bd5"] * 3
colors_phases = colors_phases[:len(present_phases)]
bars2 = ax2.bar(range(len(present_phases)), vals, color=colors_phases, width=0.6, zorder=3)
for bar, val in zip(bars2, vals):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{val:.1f}", ha="center", va="bottom", color="white", fontsize=8)
ax2.set_xticks(range(len(present_phases)))
ax2.set_xticklabels(present_phases, rotation=30, ha="right", color="white", fontsize=8)
ax2.set_ylim(min(vals) - 1.5, max(vals) + 1.5)
ax2.set_ylabel("Garçons pour 100 filles", color="white", fontsize=10)
ax2.set_title("Les 8 phases lunaires", color="white", fontsize=12, fontweight="bold")
ax2.tick_params(colors="white")
for spine in ax2.spines.values():
    spine.set_edgecolor("#333")
ax2.yaxis.grid(True, color="#333", linestyle="--", linewidth=0.6, zorder=0)
ax2.set_axisbelow(True)

fig.suptitle("Phase lunaire et sexe du bébé — CDC 1969–2008",
             color="white", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("result_chart.png", dpi=150, bbox_inches="tight")
print("\n  Graphique sauvegardé → result_chart.png")
