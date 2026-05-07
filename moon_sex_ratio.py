import pandas as pd
import numpy as np
import ephem
import matplotlib.pyplot as plt
from datetime import timedelta

# ── 1. Load & clean births ───────────────────────────────────────────────────

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

# ── 2. Load gestation distribution ───────────────────────────────────────────

gest = pd.read_csv("daily_gestation_probabilities.csv")
# dict: offset_days -> probability weight
gest_weights = dict(zip(gest["jours_depuis_conception"].astype(int), gest["probabilite"]))
offsets = sorted(gest_weights.keys())  # 200..300

# ── 3. Pre-compute moon phase for all unique conception dates ─────────────────

print("Calcul des phases lunaires pour toutes les dates de conception possibles…")

all_conception_dates = set()
for birth_date in df["date"].unique():
    for offset in offsets:
        all_conception_dates.add(birth_date - timedelta(days=offset))

def moon_info(date):
    moon = ephem.Moon()
    moon.compute(date.strftime("%Y/%m/%d"))
    illum_today = moon.phase
    moon.compute((date + timedelta(days=1)).strftime("%Y/%m/%d"))
    is_waxing = moon.phase > illum_today

    if illum_today < 3:
        phase = "Nouvelle lune"
    elif illum_today < 50 and is_waxing:
        phase = "Croissant montant"
    elif illum_today < 55 and is_waxing:
        phase = "Premier quartier"
    elif illum_today < 98 and is_waxing:
        phase = "Gibbeuse montante"
    elif illum_today >= 98:
        phase = "Pleine lune"
    elif illum_today >= 50 and not is_waxing:
        phase = "Gibbeuse descendante"
    elif illum_today >= 45 and not is_waxing:
        phase = "Dernier quartier"
    else:
        phase = "Croissant descendant"

    return illum_today, is_waxing, phase

moon_cache = {}
for d in sorted(all_conception_dates):
    moon_cache[d] = moon_info(d)

print(f"  {len(moon_cache):,} dates calculées.")

# ── 4. Weighted expansion ─────────────────────────────────────────────────────
# For each birth row, distribute its births across all 101 conception dates
# weighted by gestation probability. Result: fractional births per phase.

print("Distribution pondérée des naissances sur les phases lunaires…")

phase_births: dict[str, dict[str, float]] = {}  # phase -> {M: w, F: w}

for _, row in df.iterrows():
    birth_date = row["date"]
    gender = row["gender"]
    births = row["births"]

    for offset in offsets:
        conception_date = birth_date - timedelta(days=offset)
        weight = gest_weights[offset]
        _, _, phase = moon_cache[conception_date]

        if phase not in phase_births:
            phase_births[phase] = {"M": 0.0, "F": 0.0}
        phase_births[phase][gender] = phase_births[phase].get(gender, 0.0) + births * weight

# ── 5. Waxing vs waning aggregate ────────────────────────────────────────────

WAXING_PHASES = {"Nouvelle lune", "Croissant montant", "Premier quartier", "Gibbeuse montante"}
WANING_PHASES = {"Pleine lune", "Gibbeuse descendante", "Dernier quartier", "Croissant descendant"}

def ratio_from_dict(d):
    m, f = d.get("M", 0.0), d.get("F", 0.0)
    return (m / f) * 100 if f > 0 else 0.0

waxing_agg = {"M": 0.0, "F": 0.0}
waning_agg = {"M": 0.0, "F": 0.0}
for phase, gdict in phase_births.items():
    target = waxing_agg if phase in WAXING_PHASES else waning_agg
    for g, v in gdict.items():
        target[g] = target.get(g, 0.0) + v

ratio_waxing = ratio_from_dict(waxing_agg)
ratio_waning = ratio_from_dict(waning_agg)

# ── 6. Results ────────────────────────────────────────────────────────────────

PHASE_ORDER = [
    "Nouvelle lune", "Croissant montant", "Premier quartier", "Gibbeuse montante",
    "Pleine lune", "Gibbeuse descendante", "Dernier quartier", "Croissant descendant",
]

print()
print("=" * 65)
print("  LUNE MONTANTE / DESCENDANTE — ANALYSE PONDÉRÉE PAR GESTATION")
print("=" * 65)
print(f"  Ratio lune MONTANTE   : {ratio_waxing:.4f} garçons / 100 filles")
print(f"  Ratio lune DESCENDANTE: {ratio_waning:.4f} garçons / 100 filles")
diff = ratio_waxing - ratio_waning
print(f"  Différence            : {diff:+.4f}")
print()
print("  Ratio par phase lunaire (naissances fractionnelles pondérées) :")
phase_ratios = {}
for phase in PHASE_ORDER:
    if phase in phase_births:
        r = ratio_from_dict(phase_births[phase])
        phase_ratios[phase] = r
        print(f"    {phase:<25} {r:.4f}")

span = max(phase_ratios.values()) - min(phase_ratios.values())
print()
print("  CONCLUSION : " + (
    f"Écart max entre phases : {span:.4f} — négligeable. Aucun effet détectable."
    if span < 2 else f"Écart max : {span:.4f} — à investiguer."
))
print("=" * 65)

# ── 7. Chart ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#0f1117")

ax = axes[0]
ax.set_facecolor("#1a1d27")
bars = ax.bar(["Lune montante", "Lune descendante"],
              [ratio_waxing, ratio_waning],
              color=["#f5c518", "#5b9bd5"], width=0.4, zorder=3)
for bar, val in zip(bars, [ratio_waxing, ratio_waning]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
            f"{val:.4f}", ha="center", va="bottom", color="white", fontsize=11, fontweight="bold")
y_vals = [ratio_waxing, ratio_waning]
ax.set_ylim(min(y_vals) - 1.5, max(y_vals) + 1.5)
ax.set_ylabel("Garçons pour 100 filles", color="white", fontsize=10)
ax.set_title("Montante vs Descendante\n(pondéré par distribution de gestation)", color="white", fontsize=11, fontweight="bold")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#333")
ax.yaxis.grid(True, color="#333", linestyle="--", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

ax2 = axes[1]
ax2.set_facecolor("#1a1d27")
present_phases = [p for p in PHASE_ORDER if p in phase_ratios]
vals = [phase_ratios[p] for p in present_phases]
colors_phases = ["#555", "#6a7fb5", "#5b9bd5", "#a8c8f0", "#f5c518", "#d4a017", "#c08000", "#8a6000"]
bars2 = ax2.bar(range(len(present_phases)), vals, color=colors_phases[:len(present_phases)], width=0.6, zorder=3)
for bar, val in zip(bars2, vals):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{val:.2f}", ha="center", va="bottom", color="white", fontsize=8)
ax2.axhline(np.mean(vals), color="#ff4b4b", linewidth=1.2, linestyle="--", label=f"Moyenne : {np.mean(vals):.2f}")
ax2.set_xticks(range(len(present_phases)))
ax2.set_xticklabels(present_phases, rotation=30, ha="right", color="white", fontsize=8)
ax2.set_ylim(min(vals) - 1.5, max(vals) + 1.5)
ax2.set_ylabel("Garçons pour 100 filles", color="white", fontsize=10)
ax2.set_title("Les 8 phases lunaires\n(pondéré par distribution de gestation)", color="white", fontsize=11, fontweight="bold")
ax2.tick_params(colors="white")
for spine in ax2.spines.values():
    spine.set_edgecolor("#333")
ax2.yaxis.grid(True, color="#333", linestyle="--", linewidth=0.5, zorder=0)
ax2.set_axisbelow(True)
ax2.legend(facecolor="#1a1d27", labelcolor="white", fontsize=9)

fig.suptitle("Phase lunaire et sexe du bébé — CDC 1969–2008 (distribution de gestation Jukic 2013)",
             color="white", fontsize=11, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("result_chart.png", dpi=150, bbox_inches="tight")
print("\n  Graphique sauvegardé → result_chart.png")
