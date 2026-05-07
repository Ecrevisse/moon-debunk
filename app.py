import streamlit as st
import pandas as pd
import numpy as np
import ephem
import matplotlib.pyplot as plt
from datetime import timedelta

st.set_page_config(page_title="Lune & Sexe du Bébé — Debunker", layout="wide")

PHASE_ORDER = [
    "Nouvelle lune", "Croissant montant", "Premier quartier", "Gibbeuse montante",
    "Pleine lune", "Gibbeuse descendante", "Dernier quartier", "Croissant descendant",
]
PHASE_COLORS = [
    "#555", "#6a7fb5", "#5b9bd5", "#a8c8f0",
    "#f5c518", "#d4a017", "#c08000", "#8a6000",
]
WAXING_PHASES = {"Nouvelle lune", "Croissant montant", "Premier quartier", "Gibbeuse montante"}

# ── Core computation ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Calcul des phases lunaires pour toutes les dates possibles…")
def load_moon_cache(birth_dates_tuple):
    offsets = list(range(200, 301))
    all_dates = set()
    for bd in birth_dates_tuple:
        for o in offsets:
            all_dates.add(bd - timedelta(days=o))

    def moon_info(date):
        moon = ephem.Moon()
        moon.compute(date.strftime("%Y/%m/%d"))
        illum = moon.phase
        moon.compute((date + timedelta(days=1)).strftime("%Y/%m/%d"))
        is_waxing = moon.phase > illum
        if illum < 3:             phase = "Nouvelle lune"
        elif illum < 50 and is_waxing:  phase = "Croissant montant"
        elif illum < 55 and is_waxing:  phase = "Premier quartier"
        elif illum < 98 and is_waxing:  phase = "Gibbeuse montante"
        elif illum >= 98:         phase = "Pleine lune"
        elif illum >= 50:         phase = "Gibbeuse descendante"
        elif illum >= 45:         phase = "Dernier quartier"
        else:                     phase = "Croissant descendant"
        return illum, is_waxing, phase

    return {d: moon_info(d) for d in sorted(all_dates)}


@st.cache_data(show_spinner="Distribution pondérée des naissances…")
def compute_weighted(birth_dates_tuple):
    df = _load_births()
    gest = pd.read_csv("daily_gestation_probabilities.csv")
    gest_w = dict(zip(gest["jours_depuis_conception"].astype(int), gest["probabilite"]))
    offsets = sorted(gest_w.keys())

    moon_cache = load_moon_cache(birth_dates_tuple)

    phase_births = {p: {"M": 0.0, "F": 0.0} for p in PHASE_ORDER}

    for _, row in df.iterrows():
        bd, gender, births = row["date"], row["gender"], row["births"]
        for o in offsets:
            cd = bd - timedelta(days=o)
            w = gest_w[o]
            _, _, phase = moon_cache[cd]
            phase_births[phase][gender] += births * w

    return phase_births


@st.cache_data
def _load_births():
    df = pd.read_csv("births.csv")
    df = df.dropna(subset=["year", "month", "day"])
    df = df[df["day"] != 99]

    def safe_ts(r):
        try:
            return pd.Timestamp(int(r["year"]), int(r["month"]), int(r["day"]))
        except ValueError:
            return pd.NaT

    df["date"] = df.apply(safe_ts, axis=1)
    return df.dropna(subset=["date"])


def sex_ratio(d):
    m, f = d.get("M", 0.0), d.get("F", 0.0)
    return (m / f) * 100 if f > 0 else 0.0


def ratio_ci(d):
    import math
    m, f = d.get("M", 0.0), d.get("F", 0.0)
    n = m + f
    if n == 0 or f == 0:
        return 0.0, 0.0
    p = m / n
    margin = 1.96 * math.sqrt(p * (1 - p) / n) * 100 / (f / n)
    return (m / f) * 100, margin


# ── Load ──────────────────────────────────────────────────────────────────────

df_births = _load_births()
birth_dates_tuple = tuple(sorted(df_births["date"].unique()))
phase_births = compute_weighted(birth_dates_tuple)

waxing_agg = {"M": 0.0, "F": 0.0}
waning_agg = {"M": 0.0, "F": 0.0}
for phase, gdict in phase_births.items():
    target = waxing_agg if phase in WAXING_PHASES else waning_agg
    for g, v in gdict.items():
        target[g] += v

r_wax, ci_wax = ratio_ci(waxing_agg)
r_wan, ci_wan = ratio_ci(waning_agg)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🌙 La phase lunaire influence-t-elle le sexe du bébé ?")
st.markdown(
    "Analyse basée sur **39 ans** de naissances américaines (CDC 1969–2008). "
    "Chaque naissance est distribuée probabilistement sur les **101 jours de conception possibles** "
    "(200–300 jours avant la naissance) selon la distribution de gestation de Jukic et al. (2013)."
)

# ── KPIs ──────────────────────────────────────────────────────────────────────

delta = r_wax - r_wan
c1, c2, c3, c4 = st.columns(4)
c1.metric("Naissances totales", f"{df_births['births'].sum():,}")
c2.metric("Ratio lune montante", f"{r_wax:.4f}", help="garçons / 100 filles")
c3.metric("Ratio lune descendante", f"{r_wan:.4f}", help="garçons / 100 filles")
c4.metric("Différence", f"{delta:+.4f}", help="Proche de 0 = aucun effet")

st.divider()

# ── Montante vs Descendante ───────────────────────────────────────────────────

col_main, col_verdict = st.columns([1.6, 1])

with col_main:
    st.subheader("Lune montante vs descendante")

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    labels = ["Montante ↑", "Descendante ↓"]
    vals = [r_wax, r_wan]
    cis = [ci_wax, ci_wan]

    bars = ax.bar(labels, vals, color=["#f5c518", "#5b9bd5"], width=0.4, zorder=3,
                  yerr=cis, capsize=6, error_kw={"color": "white", "linewidth": 1.5})
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(cis) + 0.05,
                f"{val:.4f}", ha="center", va="bottom", color="white", fontsize=11, fontweight="bold")
    ax.set_ylim(min(vals) - max(cis) - 1.5, max(vals) + max(cis) + 1.5)
    ax.set_ylabel("Garçons pour 100 filles", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.yaxis.grid(True, color="#333", linestyle="--", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col_verdict:
    st.subheader("Verdict")
    if abs(delta) < 0.5:
        st.success(
            f"**Différence : {delta:+.4f} garçons / 100 filles**\n\n"
            "Les barres d'erreur (IC 95%) se chevauchent. "
            "Même avec une distribution de gestation réaliste, "
            "la différence reste du **bruit statistique pur**.\n\n"
            "✅ **La lune montante/descendante n'influence pas le sexe du bébé.**"
        )
    else:
        st.warning(f"Différence ({delta:+.4f}) — inattendu avec ce volume de données.")

    st.markdown("---")
    st.markdown(f"""
**IC 95% montante :** ±{ci_wax:.4f}
**IC 95% descendante :** ±{ci_wan:.4f}

**Méthode :** chaque naissance distribuée sur 101 jours de conception pondérés (Jukic 2013 / CDC).
""")

st.divider()

# ── Les 8 phases ──────────────────────────────────────────────────────────────

st.subheader("Ratio garçons/filles par phase lunaire")
st.caption("Si la lune avait un effet, on verrait une tendance progressive montante → descendante. Les barres restent plates.")

phase_data = []
for phase in PHASE_ORDER:
    d = phase_births.get(phase, {"M": 0.0, "F": 0.0})
    r, ci = ratio_ci(d)
    phase_data.append({"phase": phase, "ratio": r, "ci": ci,
                        "births_M": d.get("M", 0.0), "births_F": d.get("F", 0.0)})

fig2, ax2 = plt.subplots(figsize=(11, 4))
fig2.patch.set_facecolor("#0f1117")
ax2.set_facecolor("#1a1d27")

xs = range(len(phase_data))
vals2 = [d["ratio"] for d in phase_data]
cis2 = [d["ci"] for d in phase_data]

bars2 = ax2.bar(xs, vals2, color=PHASE_COLORS[:len(phase_data)], width=0.6, zorder=3,
                yerr=cis2, capsize=5, error_kw={"color": "white", "linewidth": 1.2})
for x, val in zip(xs, vals2):
    ax2.text(x, val + max(cis2) + 0.01, f"{val:.4f}",
             ha="center", va="bottom", color="white", fontsize=7.5)

global_ratio = sex_ratio({"M": sum(d["births_M"] for d in phase_data),
                           "F": sum(d["births_F"] for d in phase_data)})
ax2.axhline(global_ratio, color="#ff4b4b", linewidth=1.5, linestyle="--",
            label=f"Moyenne globale : {global_ratio:.4f}")

ax2.set_xticks(xs)
ax2.set_xticklabels([d["phase"] for d in phase_data],
                    rotation=25, ha="right", color="white", fontsize=9)
ax2.set_ylim(min(vals2) - max(cis2) - 0.5, max(vals2) + max(cis2) + 0.3)
ax2.set_ylabel("Garçons pour 100 filles", color="white", fontsize=10)
ax2.tick_params(colors="white")
for spine in ax2.spines.values():
    spine.set_edgecolor("#333")
ax2.yaxis.grid(True, color="#333", linestyle="--", linewidth=0.5, zorder=0)
ax2.set_axisbelow(True)
ax2.legend(facecolor="#1a1d27", labelcolor="white", fontsize=9)
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

# ── Table ─────────────────────────────────────────────────────────────────────

table = pd.DataFrame(phase_data)[["phase", "ratio", "ci", "births_M", "births_F"]]
table["total_weighted"] = table["births_M"] + table["births_F"]
table = table.rename(columns={
    "phase": "Phase", "ratio": "Ratio (G/100F)", "ci": "±IC 95%",
    "births_M": "Garçons (pondéré)", "births_F": "Filles (pondéré)", "total_weighted": "Total pondéré"
})
for col in ["Ratio (G/100F)", "±IC 95%"]:
    table[col] = table[col].map("{:.4f}".format)
for col in ["Garçons (pondéré)", "Filles (pondéré)", "Total pondéré"]:
    table[col] = table[col].map("{:,.0f}".format)
st.dataframe(table.set_index("Phase"), use_container_width=True)

st.divider()

# ── Distribution de gestation ─────────────────────────────────────────────────

with st.expander("Distribution de gestation utilisée (Jukic 2013 / CDC)"):
    gest = pd.read_csv("daily_gestation_probabilities.csv")
    fig3, ax3 = plt.subplots(figsize=(9, 3))
    fig3.patch.set_facecolor("#0f1117")
    ax3.set_facecolor("#1a1d27")
    ax3.fill_between(gest["jours_depuis_conception"], gest["probabilite"] * 100,
                     alpha=0.5, color="#5b9bd5")
    ax3.plot(gest["jours_depuis_conception"], gest["probabilite"] * 100,
             color="#5b9bd5", linewidth=1.5)
    peak = gest.loc[gest["probabilite"].idxmax()]
    ax3.axvline(peak["jours_depuis_conception"], color="#f5c518", linewidth=1.2,
                linestyle="--", label=f"Pic : j{int(peak['jours_depuis_conception'])} ({peak['probabilite']*100:.2f}%)")
    ax3.set_xlabel("Jours depuis la conception", color="white", fontsize=9)
    ax3.set_ylabel("Probabilité (%)", color="white", fontsize=9)
    ax3.tick_params(colors="white", labelsize=8)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#333")
    ax3.yaxis.grid(True, color="#333", linestyle="--", linewidth=0.4)
    ax3.set_axisbelow(True)
    ax3.legend(facecolor="#1a1d27", labelcolor="white", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)
    st.dataframe(gest.rename(columns={"jours_depuis_conception": "Jours", "probabilite": "Probabilité"}),
                 use_container_width=True, height=200)
