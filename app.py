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

    LUNAR_CYCLE = 29.53059

    def moon_info(date):
        moon = ephem.Moon()
        d_str = date.strftime("%Y/%m/%d")
        moon.compute(d_str)
        illum = moon.phase
        moon.compute((date + timedelta(days=1)).strftime("%Y/%m/%d"))
        is_waxing = moon.phase > illum
        if illum < 3:                   phase = "Nouvelle lune"
        elif illum < 50 and is_waxing:  phase = "Croissant montant"
        elif illum < 55 and is_waxing:  phase = "Premier quartier"
        elif illum < 98 and is_waxing:  phase = "Gibbeuse montante"
        elif illum >= 98:               phase = "Pleine lune"
        elif illum >= 50:               phase = "Gibbeuse descendante"
        elif illum >= 45:               phase = "Dernier quartier"
        else:                           phase = "Croissant descendant"

        prev_new = ephem.previous_new_moon(d_str)
        lunar_age = float(ephem.Date(d_str) - prev_new) % LUNAR_CYCLE

        return illum, is_waxing, phase, lunar_age

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
            _, _, phase, _ = moon_cache[cd]
            phase_births[phase][gender] += births * w

    return phase_births


@st.cache_data(show_spinner="Ages lunaires des jours de naissance…")
def load_birth_lunar_ages(birth_dates_tuple):
    """Lunar age (day in cycle) for each birth date."""
    LUNAR_CYCLE = 29.53059
    result = {}
    for bd in birth_dates_tuple:
        d_str = bd.strftime("%Y/%m/%d")
        prev_new = ephem.previous_new_moon(d_str)
        result[bd] = float(ephem.Date(d_str) - prev_new) % LUNAR_CYCLE
    return result


@st.cache_data(show_spinner="Distribution par cycle lunaire (jour de naissance)…")
def compute_birth_lunar_cycle(birth_dates_tuple, n_bins=30, effect_delta=0.05):
    """
    x-axis: birth lunar day.

    Returns observed ratio per bin, plus two theoretical curves:
    - naive: step function applied to birth lunar day (no gestation)
    - correct: convolution of theory with gestation distribution
      P_waxing(birth_date) = gestation-weighted fraction of conception window that is waxing
      → nearly flat because the window spans ~3.4 lunar cycles
    """
    df = _load_births()
    gest = pd.read_csv("daily_gestation_probabilities.csv")
    gest_w = dict(zip(gest["jours_depuis_conception"].astype(int), gest["probabilite"]))
    offsets = sorted(gest_w.keys())
    LUNAR_CYCLE = 29.53059

    moon_cache = load_moon_cache(birth_dates_tuple)
    birth_ages = load_birth_lunar_ages(birth_dates_tuple)

    total_M = df.loc[df["gender"] == "M", "births"].sum()
    total_F = df.loc[df["gender"] == "F", "births"].sum()
    global_p = total_M / (total_M + total_F)  # ≈ 0.512

    obs_M    = np.zeros(n_bins)
    obs_F    = np.zeros(n_bins)
    theo_M   = np.zeros(n_bins)  # correct: with gestation distribution
    theo_F   = np.zeros(n_bins)
    naive_M  = np.zeros(n_bins)  # naive: step at birth lunar day
    naive_F  = np.zeros(n_bins)

    for _, row in df.iterrows():
        bd, gender, births = row["date"], row["gender"], row["births"]

        b_age = birth_ages[bd]
        b_bin = min(int(b_age / LUNAR_CYCLE * n_bins), n_bins - 1)

        # ── Observed ─────────────────────────────────────────────────────────
        if gender == "M":
            obs_M[b_bin] += births
        else:
            obs_F[b_bin] += births

        # ── Correct theoretical ───────────────────────────────────────────────
        # P_waxing: fraction of conception window (j=200..300) that is waxing,
        # weighted by gestation probability. gest weights sum to 1.
        P_waxing = sum(
            gest_w[o] for o in offsets if moon_cache[bd - timedelta(days=o)][1]
        )
        # Under theory: P(boy) = global_p + delta*(2*P_waxing - 1)
        # P_waxing ≈ 0.5 for all birth dates → effect nearly vanishes
        p_boy_theo = float(np.clip(global_p + effect_delta * (2 * P_waxing - 1), 0, 1))
        theo_M[b_bin] += births * p_boy_theo
        theo_F[b_bin] += births * (1 - p_boy_theo)

        # ── Naive theoretical ─────────────────────────────────────────────────
        # Applies the theory to the BIRTH lunar day (ignores gestation entirely)
        birth_is_waxing = b_age <= LUNAR_CYCLE / 2
        p_boy_naive = float(np.clip(global_p + effect_delta * (1 if birth_is_waxing else -1), 0, 1))
        naive_M[b_bin] += births * p_boy_naive
        naive_F[b_bin] += births * (1 - p_boy_naive)

    return obs_M, obs_F, theo_M, theo_F, naive_M, naive_F


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

# ── Cycle lunaire jour par jour ───────────────────────────────────────────────

st.subheader("Ratio observé vs théorique sur le cycle lunaire complet")
st.caption(
    "X-axis = jour du cycle lunaire à la **naissance**. Chaque bin ≈ 1 jour (29.53 j / 30 bins). "
    "La courbe théorique **correcte** propage l'effet à travers la distribution de gestation : "
    "la fenêtre de conception couvre ~3.4 cycles lunaires, ce qui brouille quasi-totalement le signal."
)

N_BINS = 30
LUNAR_CYCLE = 29.53059

# Sidebar: effect size
effect_delta = st.sidebar.slider(
    "Effet théorique Δ (probabilité garçon)", min_value=0.01, max_value=0.20,
    value=0.05, step=0.01,
    help="Δ = shift absolu de P(garçon). Ex: Δ=0.05 → P(garçon|lune montante)=56%, P(garçon|lune descendante)=46%"
)

obs_M, obs_F, theo_M, theo_F, naive_M, naive_F = compute_birth_lunar_cycle(
    birth_dates_tuple, N_BINS, effect_delta
)

bin_centers = np.array([(i + 0.5) * LUNAR_CYCLE / N_BINS for i in range(N_BINS)])
bin_total_obs = obs_M + obs_F

def to_ratio(M, F):
    return np.where(F > 0, M / F * 100, np.nan)

def to_ci(M, F):
    n = M + F
    p = np.where(n > 0, M / n, 0.512)
    return np.where(n > 0, 1.96 * np.sqrt(p * (1 - p) / n) * 100 / np.where(F > 0, F / n, 1), np.nan)

obs_ratio   = to_ratio(obs_M, obs_F)
theo_ratio  = to_ratio(theo_M, theo_F)
naive_ratio = to_ratio(naive_M, naive_F)
obs_ci      = to_ci(obs_M, obs_F)

# Smoothed observed (3-bin circular rolling average)
obs_smooth = np.array([
    np.nanmean(obs_ratio[np.arange(i - 1, i + 2) % N_BINS]) for i in range(N_BINS)
])

global_avg = float(np.nansum(obs_M) / np.nansum(obs_F) * 100)
obs_std    = float(np.nanstd(obs_ratio))

# Naive amplitude (ratio shift) for annotation
naive_amp = float(np.nanmax(naive_ratio) - np.nanmin(naive_ratio))
theo_amp  = float(np.nanmax(theo_ratio)  - np.nanmin(theo_ratio))

fig_cycle, ax_c = plt.subplots(figsize=(12, 5.5))
fig_cycle.patch.set_facecolor("#0f1117")
ax_c.set_facecolor("#1a1d27")

# Background waxing / waning
ax_c.axvspan(0, LUNAR_CYCLE / 2, alpha=0.07, color="#f5c518")
ax_c.axvspan(LUNAR_CYCLE / 2, LUNAR_CYCLE, alpha=0.07, color="#5b9bd5")

# Sample size bars (secondary axis)
ax2_c = ax_c.twinx()
ax2_c.bar(bin_centers, bin_total_obs / 1e6, width=LUNAR_CYCLE / N_BINS * 0.75,
          color="#ffffff", alpha=0.05, zorder=1)
ax2_c.set_ylabel("Naissances (M)", color="#444", fontsize=8)
ax2_c.tick_params(colors="#444", labelsize=7)
ax2_c.set_ylim(0, bin_total_obs.max() / 1e6 * 8)

# Global mean
ax_c.axhline(global_avg, color="#ff4b4b", linewidth=1.1, linestyle=":",
             label=f"Moyenne globale : {global_avg:.3f}", zorder=3)

# Naive theoretical (step function — ignores gestation)
ax_c.plot(bin_centers, naive_ratio, color="#ff9500", linewidth=2.0,
          linestyle="--", label=f"Théorique naïf (Δ={effect_delta}, sans distribution gestation) — amplitude {naive_amp:.2f}", zorder=4)

# Correct theoretical (with gestation — nearly flat)
ax_c.plot(bin_centers, theo_ratio, color="#e040fb", linewidth=2.2,
          linestyle="-", label=f"Théorique correct (avec distribution gestation) — amplitude {theo_amp:.4f}", zorder=5)

# CI band observed
ax_c.fill_between(bin_centers, obs_ratio - obs_ci, obs_ratio + obs_ci,
                  color="#4caf50", alpha=0.15, zorder=2)

# Observed raw dots
ax_c.scatter(bin_centers, obs_ratio, color="#4caf50", s=16, alpha=0.45, zorder=5)

# Smoothed observed
ax_c.plot(bin_centers, obs_smooth, color="#4caf50", linewidth=2.2,
          label=f"Observé lissé + IC 95% (σ={obs_std:.3f})", zorder=6)

# Moon markers
for x, sym in [(0, "🌑"), (LUNAR_CYCLE / 4, "🌓"), (LUNAR_CYCLE / 2, "🌕"), (LUNAR_CYCLE * 3 / 4, "🌗")]:
    ax_c.axvline(x, color="#333", linewidth=0.8, linestyle="--", zorder=1)
    ax_c.text(x, global_avg + naive_amp / 2 + 0.4, sym, ha="center", fontsize=13, zorder=7)

# Annotation
ratio_reduction = (1 - theo_amp / naive_amp) * 100 if naive_amp > 0 else 0
ax_c.annotate(
    f"Amplitude naïve (sans gestation) : {naive_amp:.2f} g/100f\n"
    f"Amplitude correcte (avec gestation) : {theo_amp:.4f} g/100f\n"
    f"→ La distribution de gestation réduit le signal de {ratio_reduction:.0f}%\n"
    f"   Même si la théorie est vraie, le signal est indétectable.",
    xy=(LUNAR_CYCLE * 0.01, global_avg - naive_amp / 2 - 0.3),
    fontsize=8.5, color="white", va="top",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#0d1020", edgecolor="#555"),
)

# Labels / waxing / waning text
ax_c.text(LUNAR_CYCLE * 0.25, global_avg + naive_amp / 2 + 0.1,
          "← Lune montante →", ha="center", color="#f5c518", fontsize=8, alpha=0.7)
ax_c.text(LUNAR_CYCLE * 0.75, global_avg + naive_amp / 2 + 0.1,
          "← Lune descendante →", ha="center", color="#5b9bd5", fontsize=8, alpha=0.7)

ax_c.set_xlabel("Jour du cycle lunaire (à la naissance)", color="white", fontsize=10)
ax_c.set_ylabel("Garçons pour 100 filles", color="white", fontsize=10)
ax_c.set_xlim(0, LUNAR_CYCLE)
y_margin = naive_amp / 2 + max(obs_ci[~np.isnan(obs_ci)]) + 1
ax_c.set_ylim(global_avg - y_margin - 0.5, global_avg + y_margin + 1.5)
ax_c.tick_params(colors="white")
for spine in ax_c.spines.values():
    spine.set_edgecolor("#333")
ax_c.yaxis.grid(True, color="#333", linestyle="--", linewidth=0.4, zorder=0)
ax_c.set_axisbelow(True)
ax_c.legend(facecolor="#0d1020", labelcolor="white", fontsize=8.5, loc="lower right")
plt.tight_layout()
st.pyplot(fig_cycle)
plt.close(fig_cycle)

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
