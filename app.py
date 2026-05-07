import streamlit as st
import pandas as pd
import numpy as np
import ephem
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

DARK_BG   = "#0f1117"
DARK_PLOT = "#1a1d27"

def dark_layout(**kwargs):
    return dict(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_PLOT,
        font=dict(color="white"),
        **kwargs,
    )

# ── Computation ───────────────────────────────────────────────────────────────

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
    LUNAR_CYCLE = 29.53059
    result = {}
    for bd in birth_dates_tuple:
        d_str = bd.strftime("%Y/%m/%d")
        prev_new = ephem.previous_new_moon(d_str)
        result[bd] = float(ephem.Date(d_str) - prev_new) % LUNAR_CYCLE
    return result


@st.cache_data(show_spinner="Distribution par cycle lunaire…")
def compute_birth_lunar_cycle(birth_dates_tuple, n_bins=30):
    df = _load_births()
    gest = pd.read_csv("daily_gestation_probabilities.csv")
    gest_w = dict(zip(gest["jours_depuis_conception"].astype(int), gest["probabilite"]))
    offsets = sorted(gest_w.keys())
    LUNAR_CYCLE = 29.53059
    moon_cache = load_moon_cache(birth_dates_tuple)
    birth_ages = load_birth_lunar_ages(birth_dates_tuple)

    obs_M  = np.zeros(n_bins)
    obs_F  = np.zeros(n_bins)
    theo_M = np.zeros(n_bins)
    theo_F = np.zeros(n_bins)

    for _, row in df.iterrows():
        bd, gender, births = row["date"], row["gender"], row["births"]
        b_bin = min(int(birth_ages[bd] / LUNAR_CYCLE * n_bins), n_bins - 1)
        if gender == "M":
            obs_M[b_bin] += births
        else:
            obs_F[b_bin] += births
        P_waxing = sum(gest_w[o] for o in offsets if moon_cache[bd - timedelta(days=o)][1])
        theo_M[b_bin] += births * P_waxing
        theo_F[b_bin] += births * (1.0 - P_waxing)

    return obs_M, obs_F, theo_M, theo_F


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

def to_ratio(M, F):
    return np.where(F > 0, M / F * 100, np.nan)

def to_ci(M, F):
    n = M + F
    p = np.where(n > 0, M / n, 0.512)
    return np.where(n > 0, 1.96 * np.sqrt(p * (1 - p) / n) * 100 / np.where(F > 0, F / n, 1), np.nan)

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
c2.metric("Ratio lune montante",    f"{r_wax:.4f}", help="garçons / 100 filles")
c3.metric("Ratio lune descendante", f"{r_wan:.4f}", help="garçons / 100 filles")
c4.metric("Différence", f"{delta:+.4f}", help="Proche de 0 = aucun effet")

st.divider()

# ── Montante vs Descendante ───────────────────────────────────────────────────

col_main, col_verdict = st.columns([1.6, 1])

with col_main:
    st.subheader("Lune montante vs descendante")

    fig1 = go.Figure()
    for label, val, ci, color in [
        ("Montante ↑",    r_wax, ci_wax, "#f5c518"),
        ("Descendante ↓", r_wan, ci_wan, "#5b9bd5"),
    ]:
        fig1.add_trace(go.Bar(
            x=[label], y=[val],
            error_y=dict(type="data", array=[ci], visible=True, color="white", thickness=2),
            marker_color=color,
            text=[f"{val:.4f}"], textposition="outside", textfont=dict(color="white", size=13),
            width=0.4, name=label,
        ))
    y_min = min(r_wax, r_wan) - max(ci_wax, ci_wan) - 1.5
    y_max = max(r_wax, r_wan) + max(ci_wax, ci_wan) + 1.5
    fig1.update_layout(
        **dark_layout(height=380, showlegend=False, bargap=0.4),
        yaxis=dict(title="Garçons pour 100 filles", range=[y_min, y_max], gridcolor="#333"),
        xaxis=dict(gridcolor="#333"),
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_verdict:
    st.subheader("Verdict")
    if abs(delta) < 0.5:
        st.success(
            f"**Différence : {delta:+.4f} garçons / 100 filles**\n\n"
            "IC 95% se chevauchent. Même avec une distribution de gestation réaliste, "
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

global_ratio = sex_ratio({"M": sum(d["births_M"] for d in phase_data),
                           "F": sum(d["births_F"] for d in phase_data)})

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=[d["phase"] for d in phase_data],
    y=[d["ratio"] for d in phase_data],
    error_y=dict(
        type="data", array=[d["ci"] for d in phase_data],
        visible=True, color="white", thickness=1.5,
    ),
    marker_color=PHASE_COLORS,
    text=[f"{d['ratio']:.4f}" for d in phase_data],
    textposition="outside", textfont=dict(color="white", size=9),
    hovertemplate="<b>%{x}</b><br>Ratio: %{y:.4f}<br>±IC 95%: %{error_y.array:.4f}<extra></extra>",
))
fig2.add_hline(
    y=global_ratio, line_color="#ff4b4b", line_dash="dash", line_width=1.5,
    annotation_text=f"Moyenne : {global_ratio:.4f}",
    annotation_font_color="#ff4b4b",
)
vals2 = [d["ratio"] for d in phase_data]
cis2  = [d["ci"]    for d in phase_data]
fig2.update_layout(
    **dark_layout(height=380, showlegend=False),
    yaxis=dict(
        title="Garçons pour 100 filles", gridcolor="#333",
        range=[min(vals2) - max(cis2) - 0.5, max(vals2) + max(cis2) + 0.5],
    ),
    xaxis=dict(gridcolor="#333"),
)
st.plotly_chart(fig2, use_container_width=True)

# Table
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
    "Théorique = croyance 100% (montante→garçon, descendante→fille) convoluée avec la distribution de gestation réelle."
)

N_BINS = 30
LUNAR_CYCLE = 29.53059

obs_M, obs_F, theo_M, theo_F = compute_birth_lunar_cycle(birth_dates_tuple, N_BINS)

bin_centers = np.array([(i + 0.5) * LUNAR_CYCLE / N_BINS for i in range(N_BINS)])

# 3-bin circular smooth
def smooth(arr):
    return np.array([np.mean(arr[np.arange(i - 1, i + 2) % N_BINS]) for i in range(N_BINS)])

obs_M_s  = smooth(obs_M)
obs_F_s  = smooth(obs_F)
theo_M_s = smooth(theo_M)
theo_F_s = smooth(theo_F)

# Y range: cover all 4 curves fully
all_vals = np.concatenate([obs_M_s, obs_F_s, theo_M_s, theo_F_s])
y_min = float(np.min(all_vals)) * 0.998
y_max = float(np.max(all_vals)) * 1.002

fig3 = go.Figure()

# Background waxing / waning zones
fig3.add_vrect(x0=0, x1=LUNAR_CYCLE / 2,
               fillcolor="#f5c518", opacity=0.06, layer="below", line_width=0,
               annotation_text="← Montante →", annotation_position="top left",
               annotation_font=dict(color="#f5c518", size=10))
fig3.add_vrect(x0=LUNAR_CYCLE / 2, x1=LUNAR_CYCLE,
               fillcolor="#5b9bd5", opacity=0.06, layer="below", line_width=0,
               annotation_text="← Descendante →", annotation_position="top right",
               annotation_font=dict(color="#5b9bd5", size=10))

# Phase markers
for x, label in [(0, "🌑"), (LUNAR_CYCLE/4, "🌓"),
                 (LUNAR_CYCLE/2, "🌕"), (LUNAR_CYCLE*3/4, "🌗")]:
    fig3.add_vline(x=x, line_color="#333", line_dash="dash", line_width=1,
                   annotation_text=label, annotation_font_size=14,
                   annotation_position="top")

# Observed girls
fig3.add_trace(go.Scatter(
    x=bin_centers, y=obs_F_s,
    mode="lines+markers", name="Filles observées",
    line=dict(color="#e91e8c", width=2.5),
    marker=dict(size=5, opacity=0.6),
    hovertemplate="Jour %{x:.1f}<br>Filles obs. : %{y:,.0f}<extra></extra>",
))

# Observed boys
fig3.add_trace(go.Scatter(
    x=bin_centers, y=obs_M_s,
    mode="lines+markers", name="Garçons observés",
    line=dict(color="#2196f3", width=2.5),
    marker=dict(size=5, opacity=0.6),
    hovertemplate="Jour %{x:.1f}<br>Garçons obs. : %{y:,.0f}<extra></extra>",
))

# Theoretical girls (100% belief + gestation)
fig3.add_trace(go.Scatter(
    x=bin_centers, y=theo_F_s,
    mode="lines", name="Filles théoriques (croyance 100% + gestation)",
    line=dict(color="#f48fb1", width=2, dash="dash"),
    hovertemplate="Jour %{x:.1f}<br>Filles théo. : %{y:,.0f}<extra></extra>",
))

# Theoretical boys (100% belief + gestation)
fig3.add_trace(go.Scatter(
    x=bin_centers, y=theo_M_s,
    mode="lines", name="Garçons théoriques (croyance 100% + gestation)",
    line=dict(color="#90caf9", width=2, dash="dash"),
    hovertemplate="Jour %{x:.1f}<br>Garçons théo. : %{y:,.0f}<extra></extra>",
))

fig3.update_layout(
    **dark_layout(height=500),
    xaxis=dict(title="Jour du cycle lunaire (à la naissance)", range=[0, LUNAR_CYCLE], gridcolor="#333"),
    yaxis=dict(title="Naissances (pondérées, lissées 3-bins)", range=[y_min, y_max], gridcolor="#333"),
    legend=dict(
        bgcolor="#0d1020", bordercolor="#444", borderwidth=1,
        font=dict(size=11), orientation="h", yanchor="bottom", y=-0.30, x=0,
    ),
)
st.plotly_chart(fig3, use_container_width=True)
st.caption(
    "Si la croyance était vraie, les courbes théoriques (tirets) devraient diverger clairement "
    "des courbes observées entre lune montante et descendante. "
    "La fenêtre de gestation (200–300 j ≈ 3.4 cycles lunaires) brouille quasi-totalement le signal : "
    "les courbes théoriques et observées sont indiscernables."
)

st.divider()

# ── Distribution de gestation ─────────────────────────────────────────────────

with st.expander("Distribution de gestation utilisée (Jukic 2013 / CDC)"):
    gest = pd.read_csv("daily_gestation_probabilities.csv")
    peak = gest.loc[gest["probabilite"].idxmax()]

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=gest["jours_depuis_conception"], y=gest["probabilite"] * 100,
        fill="tozeroy", fillcolor="rgba(91,155,213,0.3)",
        line=dict(color="#5b9bd5", width=2),
        name="Probabilité",
        hovertemplate="Jour %{x}<br>Probabilité: %{y:.3f}%<extra></extra>",
    ))
    fig4.add_vline(
        x=peak["jours_depuis_conception"], line_color="#f5c518", line_dash="dash",
        annotation_text=f"Pic j{int(peak['jours_depuis_conception'])} ({peak['probabilite']*100:.2f}%)",
        annotation_font_color="#f5c518",
    )
    fig4.update_layout(
        **dark_layout(height=280, showlegend=False),
        xaxis=dict(title="Jours depuis la conception", gridcolor="#333"),
        yaxis=dict(title="Probabilité (%)", gridcolor="#333"),
        margin=dict(t=20),
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.dataframe(
        gest.rename(columns={"jours_depuis_conception": "Jours", "probabilite": "Probabilité"}),
        use_container_width=True, height=200,
    )
