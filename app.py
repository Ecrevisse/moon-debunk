import streamlit as st
import pandas as pd
import ephem
import matplotlib.pyplot as plt
from datetime import timedelta

st.set_page_config(page_title="Lune & Sexe du Bébé — Debunker", layout="wide")

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

PHASE_COLORS = {
    "Nouvelle lune":        "#444",
    "Croissant montant":    "#6a7fb5",
    "Premier quartier":     "#5b9bd5",
    "Gibbeuse montante":    "#a8c8f0",
    "Pleine lune":          "#f5c518",
    "Gibbeuse descendante": "#d4a017",
    "Dernier quartier":     "#c08000",
    "Croissant descendant": "#8a6000",
}

# ── Computation ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Calcul des phases lunaires…")
def load_and_compute() -> pd.DataFrame:
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
    df["conception_date"] = df["date"] - timedelta(weeks=38)

    def moon_info(date):
        moon = ephem.Moon()
        moon.compute(date.strftime("%Y/%m/%d"))
        illum_today = moon.phase
        moon.compute((date + timedelta(days=1)).strftime("%Y/%m/%d"))
        illum_next = moon.phase
        is_waxing = illum_next > illum_today

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

    info = df["conception_date"].apply(
        lambda d: pd.Series(moon_info(d), index=["moon_phase", "is_waxing", "phase_name"])
    )
    return pd.concat([df, info], axis=1)


def sex_ratio(subset: pd.DataFrame) -> float:
    m = subset.loc[subset["gender"] == "M", "births"].sum()
    f = subset.loc[subset["gender"] == "F", "births"].sum()
    return (m / f) * 100 if f > 0 else 0.0


def ratio_ci(subset: pd.DataFrame):
    """Wilson-like approx: return (ratio, margin) using sqrt(p(1-p)/n)*100."""
    import math
    m = subset.loc[subset["gender"] == "M", "births"].sum()
    f = subset.loc[subset["gender"] == "F", "births"].sum()
    n = m + f
    if n == 0:
        return 0.0, 0.0
    p = m / n
    margin = 1.96 * math.sqrt(p * (1 - p) / n) * 100 / (f / n) if f > 0 else 0
    return (m / f) * 100, margin


# ── Load ──────────────────────────────────────────────────────────────────────

df = load_and_compute()
waxing = df[df["is_waxing"]]
waning = df[~df["is_waxing"]]

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🌙 La phase lunaire influence-t-elle le sexe du bébé ?")
st.markdown(
    "Analyse basée sur **39 ans** de naissances américaines (CDC 1969–2008). "
    "La croyance : concevoir pendant une lune *montante* favoriserait les garçons, "
    "et vice versa."
)

# ── KPIs ──────────────────────────────────────────────────────────────────────

r_wax, ci_wax = ratio_ci(waxing)
r_wan, ci_wan = ratio_ci(waning)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Naissances totales", f"{df['births'].sum():,}")
c2.metric("Lune montante — ratio", f"{r_wax:.2f}", help="garçons / 100 filles")
c3.metric("Lune descendante — ratio", f"{r_wan:.2f}", help="garçons / 100 filles")
delta = r_wax - r_wan
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
    colors = ["#f5c518", "#5b9bd5"]

    bars = ax.bar(labels, vals, color=colors, width=0.4, zorder=3,
                  yerr=cis, capsize=6, error_kw={"color": "white", "linewidth": 1.5})
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(cis) + 0.05,
                f"{val:.4f}", ha="center", va="bottom",
                color="white", fontsize=11, fontweight="bold")

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
            "Les barres d'erreur (IC 95%) se chevauchent largement. "
            "La différence est pure **fluctuation statistique**.\n\n"
            "✅ **La lune montante/descendante n'influence pas le sexe du bébé.**"
        )
    elif abs(delta) < 1.5:
        st.warning(f"Différence faible ({delta:+.4f}) — probablement du bruit statistique.")
    else:
        st.error(f"Différence notable ({delta:+.4f}) — inattendu, vérifier les données.")

    st.markdown("---")
    st.markdown(f"""
**Jours lune montante :** {waxing['conception_date'].nunique():,}
**Jours lune descendante :** {waning['conception_date'].nunique():,}
**Intervalle de confiance :** ±{ci_wax:.3f} (montante), ±{ci_wan:.3f} (descendante)
""")

st.divider()

# ── Les 8 phases ──────────────────────────────────────────────────────────────

st.subheader("Ratio garçons/filles pour chacune des 8 phases lunaires")
st.caption("Si la lune avait un effet, on verrait une tendance claire — montant puis descendant ou inversement.")

phase_data = []
for phase in PHASE_ORDER:
    sub = df[df["phase_name"] == phase]
    if len(sub) == 0:
        continue
    r, ci = ratio_ci(sub)
    phase_data.append({
        "phase": phase,
        "ratio": r,
        "ci": ci,
        "n_days": sub["conception_date"].nunique(),
        "births": sub["births"].sum(),
    })

fig2, ax2 = plt.subplots(figsize=(11, 4))
fig2.patch.set_facecolor("#0f1117")
ax2.set_facecolor("#1a1d27")

xs = range(len(phase_data))
vals2 = [d["ratio"] for d in phase_data]
cis2 = [d["ci"] for d in phase_data]
colors2 = [PHASE_COLORS.get(d["phase"], "#888") for d in phase_data]

bars2 = ax2.bar(xs, vals2, color=colors2, width=0.6, zorder=3,
                yerr=cis2, capsize=5, error_kw={"color": "white", "linewidth": 1.2})
for x, val in zip(xs, vals2):
    ax2.text(x, val + max(cis2) + 0.08, f"{val:.2f}",
             ha="center", va="bottom", color="white", fontsize=8)

# Reference line = global average
global_ratio = sex_ratio(df)
ax2.axhline(global_ratio, color="#ff4b4b", linewidth=1.5, linestyle="--",
            label=f"Moyenne globale : {global_ratio:.2f}")

ax2.set_xticks(xs)
ax2.set_xticklabels([d["phase"] for d in phase_data],
                    rotation=25, ha="right", color="white", fontsize=9)
ax2.set_ylim(min(vals2) - max(cis2) - 1.5, max(vals2) + max(cis2) + 1.5)
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

# Phase summary table
st.markdown("**Détail par phase**")
table = pd.DataFrame(phase_data).rename(columns={
    "phase": "Phase", "ratio": "Ratio (G/100F)",
    "ci": "±IC 95%", "n_days": "Jours", "births": "Naissances"
})
table["Ratio (G/100F)"] = table["Ratio (G/100F)"].map("{:.4f}".format)
table["±IC 95%"] = table["±IC 95%"].map("{:.4f}".format)
table["Naissances"] = table["Naissances"].map("{:,}".format)
st.dataframe(table.set_index("Phase"), use_container_width=True)

st.divider()

# ── Explorateur ───────────────────────────────────────────────────────────────

with st.expander("Explorer les données brutes"):
    phase_filter = st.multiselect("Filtrer par phase", PHASE_ORDER, default=PHASE_ORDER)
    gender_filter = st.radio("Genre", ["Tous", "M", "F"], horizontal=True)
    filtered = df[df["phase_name"].isin(phase_filter)]
    if gender_filter != "Tous":
        filtered = filtered[filtered["gender"] == gender_filter]
    st.dataframe(
        filtered[["date", "gender", "births", "conception_date", "moon_phase", "is_waxing", "phase_name"]]
        .sort_values("date"),
        use_container_width=True, height=300,
    )
