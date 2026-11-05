from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NFL Probability Engine | Dashboard", page_icon="🏈", layout="wide")
DATA = Path(__file__).parent / "data"

@st.cache_data
def load():
    return {name: pd.read_csv(DATA / f"{name}.csv") for name in ["predictions_2026", "weekly_predictions_2026", "elo_evolution_2026", "wins_distribution_2026", "elo_final_2025"]}

d = load()
probs = d["predictions_2026"].copy()
games = d["weekly_predictions_2026"].copy()
elos = d["elo_evolution_2026"].copy()
wins = d["wins_distribution_2026"].copy()

st.title("🏈 NFL Probability Engine")
st.caption("Dashboard basado en archivos CSV proporcionados por el usuario | Temporada 2026 | Corte de resultados registrados: semana 6. No son pronósticos en tiempo real.")
st.info("Los resultados muestran una simulación histórica del modelo cargado. Para pronósticos actualizados, es necesario volver a correr el motor con resultados reales recientes.")

names = {"ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens","BUF":"Buffalo Bills","CAR":"Carolina Panthers","CHI":"Chicago Bears","CIN":"Cincinnati Bengals","CLE":"Cleveland Browns","DAL":"Dallas Cowboys","DEN":"Denver Broncos","DET":"Detroit Lions","GB":"Green Bay Packers","HOU":"Houston Texans","IND":"Indianapolis Colts","JAX":"Jacksonville Jaguars","KC":"Kansas City Chiefs","LA":"Los Angeles Rams","LAC":"Los Angeles Chargers","LV":"Las Vegas Raiders","MIA":"Miami Dolphins","MIN":"Minnesota Vikings","NE":"New England Patriots","NO":"New Orleans Saints","NYG":"New York Giants","NYJ":"New York Jets","PHI":"Philadelphia Eagles","PIT":"Pittsburgh Steelers","SEA":"Seattle Seahawks","SF":"San Francisco 49ers","TB":"Tampa Bay Buccaneers","TEN":"Tennessee Titans","WAS":"Washington Commanders"}
fmt=lambda t: f"{t} · {names.get(t,t)}"

section = st.sidebar.radio("Explorar", ["Panorama general", "Partidos por semana", "Detalle de equipos", "Simulaciones de victorias", "Metodología y datos"])

if section == "Panorama general":
    a,b,c,dcol=st.columns(4)
    a.metric("Equipos",len(probs));b.metric("Temporadas simuladas",len(wins));c.metric("Partidos del calendario",len(games));dcol.metric("Partidos marcados jugados",int(games.played.sum()))
    st.subheader("Probabilidad de ganar el Super Bowl")
    ranked = probs.sort_values("pct_sb_winner",ascending=False).copy()
    ranked["Equipo"] = ranked.team.map(fmt)
    st.bar_chart(ranked.set_index("Equipo")["pct_sb_winner"].head(16),x_label="Equipo",y_label="Probabilidad (%)",horizontal=True)
    st.subheader("Tabla de probabilidades por equipo")
    cols={"team":"Equipo","conf":"Conferencia","division":"División","avg_wins":"Victorias promedio","pct_playoffs":"Playoffs (%)","pct_div_winner":"Ganar división (%)","pct_conf_champ":"Campeón conferencia (%)","pct_sb_winner":"Super Bowl (%)"}
    st.dataframe(probs.rename(columns=cols)[list(cols.values())].sort_values("Super Bowl (%)",ascending=False),hide_index=True,use_container_width=True)

elif section == "Partidos por semana":
    week=st.select_slider("Semana",options=sorted(games.week.unique()),value=7)
    selected=games[games.week==week].copy()
    selected["Partido"] = selected.away_team.map(fmt) + " @ " + selected.home_team.map(fmt)
    selected["Local (%)"]=(selected.p_home_pre*100).round(1)
    selected["Visitante (%)"]=(selected.p_away_pre*100).round(1)
    selected["Estado"]=selected.played.map({True:"Jugado",False:"Pendiente en CSV"})
    st.dataframe(selected[["Partido","Visitante (%)","Local (%)","Estado","away_score","home_score"]].rename(columns={"away_score":"Puntos visitante","home_score":"Puntos local"}),hide_index=True,use_container_width=True)
    if not selected.empty:
        match=st.selectbox("Ver enfrentamiento",selected["Partido"].tolist())
        m=selected[selected["Partido"]==match].iloc[0]
        st.subheader(match)
        x,y=st.columns(2)
        x.metric(f"{m.away_team} · visitante",f"{m.p_away_pre:.1%}")
        y.metric(f"{m.home_team} · local",f"{m.p_home_pre:.1%}")
        st.progress(float(m.p_home_pre),text="Probabilidad de victoria local")

elif section == "Detalle de equipos":
    team=st.selectbox("Equipo",probs.team.tolist(),format_func=fmt)
    row=probs[probs.team==team].iloc[0]
    a,b,c,dcol=st.columns(4)
    a.metric("Victorias esperadas",f"{row.avg_wins:.2f}")
    b.metric("Playoffs",f"{row.pct_playoffs:.1f}%")
    c.metric("Ganar división",f"{row.pct_div_winner:.1f}%")
    dcol.metric("Super Bowl",f"{row.pct_sb_winner:.1f}%")
    st.subheader("Evolución ELO")
    e=elos[elos.team==team].sort_values("week")
    st.line_chart(e.set_index("week")["elo_pre_week"],x_label="Semana",y_label="ELO previo")
    st.subheader("Calendario del equipo")
    tg=games[(games.home_team==team)|(games.away_team==team)].copy()
    tg["Rival"]=tg.apply(lambda r:r.away_team if r.home_team==team else r.home_team,axis=1)
    tg["Local/visitante"]=tg.home_team.apply(lambda h:"Local" if h==team else "Visitante")
    tg["Prob. victoria (%)"]=tg.apply(lambda r:round(100*(r.p_home_pre if r.home_team==team else r.p_away_pre),1),axis=1)
    st.dataframe(tg[["week","Rival","Local/visitante","Prob. victoria (%)","played"]].rename(columns={"week":"Semana","played":"Marcado como jugado"}),hide_index=True,use_container_width=True)

elif section == "Simulaciones de victorias":
    team=st.selectbox("Equipo",wins.columns.tolist(),format_func=fmt)
    st.subheader(f"Distribución de victorias · {names.get(team,team)}")
    counts=wins[team].value_counts().sort_index().rename("Cantidad de simulaciones")
    st.bar_chart(counts,x_label="Victorias",y_label="Simulaciones")
    a,b,c=st.columns(3)
    a.metric("Promedio",f"{wins[team].mean():.2f}")
    b.metric("Mínimo",int(wins[team].min()))
    c.metric("Máximo",int(wins[team].max()))
else:
    st.markdown("""**Fuente:** archivos CSV exportados del proyecto NFL Probability Engine. La interfaz no reentrena modelos ni consulta la NFL en vivo.\n\n**Probabilidades de partido:** columna `p_home_pre` del archivo de predicciones semanales.\n\n**Playoffs y Super Bowl:** porcentajes de `predictions_2026.csv`.\n\n**Simulaciones de victorias:** 10,000 filas de `wins_distribution_2026.csv`.\n\n**Cautela:** el campo `played` solo registra encuentros como jugados hasta la semana 6 del archivo cargado. Los pronósticos pueden estar desactualizados. No interpretar los resultados como una garantía de acierto.""")
    st.caption("Aplicación demostrativa, sin afiliación con la NFL.")
