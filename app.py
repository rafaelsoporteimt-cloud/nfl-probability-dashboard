from pathlib import Path
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="NFL Probability Engine | 2026", page_icon="🏈", layout="wide", initial_sidebar_state="expanded")
DATA = Path(__file__).resolve().parent / "data"
NAV = [
    ("Panorama general", "🏠"),
    ("Partidos por semana", "🏈"),
    ("Detalle de equipos", "🛡️"),
    ("Simulaciones de victorias", "📊"),
    ("Precisión del modelo", "🎯"),
    ("Metodología y datos", "📘"),
]
NAMES = {"ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens","BUF":"Buffalo Bills","CAR":"Carolina Panthers","CHI":"Chicago Bears","CIN":"Cincinnati Bengals","CLE":"Cleveland Browns","DAL":"Dallas Cowboys","DEN":"Denver Broncos","DET":"Detroit Lions","GB":"Green Bay Packers","HOU":"Houston Texans","IND":"Indianapolis Colts","JAX":"Jacksonville Jaguars","KC":"Kansas City Chiefs","LA":"Los Angeles Rams","LAC":"Los Angeles Chargers","LV":"Las Vegas Raiders","MIA":"Miami Dolphins","MIN":"Minnesota Vikings","NE":"New England Patriots","NO":"New Orleans Saints","NYG":"New York Giants","NYJ":"New York Jets","PHI":"Philadelphia Eagles","PIT":"Pittsburgh Steelers","SEA":"Seattle Seahawks","SF":"San Francisco 49ers","TB":"Tampa Bay Buccaneers","TEN":"Tennessee Titans","WAS":"Washington Commanders"}
fmt = lambda t: f"{t} · {NAMES.get(t,t)}"

st.markdown("""<style>
/* Brand system */
:root { --nfl-blue:#2369d8; }
.block-container { max-width: 1350px; padding-top:1.5rem; padding-bottom:4rem; }
h1,h2,h3 { letter-spacing:-0.028em; }
h1 { font-size:clamp(1.9rem,3.1vw,2.6rem)!important; }
[data-testid="stSidebar"] { border-right:1px solid rgba(120,140,170,.17); }
[data-testid="stSidebar"] [data-testid="stButton"] button {
    border-radius:12px!important; text-align:left!important; justify-content:flex-start!important;
    padding:.77rem .9rem!important; border:1px solid rgba(120,140,170,.16)!important;
    background:transparent; font-weight:600; width:100%; transition:all .12s ease;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover {border-color:#5495ed!important;background:rgba(64,135,228,.11)!important;}
[data-testid="stMetric"] { background:rgba(110,134,167,.07); border:1px solid rgba(120,140,170,.17); padding:15px 17px; border-radius:16px; min-width:0; }
[data-testid="stMetricLabel"] {font-size:.86rem;}
[data-testid="stMetricValue"] {font-size:clamp(1.35rem,2vw,2rem);}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:17px!important;}
[data-testid="stDataFrame"] {border-radius:12px;overflow:hidden;}
.stButton button[kind="primary"] {background:#256bd9; border-radius:12px;}
.hero {padding:1.15rem 1.3rem; border-radius:18px; background:linear-gradient(110deg,rgba(37,105,216,.17),rgba(23,159,163,.07)); border:1px solid rgba(69,139,234,.22); margin-bottom:1rem;}
.hero-kicker {color:#4c9cfc; font-weight:800; font-size:.75rem; letter-spacing:.10em;}
.hero-title {font-size:clamp(1.45rem,3vw,2.2rem); font-weight:800; line-height:1.23; margin:.35rem 0;}
.hero-sub {opacity:.77; font-size:.93rem;}
.mini-label {opacity:.76; font-size:.85rem; font-weight:600;}
@media (max-width: 740px) {
 .block-container { padding:1rem .78rem 3rem!important; }
 .hero {padding:1rem; border-radius:14px;}
 [data-testid="stMetric"] {padding:.8rem .7rem;}
 [data-testid="stMetricLabel"] {font-size:.73rem;}
 [data-testid="stMetricValue"] {font-size:1.45rem;}
 [data-testid="stHorizontalBlock"] {gap:.55rem;}
 [data-testid="stDataFrame"] {font-size:.8rem;}
}
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Cargando datos de la NFL...")
def load_data():
    required = ["predictions_2026", "weekly_predictions_2026", "elo_evolution_2026", "wins_distribution_2026", "elo_final_2025"]
    missing = [f"data/{n}.csv" for n in required if not (DATA / f"{n}.csv").exists()]
    if missing:
        raise FileNotFoundError("Faltan archivos: " + ", ".join(missing))
    return {name: pd.read_csv(DATA / f"{name}.csv") for name in required}

@st.cache_data(show_spinner=False)
def load_results():
    path = DATA / "resultados_reales_2026.csv"
    if not path.exists():
        raise FileNotFoundError("Falta data/resultados_reales_2026.csv")
    r = pd.read_csv(path)
    required = {"week", "home_team", "away_team", "home_score", "away_score"}
    if not required.issubset(r.columns):
        raise ValueError("Columnas faltantes en resultados_reales_2026.csv: " + str(sorted(required-set(r.columns))))
    if r.duplicated(["week", "home_team", "away_team"]).any():
        raise ValueError("Resultados reales duplicados")
    return r.dropna(subset=["home_score","away_score"]).rename(columns={"home_score":"home_score_real","away_score":"away_score_real"})

try:
    data = load_data()
    real = load_results()
    probs = data["predictions_2026"].copy()
    games = data["weekly_predictions_2026"].copy()
    games = games.merge(real, on=["week","home_team","away_team"], how="left",validate="one_to_one")
    match_count = int(games.home_score_real.notna().sum())
    if match_count != len(real):
        raise ValueError(f"Solo coinciden {match_count} de {len(real)} resultados adicionales")
    games["marcador_local"] = games.home_score_real.combine_first(games.home_score)
    games["marcador_visitante"] = games.away_score_real.combine_first(games.away_score)
    games["jugado_actual"] = games.marcador_local.notna() & games.marcador_visitante.notna()
    games["acierto_prediccion"] = pd.NA
    valid = games.jugado_actual & (games.marcador_local != games.marcador_visitante)
    games.loc[valid,"acierto_prediccion"] = (games.loc[valid,"p_home_pre"].ge(.5) == games.loc[valid,"marcador_local"].gt(games.loc[valid,"marcador_visitante"]))
    elos = data["elo_evolution_2026"].copy()
    wins = data["wins_distribution_2026"].copy()
except (FileNotFoundError, ValueError, KeyError, pd.errors.ParserError) as exc:
    st.error(f"No se pudieron cargar los datos: {exc}")
    st.info("Confirma que app.py y la carpeta data estén al mismo nivel en GitHub. No borres los CSV anteriores.")
    st.stop()

last_week = int(games.loc[games.jugado_actual,"week"].max()) if games.jugado_actual.any() else 0
next_week = min(last_week + 1, int(games.week.max()))
num_played = int(games.jugado_actual.sum())
num_pending = int((~games.jugado_actual).sum())
num_simulations = len(wins)

if "nav" not in st.session_state or st.session_state.nav not in [x[0] for x in NAV]:
    st.session_state.nav = NAV[0][0]

with st.sidebar:
    st.markdown("### 🏈 NFL ENGINE")
    st.caption("CENTRO DE ANÁLISIS · 2026")
    st.divider()
    for name, emoji in NAV:
        if st.button(f"{emoji}  {name}", key=f"nav_{name}", type="primary" if st.session_state.nav == name else "secondary", use_container_width=True):
            st.session_state.nav = name
            st.rerun()
    st.divider()
    st.caption(f"📍 Corte de resultados: semana {last_week}")
    st.caption(f"🎲 {num_simulations:,} temporadas simuladas")

# Easy navigation when sidebar is collapsed on small screens. Hidden on desktop by CSS.
st.markdown("""<style>
@media (min-width: 741px) { [class*="st-key-mobile_nav"] {display:none!important;} }
</style>""",unsafe_allow_html=True)
st.selectbox("Explorar secciones", [x[0] for x in NAV], index=[x[0] for x in NAV].index(st.session_state.nav), key="mobile_nav", on_change=lambda: st.session_state.update(nav=st.session_state.mobile_nav))

st.markdown(f"""<div class="hero"><div class="hero-kicker">NFL PROBABILITY ENGINE · TEMPORADA 2026</div>
<div class="hero-title">{st.session_state.nav}</div>
<div class="hero-sub">Datos reales hasta la semana {last_week} &nbsp;·&nbsp; {num_played} partidos disputados &nbsp;·&nbsp; {num_simulations:,} simulaciones</div></div>""",unsafe_allow_html=True)

sec = st.session_state.nav

def colored_chart(df, x, y, horizontal=False, percent=False, height=390):
    base = alt.Chart(df).mark_bar(cornerRadiusEnd=5,color="#4b99f2")
    if horizontal:
        chart=base.encode(y=alt.Y(f"{x}:N",sort="-x",title=None),x=alt.X(f"{y}:Q",title="Probabilidad (%)" if percent else None),tooltip=[x,alt.Tooltip(y,type="quantitative",format=".2f")])
    else:
        chart=base.encode(x=alt.X(f"{x}:O",title=None),y=alt.Y(f"{y}:Q",title="Porcentaje" if percent else None),tooltip=[x,alt.Tooltip(y,type="quantitative",format=".2f")])
    st.altair_chart(chart.properties(height=height),use_container_width=True)

def game_rows(frame):
    out=frame.copy()
    out["Partido"] = out.away_team.map(lambda t: NAMES.get(t,t)) + " @ " + out.home_team.map(lambda t: NAMES.get(t,t))
    out["Visitante (%)"]=(out.p_away_pre*100).round(1)
    out["Local (%)"]=(out.p_home_pre*100).round(1)
    out["Estado"]=out.jugado_actual.map({True:"Finalizado",False:"Pendiente"})
    return out

if sec == "Panorama general":
    a,b,c,d = st.columns(4)
    a.metric("Equipos",len(probs)); b.metric("Jugados",num_played); c.metric("Pendientes",num_pending); d.metric("Simulaciones",f"{num_simulations:,}")
    st.markdown("### 🏆 Favoritos al Super Bowl")
    st.caption("Probabilidades de la simulación, no garantías de victoria.")
    ranked=probs.sort_values("pct_sb_winner",ascending=False).head(12).copy()
    ranked["Equipo"]=ranked.team.map(lambda t: NAMES.get(t,t))
    colored_chart(ranked, "Equipo", "pct_sb_winner",horizontal=True,percent=True, height=430)
    st.markdown("### 📋 Probabilidades por equipo")
    table=probs.sort_values("pct_sb_winner",ascending=False).copy()
    table["Equipo"]=table.team.map(fmt)
    table=table.rename(columns={"avg_wins":"Victorias esperadas","pct_playoffs":"Playoffs (%)","pct_div_winner":"División (%)","pct_conf_champ":"Conferencia (%)","pct_sb_winner":"Super Bowl (%)"})
    st.dataframe(table[["Equipo","Victorias esperadas","Playoffs (%)","División (%)","Conferencia (%)","Super Bowl (%)"]],hide_index=True,use_container_width=True)

elif sec == "Partidos por semana":
    weeks=sorted(games.week.unique().tolist())
    week=st.selectbox("📅 Selecciona la semana",weeks,index=weeks.index(next_week) if next_week in weeks else 0)
    selected=game_rows(games[games.week==week])
    played=int(selected.jugado_actual.sum())
    x,y,z=st.columns(3)
    x.metric("Partidos",len(selected));y.metric("Finalizados",played);z.metric("Por jugar",len(selected)-played)
    st.markdown("### Enfrentamientos")
    # Simple readable list for phones, detailed table on bigger screens
    st.dataframe(selected[["Partido","Visitante (%)","Local (%)","Estado"]],hide_index=True,use_container_width=True)
    if not selected.empty:
        keys=selected.index.tolist()
        key=st.selectbox("🔍 Detalle de partido",keys,format_func=lambda idx: f"{selected.loc[idx,'away_team']} @ {selected.loc[idx,'home_team']}")
        m=selected.loc[key]
        with st.container(border=True):
            st.markdown(f"#### {NAMES.get(m.away_team,m.away_team)} vs {NAMES.get(m.home_team,m.home_team)}")
            left,right=st.columns(2)
            left.metric("Visitante",f"{m.p_away_pre:.1%}")
            right.metric("Local",f"{m.p_home_pre:.1%}")
            st.progress(float(m.p_home_pre),text=f"Probabilidad de victoria local: {m.p_home_pre:.1%}")
            st.caption(f"Favorito ELO: {NAMES.get(m.home_team if m.p_home_pre>=.5 else m.away_team,'equipo')}")
            if bool(m.jugado_actual):
                st.success(f"Final: {m.away_team} {int(m.marcador_visitante)} – {int(m.marcador_local)} {m.home_team}")
            else:
                st.info("Partido pendiente. Porcentajes estimados antes del encuentro.")

elif sec == "Detalle de equipos":
    team=st.selectbox("🛡️ Selecciona un equipo", sorted(probs.team),format_func=fmt)
    row=probs.loc[probs.team==team].iloc[0]
    a,b,c,d=st.columns(4)
    a.metric("Victorias esperadas",f"{row.avg_wins:.2f}")
    b.metric("Playoffs",f"{row.pct_playoffs:.1f}%")
    c.metric("División",f"{row.pct_div_winner:.1f}%")
    d.metric("Super Bowl",f"{row.pct_sb_winner:.1f}%")
    st.markdown("### 📈 Evolución ELO")
    e=elos[(elos.team==team)&(elos.week<=last_week+1)].sort_values("week")
    if not e.empty:
        st.line_chart(e.set_index("week")["elo_pre_week"],x_label="Semana",y_label="ELO previo")
    st.markdown("### 📅 Calendario del equipo")
    tg=games[(games.home_team==team)|(games.away_team==team)].copy()
    tg["Rival"]=tg.apply(lambda r: fmt(r.away_team if r.home_team==team else r.home_team),axis=1)
    tg["Condición"]=tg.home_team.apply(lambda h:"Local" if h==team else "Visitante")
    tg["Prob. victoria (%)"]=tg.apply(lambda r:round(100*(r.p_home_pre if r.home_team==team else r.p_away_pre),1),axis=1)
    tg["Estado"]=tg.jugado_actual.map({True:"Finalizado",False:"Pendiente"})
    st.dataframe(tg[["week","Rival","Condición","Prob. victoria (%)","Estado"]].rename(columns={"week":"Semana"}),hide_index=True,use_container_width=True)

elif sec == "Simulaciones de victorias":
    team=st.selectbox("🎲 Selecciona un equipo", sorted(wins.columns),format_func=fmt)
    a,b,c=st.columns(3)
    a.metric("Promedio",f"{wins[team].mean():.2f}")
    b.metric("Mínimo",int(wins[team].min()))
    c.metric("Máximo",int(wins[team].max()))
    st.markdown(f"### Distribución de victorias · {NAMES.get(team,team)}")
    counts=wins[team].value_counts().sort_index().rename_axis("Victorias").reset_index(name="Simulaciones")
    colored_chart(counts,"Victorias","Simulaciones",height=375)
    st.caption("Frecuencia de victorias en las temporadas simuladas; no es una garantía de resultados futuros.")

elif sec == "Precisión del modelo":
    st.caption("Evaluación retrospectiva: se utilizan exclusivamente pronósticos guardados antes del partido.")
    history=sorted(DATA.glob("pronosticos_semana_*.csv"))
    frames=[]
    necessary={"week","home_team","away_team","p_home_pre","p_away_pre"}
    for path in history:
        p=pd.read_csv(path)
        if necessary.issubset(p.columns):
            frames.append(p[list(sorted(necessary))].copy())
        else:
            st.warning(f"Se omitió {path.name}: faltan columnas.")
    if not frames:
        st.info("Guarda los pronósticos originales como data/pronosticos_semana_N.csv para evaluar el modelo.")
    else:
        hist=pd.concat(frames,ignore_index=True)
        keys=["week","home_team","away_team"]
        if hist.duplicated(keys).any():
            st.error("Hay partidos duplicados entre los archivos históricos. Revisa los CSV.")
            st.stop()
        hist=hist.merge(games[keys+["jugado_actual","marcador_local","marcador_visitante"]],on=keys,how="left",validate="one_to_one")
        hist["jugado_actual"]=hist.jugado_actual.fillna(False).astype(bool)
        ok=hist.jugado_actual & hist.marcador_local.notna() & hist.marcador_visitante.notna() & hist.marcador_local.ne(hist.marcador_visitante)
        scored=hist.loc[ok].copy()
        pending=len(hist)-len(scored)
        if scored.empty:
            st.info("Aún no hay marcadores finales para los pronósticos guardados.")
        else:
            scored["p_home_pre"]=pd.to_numeric(scored.p_home_pre,errors="raise")
            scored["p_away_pre"]=pd.to_numeric(scored.p_away_pre,errors="raise")
            if not (scored.p_home_pre.between(0,1)&((scored.p_home_pre+scored.p_away_pre-1).abs()<.001)).all():
                st.error("Probabilidades históricas fuera de rango o que no suman 100%.")
                st.stop()
            scored["gano_local"]=scored.marcador_local.gt(scored.marcador_visitante)
            scored["favorito_local"]=scored.p_home_pre.ge(.5)
            scored["acierto"]=scored.gano_local.eq(scored.favorito_local)
            scored["brier"]=(scored.p_home_pre-scored.gano_local.astype(int))**2
            scored["favorito"]=scored.apply(lambda r:r.home_team if r.favorito_local else r.away_team,axis=1)
            scored["ganador"]=scored.apply(lambda r:r.home_team if r.gano_local else r.away_team,axis=1)
            scored["Confianza (%)"]=(scored[["p_home_pre","p_away_pre"]].max(axis=1)*100).round(1)
            aciertos=int(scored.acierto.sum()); total=len(scored)
            a,b,c,d=st.columns(4)
            a.metric("Partidos evaluados",total)
            b.metric("Aciertos",f"{aciertos}/{total}")
            c.metric("Precisión",f"{100*aciertos/total:.1f}%")
            d.metric("Brier Score",f"{scored.brier.mean():.4f}")
            st.caption("Brier Score para victoria local (0 es mejor). La precisión no mide por sí sola la calibración.")
            by_week=scored.groupby("week").agg(Partidos=("acierto","size"),Aciertos=("acierto","sum"),Precisión=("acierto",lambda x:x.mean()*100),Brier=("brier","mean")).reset_index().sort_values("week")
            st.markdown("### 📈 Rendimiento por semana")
            colored_chart(by_week.rename(columns={"week":"Semana"}),"Semana","Precisión",percent=True,height=290)
            st.dataframe(by_week.rename(columns={"week":"Semana","Brier":"Brier Score","Precisión":"Precisión (%)"}),hide_index=True,use_container_width=True)
            week=st.selectbox("🔍 Revisar una semana",sorted(scored.week.unique(),reverse=True))
            detail=scored.loc[scored.week==week].copy()
            detail["Partido"]=detail.away_team.map(fmt)+" @ "+detail.home_team.map(fmt)
            detail["Favorito"]=detail.favorito.map(fmt)
            detail["Ganador"]=detail.ganador.map(fmt)
            detail["Resultado"]=detail.acierto.map({True:"✅ Acierto",False:"❌ Error"})
            detail["Marcador"]=detail.apply(lambda r:f"{int(r.marcador_visitante)} - {int(r.marcador_local)}",axis=1)
            st.dataframe(detail[["Partido","Favorito","Confianza (%)","Ganador","Marcador","Resultado"]],hide_index=True,use_container_width=True)
        if pending:
            st.info(f"{pending} pronóstico(s) guardado(s) pendientes de marcador final. No se incluyen en las métricas.")

else:
    st.markdown("""### Cómo funciona el modelo
**ELO:** K=40, ventaja local=42.2. Se conserva el histórico inicial y se actualiza con los marcadores reales posteriores.

**Monte Carlo:** se simulan los encuentros pendientes 10,000 veces y se actualizan los ratings en cada temporada simulada. Postemporada: K=48, Super Bowl neutral.

**Precisión:** se compara cada pronóstico guardado antes de jugar con el marcador definitivo. Se utiliza Brier Score para cuantificar el error de probabilidad.

**Limitaciones:** desempates aproximados. Este sistema no reproduce todos los criterios oficiales de clasificación de la NFL ni garantiza victorias. Es una adaptación del proyecto original.

**Archivos de datos:** `predictions_2026.csv`, `weekly_predictions_2026.csv`, `elo_evolution_2026.csv`, `wins_distribution_2026.csv`, `resultados_reales_2026.csv` y `pronosticos_semana_N.csv`.

Este proyecto es demostrativo y no está afiliado con la NFL.""")

st.divider()
st.caption(f"NFL Probability Engine · Datos con corte en semana {last_week} · Interfaz adaptable a computadora y celular")
