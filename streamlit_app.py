from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cattle_collar_core import (
    build_cash_flow_table,
    build_report,
    build_sensitivity_table,
    check_connectivity,
    compute_switch_stay_economics,
    estimate_fence_repair_material,
    fetch_nasa_power_solar_proxy,
    generate_user_profile_population,
    make_payback_explanation,
    monte_carlo_uncertainty,
    safe_json_dumps,
    simulate_bandit_learning,
    statistics_alarm_system,
    train_fence_cost_model,
)

APP_TITLE = "Cattle Collar Switch-or-Stay AI"
AUTHOR = "Sykes Lamensdorf"
ADVISOR = "Dr. Qingyang Xiao"
GITHUB_URL = "https://github.com/qxiao2ub/Cattle_Collar_AI_Decision_App"
APP_URL = "https://cattle-collar-ai-decision.streamlit.app/"
MC_DRAWS = 800

# Defaults taken from the supplied ranch-oriented UI. They are editable demo
# assumptions, not verified vendor quotes.
PATH_PRESETS: Dict[str, Dict[str, float]] = {
    "cell": {
        "hardware_per_head": 279.0,
        "subscription_per_head_month": 3.5,
        "tower_upfront": 0.0,
        "install_setup": 2500.0,
        "tower_maintenance_year": 0.0,
        "platform_fee_year": 1200.0,
        "vf_management_hours_month": 6.0,
        "training_hours": 40.0,
        "replacement_rate_pct": 6.0,
    },
    "tower": {
        "hardware_per_head": 249.0,
        "subscription_per_head_month": 2.5,
        "tower_upfront": 18000.0,
        "install_setup": 4500.0,
        "tower_maintenance_year": 1500.0,
        "platform_fee_year": 1000.0,
        "vf_management_hours_month": 8.0,
        "training_hours": 48.0,
        "replacement_rate_pct": 6.0,
    },
    "satellite": {
        "hardware_per_head": 349.0,
        "subscription_per_head_month": 6.0,
        "tower_upfront": 0.0,
        "install_setup": 3000.0,
        "tower_maintenance_year": 0.0,
        "platform_fee_year": 1500.0,
        "vf_management_hours_month": 6.0,
        "training_hours": 40.0,
        "replacement_rate_pct": 7.0,
    },
}

PATH_LABELS = {
    "cell": "Cell collars",
    "tower": "Ranch tower",
    "satellite": "Satellite collars",
}

LAND_LABELS = {
    "owned": "Owned",
    "leased": "Leased",
    "mixed": "Mixed",
    "allotment": "Public allotment",
}

SECTIONS = [
    "Ranch data",
    "Current analog labor",
    "Connectivity gate",
    "Virtual-fence cost assumptions",
    "Finance & optional upside",
    "Review",
]

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Karla:wght@400;500;600;700&display=swap');

:root {
  --ranch-bg: #f6f1e7;
  --ranch-card: #fffdf8;
  --ranch-text: #40392f;
  --ranch-muted: #786f63;
  --ranch-green: #4c7958;
  --ranch-green-dark: #315b3e;
  --ranch-sand: #eee5d6;
  --ranch-orange: #b87038;
  --ranch-red: #9f4338;
  --ranch-border: #ded2c1;
}

html, body, [class*="css"] { font-family: 'Karla', ui-sans-serif, system-ui, sans-serif; }
.stApp { background: var(--ranch-bg); color: var(--ranch-text); }
.block-container { max-width: 1120px; padding-top: 1.6rem; padding-bottom: 4rem; }
h1, h2, h3, .ranch-display { font-family: 'Fraunces', Georgia, serif !important; letter-spacing: -0.015em; }

/* Streamlit chrome */
[data-testid="stHeader"] { background: rgba(246,241,231,0.86); }
[data-testid="stToolbar"] { right: 1rem; }
[data-testid="stSidebarCollapsedControl"] { color: var(--ranch-text); }

.ranch-topbar {
  border: 1px solid var(--ranch-border); background: rgba(255,253,248,.82);
  border-radius: 18px; padding: 16px 20px; margin-bottom: 26px;
  display: flex; justify-content: space-between; align-items: center; gap: 18px;
}
.ranch-kicker { color: var(--ranch-orange); font-size: .76rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
.ranch-title { font-family: 'Fraunces', Georgia, serif; color: var(--ranch-text); font-size: 1.85rem; line-height: 1.1; font-weight: 700; }
.ranch-pill { border: 1px solid var(--ranch-border); background: #fffdf8; border-radius: 999px; padding: 7px 12px; color: var(--ranch-muted); font-size: .78rem; white-space: nowrap; }

.hero-kicker { color: var(--ranch-orange); font-weight: 800; letter-spacing: .16em; text-transform: uppercase; font-size: .78rem; }
.hero-title { font-family: 'Fraunces', Georgia, serif; font-size: clamp(2.45rem, 5vw, 4.3rem); line-height: 1.04; color: var(--ranch-text); margin: .45rem 0 1rem; font-weight: 700; }
.hero-copy { font-size: 1.14rem; line-height: 1.62; color: var(--ranch-muted); max-width: 650px; }
.small-muted { color: var(--ranch-muted); font-size: .86rem; }

.check-card, .info-card, .credit-card, .verdict-card, .metric-card {
  background: var(--ranch-card); border: 1px solid var(--ranch-border); border-radius: 18px;
}
.check-card { padding: 22px; box-shadow: 0 6px 20px rgba(67,54,40,.05); }
.check-row { display: grid; grid-template-columns: 36px 1fr; gap: 12px; margin: 0 0 18px; }
.check-row:last-child { margin-bottom: 0; }
.check-icon { width: 30px; height: 30px; border-radius: 50%; background: #e6efe8; color: var(--ranch-green-dark); display:flex; align-items:center; justify-content:center; font-weight:800; }
.check-title { font-family: 'Fraunces', Georgia, serif; font-weight: 700; font-size: 1.02rem; margin-bottom: 3px; }
.check-copy { color: var(--ranch-muted); font-size: .88rem; line-height:1.45; }

.metric-card { padding: 18px; height: 100%; background: rgba(238,229,214,.44); }
.metric-number { font-family: 'Fraunces', Georgia, serif; color: var(--ranch-green-dark); font-size: 2.1rem; font-weight: 700; line-height: 1; }
.metric-label { font-weight: 700; margin-top: 6px; }
.metric-copy { color: var(--ranch-muted); font-size: .84rem; margin-top: 4px; line-height:1.4; }

.step-wrap { background: var(--ranch-card); border:1px solid var(--ranch-border); border-radius:18px; padding:14px 18px; margin-bottom:18px; }
.step-head { display:flex; justify-content:space-between; gap:12px; align-items:center; margin-bottom:12px; }
.step-label { font-weight:700; }
.step-muted { color:var(--ranch-muted); font-size:.82rem; }
.step-track { display:grid; grid-template-columns:repeat(6,1fr); gap:6px; }
.step-seg { height:7px; border-radius:999px; background:#e4dacc; }
.step-seg.done { background: var(--ranch-green); }
.step-seg.current { background: var(--ranch-orange); }

[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--ranch-card); border-color: var(--ranch-border) !important; border-radius: 18px !important;
  box-shadow: 0 5px 16px rgba(67,54,40,.035);
}
[data-testid="stMetric"] { background: rgba(238,229,214,.42); border: 1px solid var(--ranch-border); border-radius: 14px; padding: 12px 14px; }
[data-testid="stMetricLabel"] { color: var(--ranch-muted); }
[data-testid="stMetricValue"] { font-family:'Fraunces', Georgia, serif; color:var(--ranch-text); }

.stButton > button, .stDownloadButton > button, .stLinkButton > a {
  border-radius: 10px; min-height: 42px; font-weight: 700; border: 1px solid var(--ranch-border);
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] { background: var(--ranch-green); color: white; border-color: var(--ranch-green); }
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover { background: var(--ranch-green-dark); border-color:var(--ranch-green-dark); }

input, textarea, [data-baseweb="select"] > div { border-radius: 9px !important; }
[data-testid="stAlert"] { border-radius: 12px; }

.verdict-card { padding: 22px; border-left: 8px solid var(--ranch-green); }
.verdict-card.pilot { border-left-color: var(--ranch-orange); }
.verdict-card.stay, .verdict-card.blocked { border-left-color: var(--ranch-red); }
.verdict-kicker { font-size:.76rem; color:var(--ranch-muted); text-transform:uppercase; font-weight:800; letter-spacing:.12em; }
.verdict-title { font-family:'Fraunces', Georgia, serif; font-size:2.35rem; font-weight:700; margin:.2rem 0 .35rem; }
.verdict-copy { color:var(--ranch-muted); line-height:1.55; }

.note-box { padding: 12px 14px; border:1px solid var(--ranch-border); border-radius:10px; background:rgba(238,229,214,.38); color:var(--ranch-muted); font-size:.82rem; line-height:1.45; }
.demo-badge { display:inline-block; border:1px solid #d49a63; color:#845023; background:#faecd9; padding:3px 8px; border-radius:999px; font-size:.66rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
.footer-box { margin-top: 36px; padding-top:18px; border-top:1px solid var(--ranch-border); color:var(--ranch-muted); font-size:.84rem; }
.footer-box strong { color:var(--ranch-text); }

@media (max-width: 700px) {
  .ranch-topbar { align-items:flex-start; flex-direction:column; }
  .ranch-pill { white-space:normal; }
  .step-track { gap:3px; }
}
</style>
""",
    unsafe_allow_html=True,
)


def money(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def pct(value: float) -> str:
    return f"{100.0 * float(value):.0f}%"


def init_state() -> None:
    defaults: Dict[str, Any] = {
        "view": "home",
        "step": 0,
        "generated": False,
        "result_tab": "The answer",
        "ranch_name": "Home Place",
        "land_status": "owned",
        "acres": 4000.0,
        "head_cattle": 300,
        "fence_miles": 26.0,
        "repair_hours_per_month": 18.0,
        "moving_hours_per_month": 22.0,
        "labor_rate": 28.0,
        "fence_material_per_mile_year": 320.0,
        "cell_coverage_pct": 55.0,
        "tower_feasible": True,
        "satellite_feasible": True,
        "preferred_path": "cell",
        "cost_share_pct": 0.0,
        "fixed_grant": 0.0,
        "discount_rate_pct": 8.0,
        "horizon_years": 5,
        "target_payback_years": 3.0,
        "capital_available": 60000.0,
        "conservation_upside_year": 0.0,
        "grazing_upside_year": 0.0,
        "latitude": 38.5,
        "longitude": -106.0,
        "advisor_output": "",
        "advisor_title": "",
    }
    defaults.update(PATH_PRESETS["cell"])
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_path_preset() -> None:
    path = st.session_state.preferred_path
    for key, value in PATH_PRESETS[path].items():
        st.session_state[key] = value


def reset_results() -> None:
    st.session_state.generated = False
    st.session_state.result_tab = "The answer"


def start_assessment() -> None:
    st.session_state.view = "assess"
    st.session_state.step = 0
    reset_results()


def go_home() -> None:
    st.session_state.view = "home"
    reset_results()


def build_inputs() -> Dict[str, Any]:
    path = st.session_state.preferred_path
    return {
        "location_name": st.session_state.ranch_name,
        "owned_or_leased": LAND_LABELS[st.session_state.land_status],
        "acres": float(st.session_state.acres),
        "head_cattle": int(st.session_state.head_cattle),
        "fence_miles": float(st.session_state.fence_miles),
        "repair_hours_per_month": float(st.session_state.repair_hours_per_month),
        "moving_hours_per_month": float(st.session_state.moving_hours_per_month),
        "labor_rate": float(st.session_state.labor_rate),
        "fence_repair_material_per_mile_annual": float(st.session_state.fence_material_per_mile_year),
        "virtual_labor_hours_per_month": float(st.session_state.vf_management_hours_month),
        "collar_hardware_cost": float(st.session_state.hardware_per_head),
        "subscription_per_head_month": float(st.session_state.subscription_per_head_month),
        "ranch_platform_fee_annual": float(st.session_state.platform_fee_year),
        "base_station_cost": float(st.session_state.tower_upfront) if path == "tower" else 0.0,
        "install_cost": float(st.session_state.install_setup),
        "training_hours": float(st.session_state.training_hours),
        "collar_replacement_rate": float(st.session_state.replacement_rate_pct) / 100.0,
        "tower_maintenance_annual": float(st.session_state.tower_maintenance_year) if path == "tower" else 0.0,
        "cost_share_rate": float(st.session_state.cost_share_pct) / 100.0,
        "fixed_cost_share": float(st.session_state.fixed_grant),
        "discount_rate": float(st.session_state.discount_rate_pct) / 100.0,
        "horizon_years": int(st.session_state.horizon_years),
        "target_payback_years": float(st.session_state.target_payback_years),
        "system_type": path,
        "optional_conservation_value_annual": float(st.session_state.conservation_upside_year),
        "optional_grazing_gain_annual": float(st.session_state.grazing_upside_year),
        "capital_available": float(st.session_state.capital_available),
        "cell_coverage_pct": float(st.session_state.cell_coverage_pct),
    }


def connectivity_result() -> Dict[str, Any]:
    return check_connectivity(
        cell_available=float(st.session_state.cell_coverage_pct) >= 40.0,
        tower_possible=bool(st.session_state.tower_feasible),
        satellite_available=bool(st.session_state.satellite_feasible),
        preferred_path=st.session_state.preferred_path,
    )


def connectivity_score(conn: Dict[str, Any]) -> int:
    score = min(50.0, float(st.session_state.cell_coverage_pct) * 0.5)
    score += 25.0 if st.session_state.tower_feasible else 0.0
    score += 15.0 if st.session_state.satellite_feasible else 0.0
    score += 10.0 if conn.get("preferred_viable", False) else 0.0
    return int(round(max(0.0, min(100.0, score))))


def compute_results() -> Dict[str, Any]:
    inputs = build_inputs()
    conn = connectivity_result()
    chosen_path_passes = bool(conn.get("viable")) and bool(conn.get("preferred_viable"))
    score = connectivity_score(conn)
    if not chosen_path_passes:
        return {
            "inputs": inputs,
            "connectivity": conn,
            "score": score,
            "gate_passes": False,
            "economics": None,
            "sensitivity": pd.DataFrame(),
            "monte_carlo": pd.DataFrame(),
            "alarms": statistics_alarm_system(inputs, connectivity=conn),
        }

    econ = compute_switch_stay_economics(inputs)
    sensitivity = build_sensitivity_table(inputs, pct=0.25)
    mc = monte_carlo_uncertainty(inputs, n=MC_DRAWS, seed=20260908)
    alarms = statistics_alarm_system(inputs, econ, conn, mc, sensitivity)
    return {
        "inputs": inputs,
        "connectivity": conn,
        "score": score,
        "gate_passes": True,
        "economics": econ,
        "sensitivity": sensitivity,
        "monte_carlo": mc,
        "alarms": alarms,
    }


def verdict_for(results: Dict[str, Any]) -> Tuple[str, str, str]:
    if not results["gate_passes"]:
        conn = results["connectivity"]
        if conn.get("viable", False):
            return (
                "blocked",
                "Chosen path is not viable",
                f"The preferred {PATH_LABELS[st.session_state.preferred_path].lower()} path does not pass the connectivity gate. Choose the recommended fallback ({conn.get('recommended_path')}) before running the economics.",
            )
        return (
            "blocked",
            "No viable path today",
            "Connectivity is the hard gate. With no workable cell, tower, or satellite path, the app intentionally skips the ROI math.",
        )

    econ = results["economics"]
    mc = results["monte_carlo"]
    prob_positive = float((mc["npv"] > 0).mean()) if len(mc) else 0.0
    payback = float(econ["payback_years"])
    payback_ok = math.isfinite(payback) and payback <= float(econ["target_payback_years"])
    if float(econ["npv"]) > 0 and payback_ok and prob_positive >= 0.70:
        return "switch", "Switch", "The hard-dollar case survives the target payback test and holds up across most simulated seasons. Plan a staged rollout and validate the signal map before full deployment."
    if float(econ["npv"]) > 0 and prob_positive >= 0.45:
        return "pilot", "Pilot first", "The economics are promising but still depend on assumptions that should be proven on your ranch. Collar one herd for a season, record the actual labor and replacement rates, then re-run the model."
    return "stay", "Stay", "On today's hard-dollar assumptions, the collars do not create a strong enough risk-adjusted return. Keep the current system and revisit after pricing, cost-share, or labor conditions change."


def render_topbar(show_back: bool = False) -> None:
    st.markdown(
        f"""
<div class="ranch-topbar">
  <div>
    <div class="ranch-kicker">Switch-or-Stay</div>
    <div class="ranch-title">Cattle Collar Decision</div>
  </div>
  <div class="ranch-pill">Author: <strong>{AUTHOR}</strong> &nbsp;•&nbsp; Advisor: <strong>{ADVISOR}</strong></div>
</div>
""",
        unsafe_allow_html=True,
    )
    if show_back:
        cols = st.columns([1, 5])
        with cols[0]:
            if st.button("← Home", use_container_width=True):
                go_home()
                st.rerun()


def render_footer() -> None:
    st.markdown(
        f"""
<div class="footer-box">
  <strong>{APP_TITLE}</strong><br/>
  Author: <strong>{AUTHOR}</strong> &nbsp;•&nbsp; Advisor: <strong>{ADVISOR}</strong><br/>
  Open-source educational decision-support MVP under the MIT License. Ranch inputs are session-only in this implementation; no application database is included.<br/>
  <a href="{GITHUB_URL}" target="_blank">GitHub repository</a> &nbsp;•&nbsp; <a href="{APP_URL}" target="_blank">Live app</a>
</div>
""",
        unsafe_allow_html=True,
    )


def render_home() -> None:
    render_topbar()
    left, right = st.columns([1.08, 0.92], gap="large", vertical_alignment="center")
    with left:
        st.markdown('<div class="hero-kicker">Ranch decision tool</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">Should you collar the herd, or stay with what you know?</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-copy">Walk through your ranch numbers and get a clear <strong>Switch, Pilot, or Stay</strong> answer — beginning with the hard question of whether your ground can actually support the required connectivity.</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Start the ranch assessment  →", type="primary", use_container_width=True):
            start_assessment()
            st.rerun()
        st.markdown('<div class="small-muted">No account required • session-only inputs • editable assumptions</div>', unsafe_allow_html=True)

    with right:
        st.markdown(
            """
<div class="check-card">
  <div class="check-row"><div class="check-icon">✓</div><div><div class="check-title">Signal-first gate</div><div class="check-copy">No amount of savings matters if collars cannot communicate. The workflow stops cleanly when the selected path is not viable.</div></div></div>
  <div class="check-row"><div class="check-icon">✓</div><div><div class="check-title">Your quotes, your numbers</div><div class="check-copy">Edit collar prices, subscriptions, base stations, labor, repair cost, cost-share, capital, and optional upside. Defaults are placeholders.</div></div></div>
  <div class="check-row"><div class="check-icon">✓</div><div><div class="check-title">Risk you can see</div><div class="check-copy">Sensitivity sweeps, 800-run uncertainty simulation, cash-flow curves, ML estimates, RL experiments, and statistical alarms make the answer inspectable.</div></div></div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.write("")
    cards = st.columns(3, gap="medium")
    card_data = [
        ("6", "Guided stages", "From ranch profile and connectivity to review and results."),
        ("800", "Simulated seasons", "Monte Carlo uncertainty around costs, labor, and cost-share assumptions."),
        ("±25%", "Sensitivity sweep", "See which assumptions move the discounted value the most."),
    ]
    for col, (number, label, copy) in zip(cards, card_data):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-number">{number}</div><div class="metric-label">{label}</div><div class="metric-copy">{copy}</div></div>', unsafe_allow_html=True)

    st.write("")
    with st.container(border=True):
        st.markdown("### What the app evaluates")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**1. Ranch reality**\n\nAcres, head, physical fence, repair labor, cattle-moving labor, and ownership/lease status.")
        c2.markdown("**2. Connectivity + economics**\n\nCell, tower, or satellite viability followed by transparent switch/stay cash-flow and payback math.")
        c3.markdown("**3. AI + uncertainty**\n\nML cost estimation, RL learning simulation, sensitivity, Monte Carlo risk, and statistical alarms.")

    render_footer()


def render_progress() -> None:
    step = int(st.session_state.step)
    segs = []
    for i in range(len(SECTIONS)):
        cls = "done" if i < step else "current" if i == step else ""
        segs.append(f'<div class="step-seg {cls}"></div>')
    st.markdown(
        f"""
<div class="step-wrap">
  <div class="step-head"><div class="step-label">{'Review & edit' if step == 5 else f'Step {step + 1} of 5'} — {SECTIONS[step]}</div><div class="step-muted">Guided ranch assessment</div></div>
  <div class="step-track">{''.join(segs)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def nav_buttons() -> None:
    step = int(st.session_state.step)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", disabled=step == 0, use_container_width=True):
            st.session_state.step = max(0, step - 1)
            st.rerun()
    with c2:
        label = "Review my answers" if step == 4 else "Next"
        if st.button(label, type="primary", disabled=step >= 5, use_container_width=True):
            st.session_state.step = min(5, step + 1)
            st.rerun()


def render_step_ranch() -> None:
    with st.container(border=True):
        st.markdown("## Ranch data")
        st.caption("Rough numbers are fine. You can edit everything again before the model runs.")
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Ranch / scenario name", key="ranch_name")
            st.number_input("Owned or managed acres", min_value=0.0, step=50.0, key="acres")
            st.number_input("Existing fence miles", min_value=0.0, step=1.0, key="fence_miles")
        with c2:
            st.radio("Land status", options=list(LAND_LABELS), format_func=lambda x: LAND_LABELS[x], horizontal=True, key="land_status")
            st.number_input("Head needing collars", min_value=0, step=1, key="head_cattle")
            st.info("These fields identify the ranch scale and the physical fencing burden the virtual-fence option must compete against.")
    nav_buttons()


def render_step_labor() -> None:
    with st.container(border=True):
        st.markdown("## Current analog labor")
        st.caption("What the fence-and-horseback system costs today in time and repair material.")
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Fence repair hours per month", min_value=0.0, step=1.0, key="repair_hours_per_month")
            st.number_input("Loaded labor rate ($/hr)", min_value=0.0, step=1.0, key="labor_rate", help="Wage plus payroll costs, fuel, equipment, and other loaded labor burden.")
        with c2:
            st.number_input("Cattle-moving hours per month", min_value=0.0, step=1.0, key="moving_hours_per_month")
            st.number_input("Annual repair material per mile ($/mile/year)", min_value=0.0, step=10.0, key="fence_material_per_mile_year")
        st.markdown('<div class="note-box"><strong>Transparent arithmetic:</strong> the headline model compares the annual analog burden you enter against annual virtual-fence subscription, management, replacement, and infrastructure costs. Optional conservation and grazing upside remain separately labeled.</div>', unsafe_allow_html=True)
    nav_buttons()


def render_step_connectivity() -> None:
    with st.container(border=True):
        st.markdown("## Connectivity gate")
        st.caption("This is the hard gate. If the selected collar path cannot communicate reliably, the app will not pretend the economics are actionable.")
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Useful-pasture cell coverage (%)", min_value=0.0, max_value=100.0, step=5.0, key="cell_coverage_pct")
            st.toggle("Ranch tower / base station feasible", key="tower_feasible")
        with c2:
            st.toggle("Satellite collar path feasible", key="satellite_feasible")
            st.radio("Preferred virtual-fence path", list(PATH_LABELS), format_func=lambda x: PATH_LABELS[x], horizontal=True, key="preferred_path", on_change=apply_path_preset)
        st.markdown('<div class="note-box">Cell is treated as viable when useful-pasture coverage is at least 40%. Choosing a path loads the attached UI\'s editable starter cost assumptions for that path.</div>', unsafe_allow_html=True)
    nav_buttons()


def render_step_costs() -> None:
    with st.container(border=True):
        st.markdown("## Virtual-fence cost assumptions")
        st.caption("Replace these starter defaults with the quote in front of you. No value below should be treated as a current vendor price.")
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Hardware per head ($)", min_value=0.0, step=10.0, key="hardware_per_head")
            st.number_input("Tower upfront ($)", min_value=0.0, step=500.0, key="tower_upfront", help="Only included in the economics for the tower path.")
            st.number_input("Tower annual maintenance ($/yr)", min_value=0.0, step=100.0, key="tower_maintenance_year")
            st.number_input("Virtual-fence management labor (hrs/mo)", min_value=0.0, step=1.0, key="vf_management_hours_month")
            st.number_input("Collar replacement rate (%/yr)", min_value=0.0, max_value=100.0, step=1.0, key="replacement_rate_pct")
        with c2:
            st.number_input("Subscription per head per month ($)", min_value=0.0, step=0.25, key="subscription_per_head_month")
            st.number_input("Install & setup ($)", min_value=0.0, step=100.0, key="install_setup")
            st.number_input("Ranch platform fee ($/yr)", min_value=0.0, step=100.0, key="platform_fee_year")
            st.number_input("Initial training labor (hours)", min_value=0.0, step=1.0, key="training_hours")
        if st.button("Reload defaults for selected path"):
            apply_path_preset()
            st.rerun()
    nav_buttons()


def render_step_finance() -> None:
    with st.container(border=True):
        st.markdown("## Finance and optional upside")
        st.caption("Cost-share, grants, capital limits, and analysis horizon. Leave soft-benefit fields at zero if you do not want them shown.")
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Cost-share rate (%)", min_value=0.0, max_value=95.0, step=5.0, key="cost_share_pct")
            st.number_input("Discount rate (%)", min_value=0.0, max_value=50.0, step=0.5, key="discount_rate_pct")
            st.number_input("Target payback (years)", min_value=0.5, max_value=30.0, step=0.5, key="target_payback_years")
            st.number_input("Conservation / wildlife upside ($/yr)", min_value=0.0, step=500.0, key="conservation_upside_year")
        with c2:
            st.number_input("Fixed grant ($)", min_value=0.0, step=500.0, key="fixed_grant")
            st.number_input("Analysis horizon (years)", min_value=2, max_value=15, step=1, key="horizon_years")
            st.number_input("Capital available ($)", min_value=0.0, step=1000.0, key="capital_available")
            st.number_input("Grazing-productivity upside ($/yr)", min_value=0.0, step=500.0, key="grazing_upside_year")
        st.warning("Soft benefits are displayed as optional upside. The headline Switch / Pilot / Stay verdict uses the hard-dollar economics from the core model and does not silently rely on conservation or grazing benefits.")
    nav_buttons()


def render_review() -> None:
    inputs = build_inputs()
    conn = connectivity_result()
    with st.container(border=True):
        st.markdown("## Review & edit")
        st.caption("Check the numbers before the connectivity gate and economics run.")
        sections: List[Tuple[str, List[Tuple[str, str]]]] = [
            ("1. Ranch data", [
                ("Scenario", st.session_state.ranch_name),
                ("Land status", LAND_LABELS[st.session_state.land_status]),
                ("Acres", f"{st.session_state.acres:,.0f}"),
                ("Head", f"{st.session_state.head_cattle:,}"),
                ("Fence miles", f"{st.session_state.fence_miles:,.1f}"),
            ]),
            ("2. Current analog labor", [
                ("Fence repair", f"{st.session_state.repair_hours_per_month:g} hrs/mo"),
                ("Cattle moving", f"{st.session_state.moving_hours_per_month:g} hrs/mo"),
                ("Loaded labor", f"{money(st.session_state.labor_rate)}/hr"),
                ("Repair material", f"{money(st.session_state.fence_material_per_mile_year)}/mile/yr"),
            ]),
            ("3. Connectivity", [
                ("Cell coverage", f"{st.session_state.cell_coverage_pct:g}%"),
                ("Tower feasible", "Yes" if st.session_state.tower_feasible else "No"),
                ("Satellite feasible", "Yes" if st.session_state.satellite_feasible else "No"),
                ("Preferred path", PATH_LABELS[st.session_state.preferred_path]),
                ("Current gate note", conn["message"]),
            ]),
            ("4. Virtual-fence costs", [
                ("Hardware", f"{money(st.session_state.hardware_per_head)}/head"),
                ("Subscription", f"{money(st.session_state.subscription_per_head_month)}/head/mo"),
                ("Tower upfront", money(st.session_state.tower_upfront)),
                ("Install & setup", money(st.session_state.install_setup)),
                ("Platform fee", f"{money(st.session_state.platform_fee_year)}/yr"),
                ("Management labor", f"{st.session_state.vf_management_hours_month:g} hrs/mo"),
                ("Replacement", f"{st.session_state.replacement_rate_pct:g}%/yr"),
            ]),
            ("5. Finance", [
                ("Cost share", f"{st.session_state.cost_share_pct:g}%"),
                ("Fixed grant", money(st.session_state.fixed_grant)),
                ("Discount rate", f"{st.session_state.discount_rate_pct:g}%"),
                ("Horizon", f"{st.session_state.horizon_years} years"),
                ("Payback target", f"{st.session_state.target_payback_years:g} years"),
                ("Capital available", money(st.session_state.capital_available)),
                ("Optional conservation", f"{money(st.session_state.conservation_upside_year)}/yr"),
                ("Optional grazing", f"{money(st.session_state.grazing_upside_year)}/yr"),
            ]),
        ]
        for title, rows in sections:
            with st.expander(title, expanded=True):
                df = pd.DataFrame(rows, columns=["Input", "Value"])
                st.dataframe(df, hide_index=True, use_container_width=True)

    with st.container(border=True):
        st.markdown("### Ready when you are")
        st.write("Generate the connectivity outcome, switch/stay economics, risk simulation, sensitivity, alarms, and AI analytics.")
        if st.button("Generate my savings & connectivity outcome", type="primary", use_container_width=True):
            st.session_state.generated = True
            st.session_state.result_tab = "The answer"
            st.rerun()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", use_container_width=True):
            st.session_state.step = 4
            st.rerun()
    with c2:
        if st.button("Edit from the beginning", use_container_width=True):
            st.session_state.step = 0
            st.rerun()


def render_blocked(results: Dict[str, Any]) -> None:
    tone, title, summary = verdict_for(results)
    st.markdown(f'<div class="verdict-card {tone}"><div class="verdict-kicker">Connectivity gate</div><div class="verdict-title">{title}</div><div class="verdict-copy">{summary}</div></div>', unsafe_allow_html=True)
    st.write("")
    with st.container(border=True):
        st.markdown(f"### Connectivity score: {results['score']}/100")
        st.progress(results["score"] / 100.0)
        conn = results["connectivity"]
        if not conn.get("viable"):
            st.error(conn["message"])
        elif not conn.get("preferred_viable"):
            st.warning(conn["message"])
            st.info(f"Recommended fallback: **{PATH_LABELS.get(conn.get('recommended_path'), conn.get('recommended_path'))}**. Go back to Connectivity and choose that path so the correct cost assumptions can be reviewed before calculation.")
        st.markdown("**Next checks**")
        st.markdown("- Ask carriers and collar vendors for a written map of usable coverage across grazing ground.\n- Price any repeater, tower, gateway, solar/power, and service costs.\n- Confirm a satellite option if the ranch lacks reliable terrestrial coverage.\n- Return to the connectivity step after the field facts are verified.")
        if st.button("Edit connectivity answers", type="primary"):
            st.session_state.generated = False
            st.session_state.step = 2
            st.rerun()


def render_answer(results: Dict[str, Any]) -> None:
    econ = results["economics"]
    tone, title, summary = verdict_for(results)
    st.markdown(f'<div class="verdict-card {tone}"><div class="verdict-kicker">Switch-or-Stay recommendation</div><div class="verdict-title">{title}</div><div class="verdict-copy">{summary}</div></div>', unsafe_allow_html=True)
    st.write("")

    with st.container(border=True):
        st.markdown(f"### Connectivity gate: passed ({results['score']}/100)")
        st.progress(results["score"] / 100.0)
        st.caption(results["connectivity"]["message"])

    cols = st.columns(5)
    payback = float(econ["payback_years"])
    cols[0].metric("Net upfront", money(econ["net_upfront"]))
    cols[1].metric("Annual analog cost", money(econ["analog_annual_total"]))
    cols[2].metric("Annual virtual cost", money(econ["virtual_annual_total"]))
    cols[3].metric("Payback", f"{payback:.1f} yrs" if math.isfinite(payback) else "Not reached")
    cols[4].metric(f"{econ['horizon_years']}-yr NPV", money(econ["npv"]))

    with st.container(border=True):
        st.markdown("### Where the annual cost goes")
        compare = pd.DataFrame(
            [
                ["Analog", "Fence repair labor", econ["analog_repair_labor"]],
                ["Analog", "Cattle-moving labor", econ["analog_moving_labor"]],
                ["Analog", "Repair materials", econ["analog_material"]],
                ["Virtual", "Subscription + platform", econ["virtual_subscription"]],
                ["Virtual", "Management labor", econ["virtual_labor"]],
                ["Virtual", "Collar replacement", econ["collar_replacement"]],
                ["Virtual", "Tower maintenance", econ["tower_maintenance"]],
            ],
            columns=["System", "Cost component", "Annual cost"],
        )
        fig = px.bar(compare, x="System", y="Annual cost", color="Cost component", barmode="stack", text_auto=".2s")
        fig.update_layout(legend_title_text="", yaxis_title="Annual cost ($)", xaxis_title="", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Hard-dollar annual savings:** {money(econ['annual_savings'])}")
        st.caption("Optional conservation and grazing upside is not included in this headline savings figure.")

    with st.container(border=True):
        st.markdown("### Money in and out")
        st.caption("The cumulative line crossing zero is the simple payback point; discounted cash flow is also shown.")
        cash = build_cash_flow_table(econ)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cash["year"], y=cash["cumulative_undiscounted"], mode="lines+markers", name="Cumulative cash"))
        fig.add_trace(go.Scatter(x=cash["year"], y=cash["cumulative_discounted"], mode="lines+markers", name="Discounted cumulative"))
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(xaxis_title="Year", yaxis_title="Cumulative value ($)", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cash.round(0), hide_index=True, use_container_width=True)

    with st.container(border=True):
        st.markdown("### Plain-language payback explanation")
        st.write(make_payback_explanation(results["inputs"], econ))
        if econ["optional_annual_upside"] > 0:
            st.info(f"Optional labeled upside entered: {money(econ['optional_annual_upside'])}/year. NPV including that optional upside would be {money(econ['npv_with_optional'])}, versus the hard-dollar headline NPV of {money(econ['npv'])}.")


def advisor_note(results: Dict[str, Any]) -> str:
    econ = results["economics"]
    tone, title, _ = verdict_for(results)
    prob = float((results["monte_carlo"]["npv"] > 0).mean()) if len(results["monte_carlo"]) else 0.0
    return (
        f"Assessment for {st.session_state.ranch_name}: recommendation = {title}. "
        f"Hard-dollar {econ['horizon_years']}-year NPV is {money(econ['npv'])}; simple payback is "
        f"{f'{econ['payback_years']:.1f} years' if math.isfinite(float(econ['payback_years'])) else 'not reached'}; "
        f"and {prob:.0%} of {MC_DRAWS} uncertainty runs finish with positive NPV. "
        f"Connectivity score is {results['score']}/100 on the selected {PATH_LABELS[st.session_state.preferred_path].lower()} path. "
        "Before purchase, verify actual pasture coverage, vendor quote terms, collar-loss/replacement policy, and any assumed cost-share in writing."
    )


def neighbor_benchmark() -> str:
    pop = generate_user_profile_population(n=500, seed=13)
    labor = pop["repair_hours_per_month"] + pop["moving_hours_per_month"]
    user_labor = st.session_state.repair_hours_per_month + st.session_state.moving_hours_per_month
    acres_pct = float((pop["acres"] <= st.session_state.acres).mean())
    herd_pct = float((pop["head_cattle"] <= st.session_state.head_cattle).mean())
    labor_pct = float((labor <= user_labor).mean())
    return (
        "Synthetic demo benchmark only — not a survey of neighboring ranches. "
        f"Your acreage is around the {acres_pct:.0%} percentile of the generated demo population, herd size around the {herd_pct:.0%} percentile, "
        f"and entered fence + cattle-moving labor around the {labor_pct:.0%} percentile. Use this only to test the future benchmarking workflow, not as market evidence."
    )


def vendor_questions() -> str:
    path = PATH_LABELS[st.session_state.preferred_path]
    return (
        f"Questions for a {path.lower()} vendor: 1) What written coverage standard applies to my grazing ground? "
        "2) What is included in the hardware and install quote? 3) Does the subscription escalate after year one? "
        "4) What is the warranty and lost/damaged collar policy? 5) What is the expected annual replacement rate? "
        "6) What happens during network/power outages? 7) What training period and physical backup containment do you recommend? "
        "8) Can I export my herd/location data, and what is your data-retention policy?"
    )


@st.cache_resource
def cached_fence_model():
    return train_fence_cost_model(seed=42)


def render_risk(results: Dict[str, Any]) -> None:
    econ = results["economics"]
    sensitivity = results["sensitivity"].copy()
    mc = results["monte_carlo"].copy()

    with st.container(border=True):
        st.markdown("### What moves the answer")
        st.caption("One input at a time is moved 25% down and up. Bars show the change from the base-case NPV.")
        if not sensitivity.empty:
            sensitivity["low_delta"] = sensitivity["low_npv"] - sensitivity["base_npv"]
            sensitivity["high_delta"] = sensitivity["high_npv"] - sensitivity["base_npv"]
            plot_df = sensitivity[["parameter", "low_delta", "high_delta"]].melt(id_vars="parameter", var_name="Scenario", value_name="NPV change")
            plot_df["Scenario"] = plot_df["Scenario"].map({"low_delta": "25% lower", "high_delta": "25% higher"})
            fig = px.bar(plot_df, y="parameter", x="NPV change", color="Scenario", barmode="group", orientation="h")
            fig.add_vline(x=0, line_dash="dash")
            fig.update_layout(yaxis_title="", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.markdown(f"### {MC_DRAWS} simulated seasons")
        st.caption("The uncertainty engine varies key price, labor, material, and cost-share assumptions to show how fragile or robust the decision is.")
        prob_positive = float((mc["npv"] > 0).mean())
        q10, q50, q90 = np.quantile(mc["npv"].to_numpy(), [0.1, 0.5, 0.9])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Comes out ahead", pct(prob_positive))
        c2.metric("Unlucky (P10)", money(q10))
        c3.metric("Middle (P50)", money(q50))
        c4.metric("Lucky (P90)", money(q90))
        fig = px.histogram(mc, x="npv", nbins=28, labels={"npv": "NPV ($)"})
        fig.add_vline(x=0, line_dash="dash")
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        sorted_npv = np.sort(mc["npv"].to_numpy())
        ecdf = pd.DataFrame({"Percentile": np.linspace(0, 100, len(sorted_npv)), "NPV": sorted_npv})
        fig2 = px.line(ecdf, x="Percentile", y="NPV")
        fig2.add_hline(y=0, line_dash="dash")
        fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with st.container(border=True):
        st.markdown("### Watch-outs & statistical alarm system")
        alarms = results["alarms"]
        for _, row in alarms.iterrows():
            sev = str(row["severity"]).lower()
            text = f"**{str(row['severity']).upper()} — {row['alarm']}**  \n{row['recommended_action']}"
            if sev == "high":
                st.error(text)
            elif sev == "medium":
                st.warning(text)
            elif sev == "low":
                st.info(text)
            else:
                st.success(text)

    with st.container(border=True):
        st.markdown('### Advisor tools & AI lab <span class="demo-badge">Demo only</span>', unsafe_allow_html=True)
        st.caption("These panels demonstrate the product roadmap. Synthetic ML/RL data is clearly labeled and should be replaced with vetted production data before operational use.")
        b1, b2, b3 = st.columns(3)
        if b1.button("Draft advisor note", use_container_width=True):
            st.session_state.advisor_title = "Draft advisor note"
            st.session_state.advisor_output = advisor_note(results)
        if b2.button("Neighbor benchmark", use_container_width=True):
            st.session_state.advisor_title = "Synthetic neighbor benchmark"
            st.session_state.advisor_output = neighbor_benchmark()
        if b3.button("Questions for vendor", use_container_width=True):
            st.session_state.advisor_title = "Vendor diligence questions"
            st.session_state.advisor_output = vendor_questions()
        if st.session_state.advisor_output:
            st.markdown(f"**{st.session_state.advisor_title}**")
            st.info(st.session_state.advisor_output)

        st.divider()
        st.markdown("#### ML fence-cost estimator")
        st.caption("Random-forest demo trained on synthetic regional/terrain examples. It estimates annual physical-fence repair material cost per mile.")
        mc1, mc2 = st.columns(2)
        region = mc1.selectbox("Region", ["Mountain West", "Great Plains", "Southwest", "Pacific", "Southeast"], key="ml_region")
        terrain = mc2.selectbox("Terrain", ["flat", "rolling", "rough", "mountain"], key="ml_terrain")
        if st.button("Run ML estimate"):
            model, _, r2 = cached_fence_model()
            estimate = estimate_fence_repair_material(model, st.session_state.acres, st.session_state.fence_miles, st.session_state.labor_rate, region, terrain)
            st.success(f"Demo estimate: {money(estimate)} per fence mile per year. In-sample synthetic R²: {r2:.2f}.")
            st.caption("Do not treat this synthetic model as a real NRCS/vendor quote. It exists to demonstrate the trainable ML module.")

        st.divider()
        st.markdown("#### Reinforcement-learning-style strategy simulator")
        st.caption("Epsilon-greedy bandit demo. It learns which action performs best across perturbed synthetic ranch scenarios; it does not autonomously control fences.")
        if st.button("Run RL simulation"):
            history, q_df = simulate_bandit_learning(results["inputs"], results["connectivity"].get("viable_paths", []), episodes=600, epsilon=0.12, seed=22)
            st.dataframe(q_df, hide_index=True, use_container_width=True)
            fig = px.line(history, x="episode", y="rolling_reward", labels={"rolling_reward": "30-episode rolling reward"})
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("#### Optional live public-data hook")
        st.caption("NASA POWER solar/climate context is optional and does not affect the headline verdict in this MVP.")
        a1, a2 = st.columns(2)
        a1.number_input("Latitude", min_value=-90.0, max_value=90.0, step=0.1, key="latitude")
        a2.number_input("Longitude", min_value=-180.0, max_value=180.0, step=0.1, key="longitude")
        if st.button("Fetch NASA POWER solar proxy"):
            with st.spinner("Checking public climate data..."):
                solar = fetch_nasa_power_solar_proxy(st.session_state.latitude, st.session_state.longitude)
            if solar.get("ok"):
                st.success(f"Annual-average solar proxy: {solar.get('annual_average_kwh_m2_day', 0):.2f} kWh/m²/day — source: {solar.get('source')}")
            else:
                st.warning(f"Public API was unavailable. The decision model still works without it. Details: {solar.get('error')}")

    with st.container(border=True):
        st.markdown("### Take it with you")
        report = build_report(results["inputs"], results["connectivity"], econ, results["alarms"])
        report["ui_migration"] = "Streamlit port of supplied Switch-or-Stay ranch UI"
        report["author"] = AUTHOR
        report["advisor"] = ADVISOR
        report["connectivity_score"] = results["score"]
        report["monte_carlo_summary"] = {
            "draws": MC_DRAWS,
            "probability_positive_npv": float((mc["npv"] > 0).mean()),
            "p10_npv": float(np.quantile(mc["npv"], 0.10)),
            "p50_npv": float(np.quantile(mc["npv"], 0.50)),
            "p90_npv": float(np.quantile(mc["npv"], 0.90)),
        }
        st.download_button("Download results (JSON)", safe_json_dumps(report), file_name="cattle_collar_switch_or_stay_report.json", mime="application/json", type="primary")


def render_results() -> None:
    results = compute_results()
    if not results["gate_passes"]:
        render_blocked(results)
        return

    selection = st.segmented_control("Results", ["The answer", "Risk & next steps"], key="result_tab")
    if selection == "Risk & next steps":
        render_risk(results)
    else:
        render_answer(results)

    st.write("")
    c1, c2 = st.columns(2)
    if c1.button("Back to inputs", use_container_width=True):
        st.session_state.generated = False
        st.session_state.step = 5
        st.rerun()
    if c2.button("Start a new assessment", type="primary", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        init_state()
        st.session_state.view = "assess"
        st.rerun()


def render_assessment() -> None:
    render_topbar(show_back=True)
    if st.session_state.generated:
        render_results()
        render_footer()
        return
    render_progress()
    step = int(st.session_state.step)
    if step == 0:
        render_step_ranch()
    elif step == 1:
        render_step_labor()
    elif step == 2:
        render_step_connectivity()
    elif step == 3:
        render_step_costs()
    elif step == 4:
        render_step_finance()
    else:
        render_review()
    render_footer()


init_state()
if st.session_state.view == "home":
    render_home()
else:
    render_assessment()
