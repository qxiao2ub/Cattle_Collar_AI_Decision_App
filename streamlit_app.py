from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cattle_collar_core import (
    COST_PARAMETER_TABLE,
    DEFAULT_SYSTEM_PRESETS,
    apply_system_preset,
    build_cash_flow_table,
    build_report,
    build_sensitivity_table,
    check_connectivity,
    cluster_user_profiles,
    compute_switch_stay_economics,
    estimate_fence_repair_material,
    example_ranch_inputs,
    fetch_nasa_power_solar_proxy,
    generate_follow_up_questions,
    generate_user_profile_population,
    get_preset,
    make_payback_explanation,
    monte_carlo_uncertainty,
    normalize_system_type,
    safe_json_dumps,
    simulate_bandit_learning,
    statistics_alarm_system,
    train_fence_cost_model,
)

APP_DIR = Path(__file__).resolve().parent
FLOW_IMAGE = APP_DIR / "assets" / "cattle_collar_app_flow.png"

st.set_page_config(
    page_title="Cattle Collar Switch-or-Stay AI",
    page_icon="🐄",
    layout="wide",
)


def money(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "No payback"
    return f"${float(value):,.0f}"


def years(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "No finite payback"
    return f"{float(value):.1f} years"


@st.cache_resource(show_spinner=False)
def cached_fence_model():
    return train_fence_cost_model()


@st.cache_data(show_spinner=False)
def cached_profiles():
    df = generate_user_profile_population()
    return cluster_user_profiles(df)


st.title("Cattle Collar Switch-or-Stay AI App")
st.caption(
    "Decision-support MVP for ranchers comparing analog physical fencing with GPS/virtual fencing collars."
)
st.markdown(
    "**Author:** Sykes Lamensdorf  \n**Advisor:** Dr. Qingyang Xiao"
)

with st.sidebar:
    st.markdown("### Project Credits")
    st.markdown("**Author:** Sykes Lamensdorf")
    st.markdown("**Advisor:** Dr. Qingyang Xiao")
    st.divider()
    st.header("1. Ranch data")
    location_name = st.text_input("Ranch / scenario name", "Example Ranch")
    owned_or_leased = st.selectbox("Land status", ["Owned", "Leased", "Mixed"], index=0)
    acres = st.number_input("Owned or managed acres", min_value=0.0, value=5000.0, step=100.0)
    head_cattle = st.number_input("Head of cattle needing collars", min_value=0, value=600, step=10)
    fence_miles = st.number_input("Miles of existing physical fence", min_value=0.0, value=42.0, step=1.0)

    st.header("2. Current analog labor")
    repair_hours = st.number_input("Fence repair hours per month", min_value=0.0, value=20.0, step=1.0)
    moving_hours = st.number_input("Cattle-moving hours per month", min_value=0.0, value=18.0, step=1.0)
    labor_rate = st.number_input("Loaded labor rate ($/hour)", min_value=0.0, value=28.0, step=1.0)
    fence_material = st.number_input("Annual fence repair material ($/mile)", min_value=0.0, value=250.0, step=25.0)

    st.header("3. Connectivity gate")
    cell_available = st.checkbox("Cell coverage works on useful pasture area", value=True)
    tower_possible = st.checkbox("Ranch tower/base station is feasible", value=True)
    satellite_available = st.checkbox("Satellite collar path is feasible", value=True)
    preferred_label = st.selectbox(
        "Preferred virtual-fence path",
        ["Auto pick viable path"] + list(DEFAULT_SYSTEM_PRESETS.keys()),
        index=0,
    )
    preferred_path = "auto" if preferred_label.startswith("Auto") else normalize_system_type(preferred_label)
    conn = check_connectivity(cell_available, tower_possible, satellite_available, preferred_path)

    st.header("4. Virtual-fence cost assumptions")
    system_for_defaults = conn["recommended_path"] or "cell"
    if preferred_path not in {"auto", "stay"} and conn.get("preferred_viable", False):
        system_for_defaults = preferred_path
    preset = get_preset(system_for_defaults)
    system_type = preset["system_type"]
    st.caption(f"Editable defaults currently loaded for: {system_type}")
    collar_cost = st.number_input("Collar hardware cost per head ($)", min_value=0.0, value=float(preset["collar_hardware_cost"]), step=25.0)
    subscription = st.number_input("Subscription per head per month ($)", min_value=0.0, value=float(preset["subscription_per_head_month"]), step=1.0)
    base_station = st.number_input("Base station / tower up-front cost ($)", min_value=0.0, value=float(preset["base_station_cost"]), step=500.0)
    install_cost = st.number_input("Installation / setup cost ($)", min_value=0.0, value=float(preset["install_cost"]), step=500.0)
    tower_maint = st.number_input("Tower or infrastructure annual maintenance ($)", min_value=0.0, value=float(preset["tower_maintenance_annual"]), step=100.0)
    platform_fee = st.number_input("Extra ranch-level platform fee per year ($)", min_value=0.0, value=0.0, step=100.0)
    virtual_labor = st.number_input("Virtual-fence management labor hours per month", min_value=0.0, value=5.0, step=1.0)
    training_hours = st.number_input("Initial training / transition labor hours", min_value=0.0, value=24.0, step=2.0)
    replacement_rate = st.slider("Annual collar replacement rate", min_value=0.0, max_value=0.5, value=0.08, step=0.01)

    st.header("5. Finance and optional upside")
    cost_share_rate = st.slider("Cost-share offset rate", min_value=0.0, max_value=0.95, value=0.35, step=0.01)
    fixed_cost_share = st.number_input("Additional fixed cost-share/grant ($)", min_value=0.0, value=0.0, step=500.0)
    discount_rate = st.slider("Discount rate", min_value=0.0, max_value=0.30, value=0.08, step=0.01)
    horizon_years = st.slider("Analysis horizon (years)", min_value=1, max_value=20, value=7, step=1)
    target_payback = st.slider("Target payback (years)", min_value=0.5, max_value=15.0, value=4.0, step=0.5)
    capital_available = st.number_input("Available capital for switch ($)", min_value=0.0, value=250000.0, step=5000.0)
    conservation_upside = st.number_input("Optional conservation/wildlife upside per year ($)", min_value=0.0, value=0.0, step=500.0)
    grazing_upside = st.number_input("Optional grazing-productivity upside per year ($)", min_value=0.0, value=0.0, step=500.0)

inputs = {
    "location_name": location_name,
    "owned_or_leased": owned_or_leased,
    "acres": acres,
    "head_cattle": int(head_cattle),
    "fence_miles": fence_miles,
    "repair_hours_per_month": repair_hours,
    "moving_hours_per_month": moving_hours,
    "labor_rate": labor_rate,
    "fence_repair_material_per_mile_annual": fence_material,
    "virtual_labor_hours_per_month": virtual_labor,
    "collar_hardware_cost": collar_cost,
    "subscription_per_head_month": subscription,
    "ranch_platform_fee_annual": platform_fee,
    "base_station_cost": base_station,
    "install_cost": install_cost,
    "training_hours": training_hours,
    "collar_replacement_rate": replacement_rate,
    "tower_maintenance_annual": tower_maint,
    "cost_share_rate": cost_share_rate,
    "fixed_cost_share": fixed_cost_share,
    "discount_rate": discount_rate,
    "horizon_years": horizon_years,
    "target_payback_years": target_payback,
    "system_type": system_type,
    "optional_conservation_value_annual": conservation_upside,
    "optional_grazing_gain_annual": grazing_upside,
    "capital_available": capital_available,
}

if conn["viable"]:
    econ = compute_switch_stay_economics(inputs)
    sensitivity = build_sensitivity_table(inputs)
    mc = monte_carlo_uncertainty(inputs, n=800, seed=7)
    alarms = statistics_alarm_system(inputs, econ, conn, mc, sensitivity)
else:
    econ = None
    sensitivity = pd.DataFrame()
    mc = pd.DataFrame()
    alarms = statistics_alarm_system(inputs, None, conn, None, None)

tabs = st.tabs(
    [
        "Flow",
        "Inputs + LLM questions",
        "Connectivity gate",
        "Switch/stay math",
        "Sensitivity + statistics",
        "ML estimator",
        "User groups",
        "RL simulator",
        "Alarms + export",
    ]
)

with tabs[0]:
    st.subheader("Pipeline flow")
    if FLOW_IMAGE.exists():
        st.image(str(FLOW_IMAGE), use_container_width=True)
    st.markdown(
        """
        **MVP architecture**

        1. Rancher enters structured data: acres, cattle, fence, and labor.
        2. A conversation layer asks for missing ranch-specific details.
        3. A hard connectivity gate checks whether cell, tower, or satellite service can support collars.
        4. The processing layer runs transparent switch/stay math and optional API hooks.
        5. The output layer shows payback, charts, sensitivity, and a plain-English verdict.
        """
    )

with tabs[1]:
    st.subheader("Structured inputs")
    st.dataframe(pd.DataFrame([inputs]).T.rename(columns={0: "value"}), use_container_width=True)
    st.subheader("Conversation prompts to fill missing details")
    for q in generate_follow_up_questions(inputs, conn):
        st.write(f"- {q}")
    st.subheader("Cost parameters and source basis")
    st.dataframe(COST_PARAMETER_TABLE, use_container_width=True)

with tabs[2]:
    st.subheader("Connectivity hard gate")
    if conn["viable"]:
        st.success(conn["message"])
        st.write("The app can proceed to switch/stay math using the selected or recommended path.")
    else:
        st.error(conn["message"])
        st.write("Because collars cannot function without a workable connectivity path, the app skips the ROI math.")

    with st.expander("Optional API hook: solar/climate proxy"):
        lat = st.number_input("Latitude", value=38.5)
        lon = st.number_input("Longitude", value=-106.0)
        if st.button("Fetch NASA POWER solar proxy"):
            solar = fetch_nasa_power_solar_proxy(lat, lon)
            st.json(solar)

with tabs[3]:
    st.subheader("Switch/stay decision math")
    if econ is None:
        st.error("No viable connectivity path. Math is intentionally skipped.")
    else:
        rec = econ["recommendation_short"].upper()
        if rec == "SWITCH":
            st.success(econ["recommendation"])
        elif rec == "BORDERLINE":
            st.warning(econ["recommendation"])
        else:
            st.error(econ["recommendation"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Annual analog burden", money(econ["analog_annual_total"]))
        m2.metric("Annual virtual cost", money(econ["virtual_annual_total"]))
        m3.metric("Annual savings", money(econ["annual_savings"]))
        m4.metric("Payback", years(econ["payback_years"]))
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Net up-front cost", money(econ["net_upfront"]))
        m6.metric("NPV", money(econ["npv"]))
        m7.metric("ROI multiple", "No upfront" if not math.isfinite(econ["roi_multiple"]) else f"{econ['roi_multiple']:.2f}x")
        m8.metric("NPV incl. optional upside", money(econ["npv_with_optional"]))

        st.info(make_payback_explanation(inputs, econ))

        cost_df = pd.DataFrame(
            [
                {"category": "Analog repair labor", "annual_cost": econ["analog_repair_labor"]},
                {"category": "Analog moving labor", "annual_cost": econ["analog_moving_labor"]},
                {"category": "Analog repair material", "annual_cost": econ["analog_material"]},
                {"category": "Virtual subscription", "annual_cost": econ["virtual_subscription"]},
                {"category": "Virtual labor", "annual_cost": econ["virtual_labor"]},
                {"category": "Collar replacement", "annual_cost": econ["collar_replacement"]},
                {"category": "Tower maintenance", "annual_cost": econ["tower_maintenance"]},
            ]
        )
        fig = px.bar(cost_df, x="category", y="annual_cost", title="Annual cost components")
        fig.update_layout(xaxis_title="Cost category", yaxis_title="Annual dollars")
        st.plotly_chart(fig, use_container_width=True)

        cash = build_cash_flow_table(econ)
        fig2 = px.line(
            cash,
            x="year",
            y=["cumulative_undiscounted", "cumulative_discounted"],
            markers=True,
            title="Cumulative switch cash flow",
        )
        fig2.add_hline(y=0, line_dash="dash")
        fig2.update_layout(xaxis_title="Year", yaxis_title="Cumulative dollars")
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(cash, use_container_width=True)

with tabs[4]:
    st.subheader("Sensitivity and statistical credibility")
    if econ is None:
        st.warning("No sensitivity analysis because connectivity failed.")
    else:
        st.write("One-at-a-time sensitivity shows which inputs can change NPV most.")
        st.dataframe(sensitivity, use_container_width=True)
        fig = px.bar(
            sensitivity.sort_values("npv_swing", ascending=True),
            x="npv_swing",
            y="parameter",
            orientation="h",
            title="Tornado-style NPV sensitivity",
        )
        fig.update_layout(xaxis_title="Maximum absolute NPV swing", yaxis_title="Parameter")
        st.plotly_chart(fig, use_container_width=True)

        st.write("Monte Carlo uncertainty around key cost and labor assumptions.")
        finite_mc = mc.replace([np.inf, -np.inf], np.nan).dropna(subset=["payback_years"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Switch probability", f"{(mc['recommendation_short'] == 'switch').mean():.0%}")
        c2.metric("Median NPV", money(mc["npv"].median()))
        c3.metric("Median payback", years(finite_mc["payback_years"].median()) if not finite_mc.empty else "No finite payback")
        fig_hist = px.histogram(mc, x="npv", nbins=40, title="Monte Carlo NPV distribution")
        st.plotly_chart(fig_hist, use_container_width=True)

with tabs[5]:
    st.subheader("ML estimator for fuzzy analog fence costs")
    st.write(
        "This MVP trains a random-forest model on synthetic training data. In production, replace the synthetic data with vetted NRCS practice-cost, extension-budget, vendor, and ranch-history data."
    )
    model, train_df, score = cached_fence_model()
    col_a, col_b, col_c = st.columns(3)
    region = col_a.selectbox("Region for estimate", sorted(train_df["region"].unique().tolist()))
    terrain = col_b.selectbox("Terrain", sorted(train_df["terrain"].unique().tolist()))
    use_ml = col_c.checkbox("Use ML estimate in what-if", value=False)
    estimate = estimate_fence_repair_material(model, acres, fence_miles, labor_rate, region, terrain)
    st.metric("Estimated annual fence repair material per mile", money(estimate))
    st.caption(f"Training R^2 on synthetic data: {score:.2f}")
    fig_ml = px.scatter(
        train_df.sample(min(400, len(train_df)), random_state=1),
        x="fence_miles",
        y="fence_repair_material_per_mile_annual",
        color="terrain",
        title="Synthetic ML training data overview",
    )
    st.plotly_chart(fig_ml, use_container_width=True)
    if use_ml and econ is not None:
        ml_inputs = dict(inputs)
        ml_inputs["fence_repair_material_per_mile_annual"] = estimate
        ml_econ = compute_switch_stay_economics(ml_inputs)
        st.info(f"With ML-estimated fence material, recommendation becomes: {ml_econ['recommendation']}")

with tabs[6]:
    st.subheader("Anonymous user/ranch grouping")
    st.write(
        "The brief envisions deep learning to group users. This MVP uses KMeans clustering as a transparent placeholder until enough anonymous real user data exists for a deeper model."
    )
    profiles, centers = cached_profiles()
    st.dataframe(centers, use_container_width=True)
    fig_cluster = px.scatter(
        profiles,
        x="acres",
        y="head_cattle",
        color="segment",
        hover_data=["fence_miles", "repair_hours_per_month", "moving_hours_per_month"],
        title="Synthetic anonymous rancher segments",
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

with tabs[7]:
    st.subheader("Reinforcement-learning-style simulator")
    st.write(
        "This epsilon-greedy bandit learns which action has the highest simulated reward across perturbed ranch scenarios. It is a product-design simulator, not an autonomous operating policy."
    )
    if econ is None:
        st.warning("No RL simulation because connectivity failed.")
    else:
        episodes = st.slider("Simulation episodes", min_value=100, max_value=2000, value=600, step=100)
        history, q_df = simulate_bandit_learning(inputs, viable_paths=conn["viable_paths"], episodes=episodes)
        st.dataframe(q_df, use_container_width=True)
        fig_rl = px.line(history, x="episode", y="rolling_reward", title="RL simulator rolling reward")
        st.plotly_chart(fig_rl, use_container_width=True)
        fig_actions = px.histogram(history, x="action", title="Actions selected during exploration/exploitation")
        st.plotly_chart(fig_actions, use_container_width=True)

with tabs[8]:
    st.subheader("Alarm system and export")
    st.dataframe(alarms, use_container_width=True)
    if econ is None:
        st.error("No viable path: the export records the failed connectivity gate and no ROI math.")
    report = build_report(inputs, conn, econ, alarms)
    st.download_button(
        "Download JSON report",
        data=safe_json_dumps(report),
        file_name="cattle_collar_switch_stay_report.json",
        mime="application/json",
    )
    st.markdown(
        """
        **Responsible-use notes**

        - The app is decision support, not a guarantee of profit or a substitute for ranch-specific professional advice.
        - Soft benefits such as conservation or grazing gains remain optional upside and are not silently included in the headline ROI.
        - Anonymous user data should be minimized, aggregated, and separated from personal identity before model training.
        - Vendor pricing, public cost-share rules, and connectivity status should be refreshed before production use.
        """
    )
