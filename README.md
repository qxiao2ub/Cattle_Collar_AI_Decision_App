# Cattle Collar Switch-or-Stay AI Decision App

A Streamlit-ready ranch decision-support application that helps a rancher decide whether to **Switch**, **Pilot**, or **Stay** with physical fencing after checking connectivity, costs, payback, uncertainty, and operational risk.

## Project credits

- **Author:** Sykes Lamensdorf
- **Advisor:** Dr. Qingyang Xiao
- **License:** MIT

## Live links

- GitHub target: https://github.com/qxiao2ub/Cattle_Collar_AI_Decision_App
- Streamlit app: https://cattle-collar-ai-decision.streamlit.app/

## UI migration

The supplied `cattle-collar-switch-or-stay-ai-source.zip` design was a React/Lovable-style frontend. This repository ports that visual language and interaction flow into native **Streamlit** so Streamlit Community Cloud can deploy the app directly from Python.

The migrated experience includes:

1. Ranch-oriented landing page with the **Switch-or-Stay** identity.
2. Six-stage guided assessment and review workflow.
3. Signal-first hard connectivity gate.
4. Editable cell / tower / satellite cost assumptions.
5. Hard-dollar switch/stay economics and payback explanation.
6. Switch / Pilot / Stay recommendation framing.
7. Cash-flow and annual-cost charts.
8. ±25% one-at-a-time sensitivity analysis.
9. 800-run Monte Carlo uncertainty simulation.
10. Statistical alarm / watch-out system.
11. Synthetic ML fence-cost estimator demonstration.
12. Reinforcement-learning-style bandit simulation.
13. Optional NASA POWER public-data hook.
14. JSON report download.
15. Author and advisor credits directly in the application.

## Repository structure

```text
.
├── streamlit_app.py
├── cattle_collar_core.py
├── requirements.txt
├── LICENSE
├── README.md
├── UI_MIGRATION_NOTES.md
├── .streamlit/
│   └── config.toml
├── assets/
│   └── cattle_collar_app_flow.png
├── notebooks/
│   └── cattle_collar_ai_decision_pipeline.ipynb
└── sample_inputs/
    └── sample_ranch_profile.json
```

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud

1. Create or open the GitHub repository `Cattle_Collar_AI_Decision_App`.
2. Upload the **contents** of this repository zip to the repository root.
3. In Streamlit Community Cloud, choose the GitHub repository.
4. Set the main file path to:

```text
streamlit_app.py
```

5. Deploy. No API secret is required for the core MVP. The optional NASA POWER call uses a public endpoint and fails gracefully if it is unavailable.

## Important modeling notes

- Starter collar, subscription, tower, and platform numbers are **editable demo assumptions**, not verified current vendor quotes.
- The ML estimator trains on synthetic demonstration data and must be replaced with vetted NRCS, extension, vendor, and ranch-history data before operational use.
- The RL module is an experimentation simulator; it does **not** autonomously control cattle, virtual fences, or animal-welfare decisions.
- Optional conservation/wildlife and grazing-productivity benefits are labeled separately. The headline recommendation uses the core hard-dollar economics rather than silently relying on soft benefits.
- Connectivity, cost-share eligibility, and vendor pricing should be independently verified for the specific ranch before a purchase decision.

## License

This project is released under the [MIT License](LICENSE).
