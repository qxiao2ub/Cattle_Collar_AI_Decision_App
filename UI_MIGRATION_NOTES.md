# UI Migration Notes

The attached UI source defined the ranch-facing experience around these principles:

- calm ranch-oriented cream / green / earth-tone visual design;
- a landing page with a prominent Switch-or-Stay question;
- signal-first decision gating;
- five rancher input stages plus review;
- clear Switch / Pilot / Stay output;
- cumulative cash-flow, sensitivity, and 800-season uncertainty visuals;
- advisor-side demo tools separated from rancher inputs.

This Streamlit port recreates those ideas using native Streamlit widgets, Plotly charts, session state, and custom CSS. The original React runtime is intentionally not required for deployment because Streamlit Community Cloud launches Python applications from `streamlit_app.py`.

The existing Python decision engine remains the source for connectivity, economics, sensitivity, Monte Carlo, ML, RL, alarms, API hooks, and exports.
