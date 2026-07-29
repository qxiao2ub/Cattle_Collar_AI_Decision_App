# Cattle Collar Switch-or-Stay AI App

This repository is a Streamlit-ready MVP for the cattle collar / virtual fencing decision-support app.

The app follows the provided product flow:

1. Rancher enters structured data: acres, cattle, fence, and labor.
2. A conversation layer asks for missing ranch-specific details.
3. A hard connectivity gate checks whether cell, tower, or satellite service can support collars.
4. A processing layer runs transparent switch/stay math and optional API hooks.
5. The app outputs charts, payback period, sensitivity analysis, an alarm system, and an exportable JSON report.

## Repository contents

```text
cattle_collar_core.py                Core math, ML, RL, sensitivity, alarm functions
streamlit_app.py                     Streamlit Community Cloud entry point
requirements.txt                     Python dependencies
assets/cattle_collar_app_flow.png    Provided architecture flow image
notebooks/cattle_collar_ai_decision_pipeline.ipynb  Colab-ready notebook
sample_inputs/sample_ranch_profile.json             Example input profile
.streamlit/config.toml               Streamlit theme settings
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload the contents of this zip file to the repository root.
3. Go to Streamlit Community Cloud and create a new app.
4. Choose the GitHub repo and set the main file path to:

```text
streamlit_app.py
```

5. No API keys are required for the MVP. The optional NASA POWER solar proxy uses a public endpoint and fails gracefully if network access is unavailable.

## Important MVP notes

- The default collar, subscription, tower, labor, and cost-share numbers are editable starter assumptions, not verified vendor quotes.
- The ML estimator is trained on synthetic data. Replace this with vetted NRCS, extension, vendor, and ranch-history data before operational use.
- The RL module is a simulator to test recommendation learning. It does not autonomously operate fences or trade off animal welfare/safety constraints.
- Soft benefits such as wildlife/conservation upside and grazing gains are shown separately from the headline ROI.
