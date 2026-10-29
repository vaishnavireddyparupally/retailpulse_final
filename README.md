# RetailPulse — Fixed Streamlit Project

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Fixes applied
- Fixed demand forecast filter to use `sales["product"]` rather than `sales.product`.
- Fixed distinct product count to use `sales["product"].nunique()`.
- Python syntax parsed successfully.

Demo data is synthetic. Forecasting and monitoring components are demonstration baselines, not production MLOps integrations.
