# BMS Troubleshooter — RAG + Agents (Streamlit)

## Setup
1. Clone this repository.
2. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` or other tokens.
3. (Optional) create a virtualenv: `python -m venv .venv && source .venv/bin/activate`
4. Install deps: `pip install -r requirements.txt`
5. Run locally: `streamlit run app.py`

## Deploy to Streamlit Cloud
- Ensure `requirements.txt` present at repo root.
- Push to GitHub and create a new Streamlit app pointing to `app.py`.
- Set environment secrets (OPENAI_API_KEY) in Streamlit's dashboard.

## Replace placeholders
- Tools like `tool_check_chiller_status` are simulated; replace with OPC-UA / BACnet / MQTT calls.
