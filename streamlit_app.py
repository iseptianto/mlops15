import json
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

# ====== CONFIG ======
API_BASE: str = st.secrets.get("API_BASE", "http://localhost:8101").rstrip("/")
ENDPOINT_HEALTH = f"{API_BASE}/health"
ENDPOINT_PREDICT = f"{API_BASE}/predict"
ENDPOINT_PREDICT_BATCH = f"{API_BASE}/predict-batch"  # optional; will fallback if not found
REQUEST_TIMEOUT = 20  # seconds
MAX_RETRIES = 2
RETRY_SLEEP = 1.2  # seconds

st.set_page_config(
    page_title="Tourism Recommender",
    page_icon="🧭",
    layout="wide",
)


# ====== HELPERS ======
def http_post(url: str, json_payload: Any) -> requests.Response:
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return requests.post(url, json=json_payload, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            last_exc = e
            time.sleep(RETRY_SLEEP)
    raise last_exc  # type: ignore


@st.cache_data(ttl=15)
def check_health() -> Dict[str, Any]:
    try:
        r = requests.get(ENDPOINT_HEALTH, timeout=REQUEST_TIMEOUT)
        return {"ok": r.ok, "status": r.status_code, "body": safe_json(r)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def safe_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return {"text": resp.text}


def infer_has_batch_endpoint() -> bool:
    try:
        # HEAD is cheaper; if server doesn't allow it, fallback to GET and accept 405 as "exists"
        r = requests.options(ENDPOINT_PREDICT_BATCH, timeout=REQUEST_TIMEOUT)
        if r.status_code in (200, 204, 401, 403, 405):  # treat as "likely exists"
            return True
    except Exception:
        pass
    return False


def payload_template() -> str:
    """
    EDIT THIS TEMPLATE to match your FastAPI schema.
    Example fields are placeholders; keep key names as your API expects.
    """
    example = {
        "city": "Yogyakarta",
        "month": 8,
        "budget": 150.0,
        "group_size": 2,
        "preferences": ["culture", "culinary"],
        # tambahkan fitur lain yang dipakai model...
    }
    return json.dumps(example, indent=2, ensure_ascii=False)


# ====== SIDEBAR ======
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("**API Base**")
    st.text(API_BASE)

    health = check_health()
    if health.get("ok"):
        st.success(f"API healthy (HTTP {health.get('status')})")
    else:
        st.error("API not reachable")
        if "error" in health:
            st.caption(health["error"])
    with st.expander("Health response"):
        st.json(health)

    st.markdown("---")
    st.caption("Tip: set `API_BASE` di **secrets** Streamlit Cloud.")
    st.code(
        '''# .streamlit/secrets.toml
API_BASE = "https://your-domain-or-ngrok:8101"''',
        language="toml",
    )

st.title("🧭 Tourism Recommender")
st.caption("Frontend ringan untuk model/servis rekomendasi pariwisata (FastAPI).")

tab_single, tab_batch, tab_about = st.tabs(["Single Predict", "Batch Predict", "About"])

# ====== SINGLE PREDICT ======
with tab_single:
    st.subheader("Single Predict")
    st.caption("Edit payload JSON sesuai skema endpoint `/predict` FastAPI kamu.")
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        json_text = st.text_area(
            "Request JSON",
            value=payload_template(),
            height=320,
            help="Sesuaikan keys/values agar cocok dengan Pydantic model di API.",
        )
        btn = st.button("🚀 Predict", type="primary")

    with col2:
        st.markdown("**Options**")
        timeout = st.number_input("Request timeout (s)", 1, 120, REQUEST_TIMEOUT)
        retries = st.number_input("Max retries", 0, 5, MAX_RETRIES)
        # sync back (optional)
        REQUEST_TIMEOUT = int(timeout)
        MAX_RETRIES = int(retries)

    if btn:
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError as e:
            st.error(f"JSON invalid: {e}")
            st.stop()

        with st.spinner("Calling /predict ..."):
            try:
                resp = http_post(ENDPOINT_PREDICT, payload)
            except Exception as e:
                st.error(f"Gagal call /predict: {e}")
                st.stop()

        st.markdown("**Raw response**")
        st.code(resp.text, language="json")

        data = safe_json(resp)
        # Best effort: if result is a dict or list that looks table-ish, show as table
        if isinstance(data, dict) and "result" in data:
            try:
                df = pd.DataFrame(data["result"] if isinstance(data["result"], list) else [data["result"]])
                st.markdown("**Parsed result**")
                st.dataframe(df, use_container_width=True)
            except Exception:
                pass
        elif isinstance(data, list):
            try:
                df = pd.DataFrame(data)
                st.markdown("**Parsed result**")
                st.dataframe(df, use_container_width=True)
            except Exception:
                pass

# ====== BATCH PREDICT ======
with tab_batch:
    st.subheader("Batch Predict")
    st.caption(
        "Upload CSV dengan kolom sesuai fitur yang API harapkan. "
        "App akan mencoba /predict-batch bila tersedia; jika tidak, akan loop /predict per baris."
    )
    uploaded = st.file_uploader("CSV file", type=["csv"])
    run_batch = st.button("🚀 Run Batch")

    if run_batch:
        if not uploaded:
            st.warning("Upload CSV dulu.")
            st.stop()

        df = pd.read_csv(uploaded)
        st.write("Preview:")
        st.dataframe(df.head(20), use_container_width=True)

        has_batch = infer_has_batch_endpoint()
        st.info(f"Batch endpoint detected: **{has_batch}**")

        if has_batch:
            # send list of records as JSON
            records = df.to_dict(orient="records")
            with st.spinner("Calling /predict-batch ..."):
                try:
                    resp = http_post(ENDPOINT_PREDICT_BATCH, {"instances": records})
                except Exception as e:
                    st.error(f"Gagal call /predict-batch: {e}")
                    st.stop()
            st.markdown("**Raw response**")
            st.code(resp.text, language="json")
            data = safe_json(resp)

            # Try to show as table
            if isinstance(data, dict) and "predictions" in data:
                try:
                    out = data["predictions"]
                    out_df = pd.DataFrame(out if isinstance(out, list) else [out])
                    st.success("Done.")
                    st.dataframe(out_df, use_container_width=True)
                    st.download_button(
                        "⬇️ Download predictions (CSV)",
                        out_df.to_csv(index=False).encode("utf-8"),
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
                except Exception:
                    pass
        else:
            # fallback: per-row predict
            results: List[Dict[str, Any]] = []
            progress = st.progress(0, text="Processing...")
            for i, rec in enumerate(df.to_dict(orient="records"), start=1):
                try:
                    r = http_post(ENDPOINT_PREDICT, rec)
                    data = safe_json(r)
                    # flatten and keep original + prediction
                    if isinstance(data, dict):
                        merged = {**rec, **{f"pred_{k}": v for k, v in data.items()}}
                    else:
                        merged = {**rec, "prediction": data}
                    results.append(merged)
                except Exception as e:
                    results.append({**rec, "error": str(e)})
                progress.progress(i / len(df), text=f"{i}/{len(df)}")

            out_df = pd.DataFrame(results)
            st.success("Batch selesai.")
            st.dataframe(out_df, use_container_width=True)
            st.download_button(
                "⬇️ Download predictions (CSV)",
                out_df.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )

# ====== ABOUT ======
with tab_about:
    st.subheader("About")
    st.markdown(
        """
**Tourism Recommender UI**  
- Backend: FastAPI (`/health`, `/predict`, opsional `/predict-batch`)  
- Frontend: Streamlit (file tunggal)  
- Set `API_BASE` lewat `secrets.toml` (lihat sidebar).  

> Catatan: Sesuaikan **payload template** di fungsi `payload_template()` agar match dengan Pydantic request model di API kamu.
"""
    )
