"""
Streamlit frontend — factory price prediction UI.

Run with:
    streamlit run frontend/app.py

Requires backend to be running:
    uvicorn backend.api:app --reload --port 8000
"""

import os
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Factory Price Predictor",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 Factory Price Predictor")
st.caption("Singapore industrial property — unit price estimate ($ psf)")

# Fetch dropdown options from backend
try:
    options = requests.get(f"{BACKEND_URL}/options", timeout=5).json()
    PLANNING_AREAS = options["planning_areas"]
    REGIONS = options["regions"]
    FLOOR_LEVELS = options["floor_levels"]
    SALE_TYPES = options["sale_types"]
except Exception:
    st.error("Cannot connect to backend. Make sure the API is running on port 8000.")
    st.stop()

st.divider()

# --- Input Form ---
with st.form("prediction_form"):
    st.subheader("Factory Space Details")

    left_col, right_col = st.columns(2)

    with left_col:
        area_sqft = st.number_input(
            "Floor Area (sqft)",
            min_value=100.0,
            max_value=50000.0,
            value=1500.0,
            step=50.0,
        )
        remaining_lease_years = st.number_input(
            "Remaining Lease (years)",
            min_value=1.0,
            max_value=99.0,
            value=45.0,
            step=1.0,
        )
        lease_duration = st.selectbox(
            "Total Lease Duration (years)",
            options=[30, 60, 99],
            index=1,
        )
        dist_to_mrt_m = st.number_input(
            "Distance to Nearest MRT (metres)",
            min_value=0.0,
            max_value=10000.0,
            value=800.0,
            step=50.0,
        )

    with right_col:
        planning_area = st.selectbox("Planning Area", options=PLANNING_AREAS)
        region = st.selectbox("Region", options=REGIONS)
        floor_level = st.selectbox("Floor Level", options=FLOOR_LEVELS)
        type_of_sale = st.selectbox("Type of Sale", options=SALE_TYPES)

    submitted = st.form_submit_button("Predict Price", use_container_width=True)

# --- Prediction ---
if submitted:
    if remaining_lease_years > lease_duration:
        st.error("Remaining lease years cannot exceed total lease duration.")
    else:
        payload = {
            "area_sqft": area_sqft,
            "remaining_lease_years": remaining_lease_years,
            "lease_duration": float(lease_duration),
            "planning_area": planning_area,
            "floor_level": floor_level,
            "type_of_sale": type_of_sale,
            "region": region,
            "dist_to_mrt_m": dist_to_mrt_m,
        }

        with st.spinner("Predicting..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/predict",
                    json=payload,
                    timeout=10,
                )
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.HTTPError as e:
                st.error(f"Prediction failed: {response.json().get('detail', str(e))}")
                st.stop()
            except Exception as e:
                st.error(f"Could not reach backend: {e}")
                st.stop()

        st.divider()
        st.subheader("Prediction Result")

        unit_price, total_price, lower_bound, upper_bound = st.columns(4)
        with unit_price:
            st.metric(
                label="Estimated Unit Price",
                value=f"${result['predicted_psf']:,.2f} psf",
            )
        with total_price:
            st.metric(
                label="Estimated Total Price",
                value=f"${result['total_price']:,.0f}",
                help=f"Unit price x floor area ({area_sqft:,.0f} sqft)",
            )
        with lower_bound:
            st.metric(
                label="Lower Bound",
                value=f"${result['lower_bound']:,.2f} psf",
                help=f"Predicted price minus model RMSE (-${result['rmse']:.2f})",
            )
        with upper_bound:
            st.metric(
                label="Upper Bound",
                value=f"${result['upper_bound']:,.2f} psf",
                help=f"Predicted price plus model RMSE (+${result['rmse']:.2f})",
            )

        with st.expander("Input Summary"):
            st.json(payload)
