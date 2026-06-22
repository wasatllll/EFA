# -*- coding: utf-8 -*-
"""
Le Fort I Personalized Titanium Plate Calculator
Research-use-only, FEA-informed GPR calculator.

User input:
    1) Maxillary advancement (2.0–8.0 mm)
    2) Loading mode (Vertical / Oblique)

Automatic output:
    All 6 material–structure combinations:
    Pure Ti / TC4 × S1 / S2 / S3

Primary endpoint:
    GPR-estimated minimum robustly safe thickness, defined as:
    (P99 upper 95% model-estimated limit / Rp0.2) <= 0.80

Place this file in the SAME folder as:
    gpr_p99_min_safe_thickness_bundle.joblib
    requirements.txt
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# Page configuration
# =============================================================================
st.set_page_config(
    page_title="Le Fort I Plate Thickness Calculator",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {max-width: 1280px; padding-top: 2.0rem; padding-bottom: 2.0rem;}
        h1, h2, h3 {font-family: "Times New Roman", serif;}
        .stDataFrame {font-size: 15px;}
    </style>
    """,
    unsafe_allow_html=True,
)

APP_DIR = Path(__file__).resolve().parent
MODEL_BUNDLE_FILE = APP_DIR / "gpr_p99_min_safe_thickness_bundle.joblib"


# =============================================================================
# Model loading and validation
# =============================================================================
@st.cache_resource(show_spinner="Loading GPR model bundle...")
def load_model_bundle(model_path_str: str) -> Dict[str, Any]:
    """Load the compact deployment bundle once per server session."""
    model_path = Path(model_path_str)

    if not model_path.exists():
        raise FileNotFoundError(
            "Model bundle not found. Place "
            "'gpr_p99_min_safe_thickness_bundle.joblib' in the same folder as "
            "'streamlit_app.py'."
        )

    bundle = joblib.load(model_path)

    required_top_level = {"models", "input_schema", "study_constants", "interpretation"}
    missing = required_top_level.difference(bundle.keys())
    if missing:
        raise KeyError(f"Invalid model bundle. Missing keys: {sorted(missing)}")

    for load_name in ["Vertical", "Oblique"]:
        if load_name not in bundle["models"]:
            raise KeyError(f"Invalid model bundle. Missing {load_name} P99 model.")
        if "pipeline" not in bundle["models"][load_name]:
            raise KeyError(f"Invalid {load_name} model entry. Missing fitted pipeline.")

    return bundle


def get_constants(bundle: Dict[str, Any]) -> Tuple[list, list, list, dict, dict, dict]:
    """Extract model schema and scientific constants from the saved bundle."""
    schema = bundle["input_schema"]
    constants = bundle["study_constants"]

    feature_order = list(schema["model_feature_order"])
    materials = list(schema["automatically_enumerated"]["Material"])
    structures = list(schema["automatically_enumerated"]["Structure"])

    rp02 = dict(constants["Rp0.2_MPa"])
    rules = dict(constants["safety_rules"])

    return feature_order, materials, structures, rp02, rules, schema


# =============================================================================
# Prediction and screening functions
# =============================================================================
def predict_p99(
    bundle: Dict[str, Any],
    load_name: str,
    X: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict P99 mean and GPR standard deviation.
    The fitted pipeline includes the original StandardScaler and OneHotEncoder.
    """
    pipeline = bundle["models"][load_name]["pipeline"]
    mean, std = pipeline.predict(X, return_std=True)

    mean = np.maximum(np.asarray(mean, dtype=float), 0.0)
    std = np.maximum(np.asarray(std, dtype=float), 0.0)
    return mean, std


def round_up_to_increment(value: float, increment: float) -> float:
    """Round upward, avoiding false precision and never reducing a safe thickness."""
    return round(math.ceil((value - 1e-10) / increment) * increment, 4)


def is_observed_design_node(advancement_mm: float, thickness_mm: float) -> bool:
    """
    Identify whether the entered condition coincides with an original FEA design node.
    This labels the *design location*, not the source of the displayed prediction.
    """
    node_thicknesses = {
        2.0: {0.6, 1.0, 1.5, 2.0},
        4.0: {0.6, 0.8, 1.0, 1.2, 1.5, 2.0},
        8.0: {0.6, 1.0, 1.5, 2.0},
    }

    for observed_advancement, observed_thickness_set in node_thicknesses.items():
        if abs(advancement_mm - observed_advancement) < 1e-8:
            return any(abs(thickness_mm - t) < 1e-8 for t in observed_thickness_set)

    return False


def evaluate_six_candidates(
    bundle: Dict[str, Any],
    advancement_mm: float,
    load_name: str,
) -> pd.DataFrame:
    """
    Enumerate Pure Ti / TC4 × S1 / S2 / S3 and identify the minimum
    robustly safe thickness for each combination.

    Safety definition:
        (P99 mean + z * P99 std) / Rp0.2 <= threshold
    """
    feature_order, materials, structures, rp02, rules, _ = get_constants(bundle)

    t_min, t_max = map(float, rules["thickness_domain_mm"])
    search_step = float(rules["search_step_mm"])
    display_increment = float(rules["display_increment_mm"])
    threshold = float(rules["stress_ratio_threshold"])
    z_value = float(rules["prediction_interval_z"])

    # Use a fixed grid defined in the saved bundle, rather than a hard-coded web rule.
    n_points = int(round((t_max - t_min) / search_step)) + 1
    thickness_grid = np.round(np.linspace(t_min, t_max, n_points), 4)

    all_rows = []
    for material in materials:
        for structure in structures:
            for thickness_mm in thickness_grid:
                all_rows.append(
                    {
                        "MaxillaryAdvancement_mm": float(advancement_mm),
                        "Thickness_mm": float(thickness_mm),
                        "Material": material,
                        "Structure": structure,
                    }
                )

    X_grid = pd.DataFrame(all_rows, columns=feature_order)
    p99_mean, p99_std = predict_p99(bundle, load_name, X_grid)

    grid = X_grid.copy()
    grid["P99_mean_MPa"] = p99_mean
    grid["P99_std_MPa"] = p99_std
    grid["P99_upper95_MPa"] = grid["P99_mean_MPa"] + z_value * grid["P99_std_MPa"]
    grid["Rp0.2_MPa"] = grid["Material"].map(rp02)
    grid["StressRatio_upper95"] = grid["P99_upper95_MPa"] / grid["Rp0.2_MPa"]
    grid["RobustlySafe"] = grid["StressRatio_upper95"] <= threshold

    results = []

    for material in materials:
        for structure in structures:
            subset = grid.loc[
                (grid["Material"] == material) & (grid["Structure"] == structure)
            ].sort_values("Thickness_mm")

            first_safe = subset.loc[subset["RobustlySafe"]].head(1)

            # No valid robustly safe design within the validated thickness domain.
            if first_safe.empty:
                results.append(
                    {
                        "Material": material,
                        "Structure": structure,
                        "Minimum robustly safe thickness (mm)": np.nan,
                        "Predicted P99 (MPa)": np.nan,
                        "P99 upper 95% limit (MPa)": np.nan,
                        "Conservative stress ratio": np.nan,
                        "Conservative safety factor": np.nan,
                        "Design status": (
                            f"No robustly safe thickness within {t_min:.2f}–{t_max:.2f} mm"
                        ),
                        "Evidence": "GPR in-domain estimate",
                    }
                )
                continue

            raw_safe_thickness = float(first_safe.iloc[0]["Thickness_mm"])
            reported_thickness = round_up_to_increment(raw_safe_thickness, display_increment)

            # Rounding upward may move to a different point; reevaluate the reported design.
            if reported_thickness > t_max + 1e-8:
                results.append(
                    {
                        "Material": material,
                        "Structure": structure,
                        "Minimum robustly safe thickness (mm)": np.nan,
                        "Predicted P99 (MPa)": np.nan,
                        "P99 upper 95% limit (MPa)": np.nan,
                        "Conservative stress ratio": np.nan,
                        "Conservative safety factor": np.nan,
                        "Design status": (
                            f"No reportable robustly safe thickness within {t_min:.2f}–{t_max:.2f} mm"
                        ),
                        "Evidence": "GPR in-domain estimate",
                    }
                )
                continue

            X_reported = pd.DataFrame(
                [
                    {
                        "MaxillaryAdvancement_mm": float(advancement_mm),
                        "Thickness_mm": float(reported_thickness),
                        "Material": material,
                        "Structure": structure,
                    }
                ],
                columns=feature_order,
            )
            reported_mean, reported_std = predict_p99(bundle, load_name, X_reported)

            p99_mean_value = float(reported_mean[0])
            p99_upper_value = float(reported_mean[0] + z_value * reported_std[0])
            ratio_upper = p99_upper_value / float(rp02[material])
            safety_factor = (
                float(rp02[material]) / p99_upper_value
                if p99_upper_value > 0
                else np.inf
            )

            if reported_thickness <= t_min + 1e-8:
                status = "Robustly safe at the minimum modelled thickness"
            else:
                status = "Robustly safe"

            evidence = (
                "Observed FEA design node"
                if is_observed_design_node(advancement_mm, reported_thickness)
                else "GPR in-domain interpolation"
            )

            results.append(
                {
                    "Material": material,
                    "Structure": structure,
                    "Minimum robustly safe thickness (mm)": reported_thickness,
                    "Predicted P99 (MPa)": p99_mean_value,
                    "P99 upper 95% limit (MPa)": p99_upper_value,
                    "Conservative stress ratio": ratio_upper,
                    "Conservative safety factor": safety_factor,
                    "Design status": status,
                    "Evidence": evidence,
                }
            )

    result_df = pd.DataFrame(results)

    # Preserve the manuscript order: Pure Ti/TC4 and S1/S2/S3.
    material_rank = {m: i for i, m in enumerate(materials)}
    structure_rank = {s: i for i, s in enumerate(structures)}
    result_df["_material_rank"] = result_df["Material"].map(material_rank)
    result_df["_structure_rank"] = result_df["Structure"].map(structure_rank)
    result_df = (
        result_df.sort_values(["_material_rank", "_structure_rank"])
        .drop(columns=["_material_rank", "_structure_rank"])
        .reset_index(drop=True)
    )

    return result_df


def make_display_table(result_df: pd.DataFrame) -> pd.DataFrame:
    """Format values for clear display without changing downloadable numeric results."""
    display = result_df.copy()

    numeric_cols = [
        "Minimum robustly safe thickness (mm)",
        "Predicted P99 (MPa)",
        "P99 upper 95% limit (MPa)",
        "Conservative stress ratio",
        "Conservative safety factor",
    ]

    for col in numeric_cols:
        display[col] = display[col].map(
            lambda x: "—" if pd.isna(x) or np.isinf(x) else f"{float(x):.2f}"
        )

    return display


# =============================================================================
# App layout
# =============================================================================
try:
    bundle = load_model_bundle(str(MODEL_BUNDLE_FILE))
except Exception as exc:
    st.error("The model bundle could not be loaded.")
    st.exception(exc)
    st.stop()

feature_order, materials, structures, rp02, rules, _ = get_constants(bundle)
threshold = float(rules["stress_ratio_threshold"])
z_value = float(rules["prediction_interval_z"])
adv_min, adv_max = map(float, rules["advancement_domain_mm"])
t_min, t_max = map(float, rules["thickness_domain_mm"])

st.title("Le Fort I Personalized Titanium Plate Calculator")
st.caption(
    "FEA-informed GPR calculator for minimum robustly safe thickness screening "
    "across all material–structure combinations."
)

with st.sidebar:
    st.header("Input")

    advancement_mm = st.slider(
        "Maxillary advancement (mm)",
        min_value=float(adv_min),
        max_value=float(adv_max),
        value=4.0,
        step=0.1,
        help="Validated model domain: 2.0–8.0 mm.",
    )

    load_display = st.radio(
        "Loading mode",
        options=[
            "Vertical (200 N)",
            "Oblique (120 N, 60°)",
        ],
        index=1,
    )
    load_name = "Vertical" if load_display.startswith("Vertical") else "Oblique"

    calculate = st.button("Calculate six candidate designs", use_container_width=True)

    st.divider()
    st.caption(
        "Safety rule: upper 95% model-estimated P99 stress ratio "
        f"≤ {threshold:.2f}."
    )
    st.caption(
        f"Thickness search domain: {t_min:.2f}–{t_max:.2f} mm. "
        "Reported values are rounded upward to 0.05 mm."
    )

if "results" not in st.session_state:
    st.session_state["results"] = None
    st.session_state["result_input"] = None

if calculate:
    with st.spinner("Screening six material–structure combinations..."):
        st.session_state["results"] = evaluate_six_candidates(
            bundle=bundle,
            advancement_mm=advancement_mm,
            load_name=load_name,
        )
        st.session_state["result_input"] = {
            "advancement_mm": advancement_mm,
            "load_name": load_name,
            "load_display": load_display,
        }

if st.session_state["results"] is None:
    st.info("Select the advancement and loading mode, then click “Calculate six candidate designs”.")
    st.stop()

result_df = st.session_state["results"]
result_input = st.session_state["result_input"]

st.subheader("Six material–structure candidate designs")
st.write(
    f"**Input:** {result_input['advancement_mm']:.1f} mm advancement; "
    f"**{result_input['load_display']}**."
)

safe_count = int(result_df["Minimum robustly safe thickness (mm)"].notna().sum())
col1, col2, col3 = st.columns(3)
col1.metric("Robustly safe candidates", f"{safe_count}/6")
col2.metric("Safety threshold", f"Stress ratio ≤ {threshold:.2f}")
col3.metric("Prediction interval", f"Upper 95% limit (z = {z_value:.2f})")

display_df = make_display_table(result_df)
st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Minimum robustly safe thickness (mm)": st.column_config.TextColumn(
            "Minimum robustly safe thickness (mm)"
        ),
        "Predicted P99 (MPa)": st.column_config.TextColumn("Predicted P99 (MPa)"),
        "P99 upper 95% limit (MPa)": st.column_config.TextColumn(
            "P99 upper 95% limit (MPa)"
        ),
        "Conservative stress ratio": st.column_config.TextColumn(
            "Conservative stress ratio"
        ),
        "Conservative safety factor": st.column_config.TextColumn(
            "Conservative safety factor"
        ),
    },
)

# Simple visual comparison of safe thicknesses; no ranking or clinical recommendation.
chart_data = result_df.dropna(subset=["Minimum robustly safe thickness (mm)"]).copy()
if not chart_data.empty:
    chart_data["Candidate"] = chart_data["Material"] + " – " + chart_data["Structure"]
    chart_data = chart_data.set_index("Candidate")[["Minimum robustly safe thickness (mm)"]]
    st.subheader("Minimum robustly safe thickness comparison")
    st.bar_chart(chart_data, use_container_width=True)

csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="Download results as CSV",
    data=csv_bytes,
    file_name=(
        f"LeFortI_GPR_min_safe_thickness_"
        f"{result_input['load_name']}_{result_input['advancement_mm']:.1f}mm.csv"
    ),
    mime="text/csv",
)

st.divider()
st.subheader("Interpretation and scope")
st.markdown(
    f"""
- **Minimum robustly safe thickness** is the smallest thickness within the validated domain
  for which `(P99 upper 95% model-estimated limit / Rp0.2) ≤ {threshold:.2f}`.
- This calculator enumerates all six material–structure combinations and does **not**
  issue a clinical recommendation or rank a single “best” design.
- Predictions are valid only within **{adv_min:.1f}–{adv_max:.1f} mm** maxillary advancement
  and **{t_min:.1f}–{t_max:.1f} mm** plate thickness.
- Designs near the safety boundary require confirmatory finite element analysis.
- This is a **research-use-only, FEA-informed design-support prototype** and is not a
  substitute for clinical judgment.
"""
)

with st.expander("Model and study information"):
    st.write(f"Model bundle: `{bundle.get('bundle_version', 'not specified')}`")
    st.write(
        "Material-level tensile Rp0.2 references: "
        + "; ".join(f"{m}: {rp02[m]:.2f} MPa" for m in materials)
        + "."
    )
    st.write(
        "The displayed P99 values are GPR predictions. "
        "“Observed FEA design node” indicates that the reported input location "
        "coincides with an original FEA design node; it does not replace the original FEA result."
    )
