"""
app.py
------
This is the STREAMLIT WEB INTERFACE for your HFC Quality Engine.

WHY THIS IS A SEPARATE FILE FROM hfc_quality_engine_single.py:
    hfc_quality_engine_single.py contains your actual OOP logic - the
    classes that demonstrate Abstraction, Inheritance, Polymorphism, and
    Encapsulation. That file doesn't know or care whether it's being run
    from a terminal, a notebook, or a website - it just does the auditing.

    app.py's ONLY job is to build a simple webpage AROUND that existing
    logic - a button to click, a table to show results in. It IMPORTS
    the classes from your engine file rather than rewriting them.

    This separation is actually good software design: your "engine"
    (the OOP logic) stays reusable and independent from any particular
    way of presenting it (terminal vs. web vs. notebook). You could
    swap this Streamlit interface out for a different one later without
    touching your OOP classes at all.

HOW THEY CONNECT:
    The line below - `from hfc_quality_engine_single import ...` - is
    what pulls in your classes. This ONLY works if app.py sits in the
    SAME FOLDER as hfc_quality_engine_single.py.

HOW TO RUN THIS LOCALLY (on your own computer / "localhost"):
    1. Open a terminal in the folder containing both files
    2. Run:  streamlit run app.py
    3. It will automatically open a browser tab at something like
       http://localhost:8501 - that IS your "localhost" demo link.

HOW TO GET A REAL PUBLIC LINK (Streamlit Community Cloud):
    1. Push this folder (app.py + hfc_quality_engine_single.py +
       requirements.txt) to a GitHub repository
    2. Go to https://share.streamlit.io , sign in with GitHub
    3. Click "New app", select your repo, and point it at app.py
    4. Streamlit Cloud installs your requirements.txt and gives you a
       public URL like https://your-app-name.streamlit.app
"""

import streamlit as st
import pandas as pd

# Importing everything we need from your existing OOP engine file.
# Nothing below this line rewrites any OOP logic - it all comes from there.
from hfc_1 import (
    DurationCheck,
    OutlierCheck,
    MissingnessCheck,
    DuplicateGPSCheck,
    SurveyData,
    DataAuditor,
    generate_sample_data,
)

# --- Page setup ------------------------------------------------------------
st.set_page_config(page_title="HFC Quality Engine", layout="wide")

st.title("📋 High-Frequency Check (HFC) & Data Quality Engine")
st.markdown(
    "An OOP-based tool demonstrating **Abstraction, Inheritance, "
    "Polymorphism, and Encapsulation** applied to survey data quality checks."
)

# --- Sidebar: choose data source --------------------------------------------
st.sidebar.header("1. Choose your data")
data_source = st.sidebar.radio(
    "Data source",
    ["Use sample (fake) data", "Upload my own CSV"],
)

uploaded_file = None
if data_source == "Upload my own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    st.sidebar.caption(
        "Your CSV must include these columns: survey_id, enumerator_id, "
        "duration_minutes, monthly_income_ugx, household_size, gps_lat, gps_lon."
    )

# --- Sidebar: choose which checks to run ------------------------------------
st.sidebar.header("2. Choose checks to run")
run_duration = st.sidebar.checkbox("Duration Check", value=True)
run_outlier = st.sidebar.checkbox("Outlier Check (income)", value=True)
run_missing = st.sidebar.checkbox("Missingness Check", value=True)
run_gps = st.sidebar.checkbox("Duplicate GPS Check", value=True)

run_button = st.sidebar.button("🚀 Run Quality Checks", type="primary")

# --- Load the data -----------------------------------------------------------
if data_source == "Use sample (fake) data":
    raw_df = generate_sample_data(n_surveys=30)
else:
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = None

# --- Wrap the data in SurveyData right away -> ENCAPSULATION -----------------
# This happens BEFORE any preview is shown, so PII (e.g. respondent_name)
# is protected from the very first moment the data is loaded - not just
# once the checks run. This is what makes the encapsulation real rather
# than cosmetic.
survey_data = None
if raw_df is not None:
    pii_cols = ["respondent_name"] if "respondent_name" in raw_df.columns else []
    survey_data = SurveyData(raw_dataframe=raw_df, pii_columns=pii_cols)

# --- Show a preview of the data (PII-safe, via the anonymized getter) --------
if survey_data is not None:
    with st.expander("Preview data (before checks) - PII removed"):
        st.dataframe(survey_data.get_anonymized_data(), use_container_width=True)

# --- Run the checks when the button is clicked -------------------------------
if run_button:
    if raw_df is None or survey_data is None:
        st.error("Please upload a CSV file first, or switch to sample data.")
    else:
        # STEP 2: Build the list of check objects based on sidebar choices
        # -> INHERITANCE (every item below is a QualityCheck subclass)
        checks = []
        if run_duration:
            checks.append(DurationCheck(min_minutes=10, max_minutes=90))
        if run_outlier and "monthly_income_ugx" in raw_df.columns:
            checks.append(OutlierCheck(column="monthly_income_ugx", n_std=3))
        if run_missing:
            missing_cols = [c for c in ["monthly_income_ugx", "household_size"]
                             if c in raw_df.columns]
            if missing_cols:
                checks.append(MissingnessCheck(columns=missing_cols,
                                                max_missing_allowed=1))
        if run_gps and {"gps_lat", "gps_lon"}.issubset(raw_df.columns):
            checks.append(DuplicateGPSCheck())

        if not checks:
            st.warning("No checks selected (or your data is missing the "
                       "columns needed for the checks you picked).")
        else:
            # STEP 3: Run everything through DataAuditor -> POLYMORPHISM
            auditor = DataAuditor(survey_data=survey_data, checks=checks)
            report = auditor.run_all_checks()

            st.subheader("🚩 Flagged Records Report")
            if report.empty:
                st.success("No issues found. Data looks clean!")
            else:
                st.dataframe(report, use_container_width=True)

                st.subheader("📊 Flags per Enumerator")
                summary = auditor.summarize_by_enumerator(report)
                st.dataframe(summary, use_container_width=True)

                # A quick bar chart, since Streamlit makes this almost free
                st.bar_chart(summary.set_index("enumerator_id")["total_flags"])

            # STEP 4: Show aggregate summary stats -> ENCAPSULATION proof
            # (Note: the anonymized row-level data is already shown in the
            # preview above, before checks run - no need to repeat it here.)
            st.subheader("📈 Summary Statistics")
            st.caption("Aggregated only - no individual survey rows shown here.")
            stats = survey_data.get_summary_stats()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total surveys", stats["total_surveys"])
                st.metric("Enumerators", stats["enumerators_count"])
            with col2:
                st.write("**Surveys per enumerator**")
                surveys_per_enum_df=pd.DataFrame(
                    list(stats["surveys_per_enumerator"].items()),
                    columns=["Enumerator","Surveys"]
                )
                st.dataframe(surveys_per_enum_df,use_container_width=True,
                             hide_index=True)
                st.write("**Avergae values (numeric columns)**")
                avg_df=pd.DataFrame(
                    list(stats["numeric_column_averages"].items()),
                    columns=["Column","Average"]
                )
                st.dataframe(avg_df,ue_container_widith=True,hide_index=True)         
else:
    st.info("Choose your data and checks in the sidebar, then click "
            "**Run Quality Checks**.")
