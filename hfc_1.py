"""
hfc_quality_engine.py
=======================
HIGH-FREQUENCY CHECK (HFC) & DATA QUALITY ENGINE
--------------------------------------------------
A single-file OOP project demonstrating the four pillars of OOP:
ABSTRACTION, INHERITANCE, POLYMORPHISM, and ENCAPSULATION.

CONTEXT:
    When fielding live surveys (e.g. via SurveyCTO / ODK / KoboToolbox),
    field managers need automated daily audits to catch enumerator
    fraud, outlier values, and duration anomalies before data
    collection closes. This is what an HFC (High Frequency Check)
    pipeline does at organizations like IDinsight, J-PAL, and DIME
    (World Bank).

HOW TO RUN THIS FILE:
    python hfc_quality_engine.py

    (requires: pip install pandas numpy)

STRUCTURE OF THIS FILE (in order):
    1. Imports
    2. QualityCheck        - abstract base class          -> ABSTRACTION
    3. DurationCheck        \
       OutlierCheck          |- concrete check subclasses  -> INHERITANCE
       MissingnessCheck      |                                & POLYMORPHISM
       DuplicateGPSCheck    /
    4. SurveyData           - wraps data, hides PII        -> ENCAPSULATION
    5. DataAuditor          - runs all checks uniformly    -> POLYMORPHISM
    6. generate_sample_data - fake data generator (demo only)
    7. main()               - ties everything together and runs the demo
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)
np.random.seed(42)  # reproducible fake data for the demo


# ============================================================
# 2. ABSTRACTION
# ============================================================
class QualityCheck(ABC):
    """
    Abstract base class for all quality checks.

    WHY THIS DEMONSTRATES ABSTRACTION:
        Nobody using this system needs to know HOW a check works
        internally (what formula OutlierCheck uses, what threshold
        DurationCheck applies). They only need to know every check
        object has a `.run_check(df)` method they can call.

    Python's `abc` module ENFORCES this: you cannot create a
    QualityCheck directly, and any subclass that forgets to implement
    `run_check` will raise an error.
    """

    def __init__(self, name: str):
        # Shared setup every check inherits - see child classes' super().__init__()
        self.name = name

    @abstractmethod
    def run_check(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Must be implemented by every subclass.
        Should return a DataFrame of FLAGGED rows only, with columns:
        ['survey_id', 'enumerator_id', 'flag_type', 'flag_reason']
        """
        pass

    def __str__(self):
        return self.name


# ============================================================
# 3. INHERITANCE + POLYMORPHISM (concrete checks)
# ============================================================
class DurationCheck(QualityCheck):
    """Flags surveys completed suspiciously fast or slow."""

    def __init__(self, min_minutes: float = 10, max_minutes: float = 90):
        super().__init__(name="Duration Check")  # reuse parent's setup -> INHERITANCE
        self.min_minutes = min_minutes
        self.max_minutes = max_minutes

    def run_check(self, df: pd.DataFrame) -> pd.DataFrame:
        # Own implementation of run_check -> this + the shared call
        # signature is what makes POLYMORPHISM possible in DataAuditor.
        flagged_rows = []
        for _, row in df.iterrows():
            duration = row["duration_minutes"]
            if duration < self.min_minutes:
                flagged_rows.append({
                    "survey_id": row["survey_id"],
                    "enumerator_id": row["enumerator_id"],
                    "flag_type": self.name,
                    "flag_reason": f"Too fast: {duration} min "
                                   f"(minimum expected {self.min_minutes} min)"
                })
            elif duration > self.max_minutes:
                flagged_rows.append({
                    "survey_id": row["survey_id"],
                    "enumerator_id": row["enumerator_id"],
                    "flag_type": self.name,
                    "flag_reason": f"Too slow: {duration} min "
                                   f"(maximum expected {self.max_minutes} min)"
                })
        return pd.DataFrame(flagged_rows)


class OutlierCheck(QualityCheck):
    """Flags numeric responses outside N standard deviations from the mean."""

    def __init__(self, column: str, n_std: float = 3):
        super().__init__(name=f"Outlier Check ({column})")
        self.column = column
        self.n_std = n_std

    def run_check(self, df: pd.DataFrame) -> pd.DataFrame:
        mean = df[self.column].mean()
        std = df[self.column].std()
        lower_bound = mean - (self.n_std * std)
        upper_bound = mean + (self.n_std * std)

        flagged_rows = []
        for _, row in df.iterrows():
            value = row[self.column]
            if value < lower_bound or value > upper_bound:
                flagged_rows.append({
                    "survey_id": row["survey_id"],
                    "enumerator_id": row["enumerator_id"],
                    "flag_type": self.name,
                    "flag_reason": f"{self.column}={value} is outside "
                                   f"[{lower_bound:.0f}, {upper_bound:.0f}]"
                })
        return pd.DataFrame(flagged_rows)


class MissingnessCheck(QualityCheck):
    """Flags surveys with too many blank / refused responses."""

    def __init__(self, columns: list, max_missing_allowed: int = 2):
        super().__init__(name="Missingness Check")
        self.columns = columns
        self.max_missing_allowed = max_missing_allowed

    def run_check(self, df: pd.DataFrame) -> pd.DataFrame:
        flagged_rows = []
        for _, row in df.iterrows():
            missing_count = row[self.columns].isna().sum()
            if missing_count > self.max_missing_allowed:
                flagged_rows.append({
                    "survey_id": row["survey_id"],
                    "enumerator_id": row["enumerator_id"],
                    "flag_type": self.name,
                    "flag_reason": f"{missing_count} missing responses "
                                   f"(max allowed: {self.max_missing_allowed})"
                })
        return pd.DataFrame(flagged_rows)


class DuplicateGPSCheck(QualityCheck):
    """Flags surveys submitted from the exact same GPS coordinates."""

    def __init__(self):
        super().__init__(name="Duplicate GPS Check")

    def run_check(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["gps_rounded"] = (
            df["gps_lat"].round(4).astype(str) + "_" +
            df["gps_lon"].round(4).astype(str)
        )
        duplicate_locations = df["gps_rounded"].value_counts()
        duplicate_locations = duplicate_locations[duplicate_locations > 1]

        flagged_rows = []
        for _, row in df.iterrows():
            if row["gps_rounded"] in duplicate_locations.index:
                flagged_rows.append({
                    "survey_id": row["survey_id"],
                    "enumerator_id": row["enumerator_id"],
                    "flag_type": self.name,
                    "flag_reason": f"Shared GPS location with "
                                   f"{duplicate_locations[row['gps_rounded']] - 1} "
                                   f"other survey(s)"
                })
        return pd.DataFrame(flagged_rows)


# ============================================================
# 4. ENCAPSULATION
# ============================================================
class SurveyData:
    """
    Wraps the raw survey dataframe and hides sensitive (PII) columns.

    WHY THIS DEMONSTRATES ENCAPSULATION:
        __raw_data and __pii_columns use Python's double-underscore
        "name mangling", which makes them PRIVATE - code outside this
        class cannot access survey_data.__raw_data directly. Instead,
        outside code must go through controlled public methods below.
    """

    def __init__(self, raw_dataframe: pd.DataFrame, pii_columns: list):
        self.__raw_data = raw_dataframe          # PRIVATE
        self.__pii_columns = pii_columns          # PRIVATE

    def get_data_for_checks(self) -> pd.DataFrame:
        """Controlled access: checks need full data (incl. GPS/enumerator ID)."""
        return self.__raw_data.copy()

    def get_anonymized_data(self) -> pd.DataFrame:
        """
        Controlled access: strips PII columns entirely - safe to share.
        NOTE: this is still row-level data (every survey record), just
        with sensitive columns removed - NOT an aggregated summary.
        Use get_summary_stats() below for real aggregates.
        """
        return self.__raw_data.drop(columns=self.__pii_columns, errors="ignore")

    def get_summary_stats(self) -> dict:
        """
        Controlled access: returns a genuine AGGREGATED summary - counts
        and groupings, never individual survey rows. Safe to share
        externally since no row-level data is exposed at all.
        """
        anonymized = self.get_anonymized_data()
        return {
            "total_surveys": len(anonymized),
            "enumerators_count": anonymized["enumerator_id"].nunique()
                                  if "enumerator_id" in anonymized.columns else None,
            "surveys_per_enumerator": (
                anonymized["enumerator_id"].value_counts().to_dict()
                if "enumerator_id" in anonymized.columns else {}
            ),
            "numeric_column_averages": (
                anonymized.select_dtypes(include="number").mean().round(1).to_dict()
            ),
        }

    def get_row_count(self) -> int:
        return len(self.__raw_data)

    def __str__(self):
        return (f"SurveyData({self.get_row_count()} surveys, "
                f"{len(self.__pii_columns)} PII columns protected)")


# ============================================================
# 5. POLYMORPHISM in action (the orchestrator)
# ============================================================
class DataAuditor:
    """
    Runs a LIST of different check objects against a SurveyData object,
    treating every check identically regardless of its internal logic.

    WHY THIS IS WHERE POLYMORPHISM REALLY SHINES:
        run_all_checks() loops through self.checks and calls
        check.run_check(df) on each - no if/elif chain needed to
        figure out "which kind" of check it is. Adding a brand new
        check type later requires ZERO changes to this class.
    """

    def __init__(self, survey_data: SurveyData, checks: list):
        self.survey_data = survey_data
        for check in checks:
            if not isinstance(check, QualityCheck):
                raise TypeError(
                    f"{check} is not a QualityCheck - it must inherit "
                    f"from QualityCheck to be used by DataAuditor."
                )
        self.checks = checks

    def run_all_checks(self) -> pd.DataFrame:
        df = self.survey_data.get_data_for_checks()
        all_flags = []
        for check in self.checks:
            print(f"Running: {check}...")
            flags_from_this_check = check.run_check(df)  # <- polymorphic call
            if not flags_from_this_check.empty:
                all_flags.append(flags_from_this_check)

        if all_flags:
            return pd.concat(all_flags, ignore_index=True)
        return pd.DataFrame(columns=["survey_id", "enumerator_id", "flag_type", "flag_reason"])

    def summarize_by_enumerator(self, report: pd.DataFrame) -> pd.DataFrame:
        if report.empty:
            return pd.DataFrame(columns=["enumerator_id", "total_flags"])
        return (
            report.groupby("enumerator_id")
            .size()
            .reset_index(name="total_flags")
            .sort_values("total_flags", ascending=False)
        )


# ============================================================
# 6. Fake data generator (for demo purposes only)
# ============================================================
def generate_sample_data(n_surveys: int = 30) -> pd.DataFrame:
    """Creates a fake survey dataset, with some records deliberately
    'broken' so the checks above have something real to flag."""
    enumerator_ids = [f"ENUM{str(i).zfill(3)}" for i in range(1, 6)]

    data = {
        "survey_id": [f"SVY{str(i).zfill(4)}" for i in range(1, n_surveys + 1)],
        "enumerator_id": np.random.choice(enumerator_ids, n_surveys),
        "duration_minutes": np.random.normal(loc=30, scale=6, size=n_surveys).round(1),
        "monthly_income_ugx": np.random.normal(loc=450000, scale=80000, size=n_surveys).round(0),
        "household_size": np.random.randint(1, 9, size=n_surveys),
        "gps_lat": np.random.uniform(0.30, 0.35, size=n_surveys).round(5),
        "gps_lon": np.random.uniform(32.55, 32.60, size=n_surveys).round(5),
        "respondent_name": [f"Respondent_{i}" for i in range(1, n_surveys + 1)],  # PII
    }
    df = pd.DataFrame(data)

    # Deliberately inject "bad" records so the checks have something to catch
    df.loc[0, "duration_minutes"] = 3.2                      # too fast
    df.loc[1, "duration_minutes"] = 145.0                     # too slow
    df.loc[2, "monthly_income_ugx"] = 9500000                 # extreme outlier
    df.loc[3, ["gps_lat", "gps_lon"]] = [0.31500, 32.58200]   # duplicate GPS pair
    df.loc[4, ["gps_lat", "gps_lon"]] = [0.31500, 32.58200]
    df.loc[5, "monthly_income_ugx"] = np.nan                  # missingness
    df.loc[5, "household_size"] = np.nan

    return df


# ============================================================
# 7. Demo entry point
# ============================================================
def main():
    print("=" * 70)
    print("HIGH-FREQUENCY CHECK (HFC) & DATA QUALITY ENGINE")
    print("=" * 70)

    # STEP 1: Load data (swap this line for pd.read_csv(...) to use real data)
    raw_df = generate_sample_data(n_surveys=30)
    print(f"\nLoaded {len(raw_df)} survey records.\n")

    # STEP 2: Wrap in SurveyData -> ENCAPSULATION
    survey_data = SurveyData(raw_dataframe=raw_df, pii_columns=["respondent_name"])
    print(survey_data)

    # STEP 3: Create check objects -> INHERITANCE
    checks = [
        DurationCheck(min_minutes=10, max_minutes=90),
        OutlierCheck(column="monthly_income_ugx", n_std=3),
        MissingnessCheck(columns=["monthly_income_ugx", "household_size"], max_missing_allowed=1),
        DuplicateGPSCheck(),
    ]

    # STEP 4: Run everything -> POLYMORPHISM
    auditor = DataAuditor(survey_data=survey_data, checks=checks)
    print("\n--- Running all quality checks ---")
    report = auditor.run_all_checks()

    print("\n--- FLAGGED RECORDS REPORT ---")
    print("No issues found. Data looks clean!" if report.empty else report.to_string(index=False))

    # STEP 5: Summarize -> ABSTRACTION (simple method, hidden internals)
    print("\n--- FLAGS PER ENUMERATOR (for supervisor follow-up) ---")
    summary = auditor.summarize_by_enumerator(report)
    print(summary.to_string(index=False))

    print("\n--- Anonymized summary (safe to share externally, no PII) ---")
    print(survey_data.get_anonymized_data().head().to_string(index=False))


if __name__ == "__main__":
    main()
