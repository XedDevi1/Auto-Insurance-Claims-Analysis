import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ścieżki do plików wejściowych i wyjściowych
RAW_CLAIMS_PATH = os.path.join("data", "raw", "insurance_claims.csv")
RAW_FRAUD_PATH = os.path.join("data", "raw", "fraud_oracle.csv")
PROCESSED_FRAUD_PATH = os.path.join("data", "processed", "fraud_oracle_clean.csv")
FIGURES_DIR = os.path.join("reports", "figures")

os.makedirs(os.path.dirname(PROCESSED_FRAUD_PATH), exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Słowniki mapujące przedziały tekstowe z pliku fraud_oracle.csv na wartości liczbowe
VEHICLE_PRICE_MAP = {
    "less than 20000": 15000,
    "20000 to 29000": 24500,
    "30000 to 39000": 34500,
    "40000 to 59000": 49500,
    "60000 to 69000": 64500,
    "more than 69000": 80000,
}

AGE_OF_VEHICLE_MAP = {
    "new": 0,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "more than 7": 9,
}

PAST_CLAIMS_MAP = {
    "none": 0,
    "1": 1,
    "2 to 4": 3,
    "more than 4": 5,
}

DAYS_POLICY_MAP = {
    "none": 0,
    "1 to 7": 4,
    "8 to 15": 11.5,
    "15 to 30": 22.5,
    "more than 30": 45,
}

SUPPLIMENTS_MAP = {
    "none": 0,
    "1 to 2": 1.5,
    "3 to 5": 4,
    "more than 5": 6,
}


def generate_fallback_datasets() -> None:
    """Generuje pliki demonstracyjne o identycznej strukturze kolumn, jeśli pliki CSV nie są jeszcze obecne."""
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    rng = np.random.default_rng(42)

    if not os.path.exists(RAW_CLAIMS_PATH):
        n = 1000
        months = rng.integers(1, 480, size=n)
        age = np.clip((months // 12) + rng.integers(18, 35, size=n), 18, 75)
        auto_year = rng.integers(1995, 2016, size=n)
        vehicle_age = 2015 - auto_year
        deductable = rng.choice([500, 1000, 2000], size=n)
        annual_premium = np.clip(
            1400 - 15 * vehicle_age - 1.2 * months + rng.normal(0, 180, size=n),
            400,
            2500,
        )
        # Rozkład silnie prawostronnie skośny dla kwoty roszczenia
        total_claim = np.round(rng.lognormal(mean=9.8, sigma=0.95, size=n), 2)
        vehicle_claim = np.round(total_claim * 0.72, 2)
        injury_claim = np.round(total_claim * 0.15, 2)
        property_claim = np.round(total_claim - vehicle_claim - injury_claim, 2)

        df_claims = pd.DataFrame({
            "months_as_customer": months,
            "age": age,
            "policy_number": rng.integers(100000, 999999, size=n),
            "policy_bind_date": "2014-01-01",
            "policy_state": rng.choice(["OH", "IL", "IN"], size=n),
            "policy_csl": rng.choice(["100/300", "250/500", "500/1000"], size=n),
            "policy_deductable": deductable,
            "policy_annual_premium": np.round(annual_premium, 2),
            "umbrella_limit": rng.choice([0, 1000000, 2000000, 5000000], size=n, p=[0.8, 0.1, 0.05, 0.05]),
            "insured_zip": rng.integers(430000, 620000, size=n),
            "insured_sex": rng.choice(["MALE", "FEMALE"], size=n),
            "insured_education_level": rng.choice(["JD", "High School", "Associate", "MD", "Masters", "PhD", "College"], size=n),
            "insured_occupation": rng.choice(["craft-repair", "machine-op-inspct", "sales", "tech-support", "exec-managerial"], size=n),
            "insured_hobbies": rng.choice(["reading", "exercise", "movies", "camping", "chess", "cross-fit"], size=n),
            "insured_relationship": rng.choice(["husband", "other-relative", "own-child", "unmarried", "wife"], size=n),
            "capital-gains": rng.choice([0, 25000, 50000], size=n, p=[0.6, 0.25, 0.15]),
            "capital-loss": rng.choice([0, -30000, -50000], size=n, p=[0.6, 0.25, 0.15]),
            "incident_date": "2015-01-15",
            "incident_type": rng.choice(["Single Vehicle Collision", "Multi-vehicle Collision", "Parked Car", "Vehicle Theft"], size=n),
            "collision_type": rng.choice(["Side Collision", "Rear Collision", "Front Collision", "?"], size=n, p=[0.3, 0.3, 0.3, 0.1]),
            "incident_severity": rng.choice(["Minor Damage", "Total Loss", "Major Damage", "Trivial Damage"], size=n),
            "authorities_contacted": rng.choice(["Police", "Fire", "Ambulance", "Other", "None"], size=n),
            "incident_state": rng.choice(["NY", "SC", "WV", "VA", "NC", "PA", "OH"], size=n),
            "incident_city": rng.choice(["Columbus", "Riverwood", "Arlington", "Springfield", "Hillsdale"], size=n),
            "incident_location": rng.choice(["Intersection Ridge", "Highway Ave", "Main Crossing", "Elm Street"], size=n),
            "incident_hour_of_the_day": rng.integers(0, 24, size=n),
            "number_of_vehicles_involved": rng.integers(1, 5, size=n),
            "property_damage": rng.choice(["YES", "NO", "?"], size=n, p=[0.35, 0.35, 0.30]),
            "bodily_injuries": rng.integers(0, 3, size=n),
            "witnesses": rng.integers(0, 4, size=n),
            "police_report_available": rng.choice(["YES", "NO", "?"], size=n, p=[0.35, 0.35, 0.30]),
            "total_claim_amount": total_claim,
            "injury_claim": injury_claim,
            "property_claim": property_claim,
            "vehicle_claim": vehicle_claim,
            "auto_make": rng.choice(["Saab", "Mercedes", "Dodge", "Chevrolet", "Accura", "Ford", "Toyota", "Audi"], size=n),
            "auto_model": rng.choice(["92x", "E400", "RAM", "Tahoe", "TL", "F150", "Camry", "A5"], size=n),
            "auto_year": auto_year,
            "fraud_reported": rng.choice(["Y", "N"], size=n, p=[0.24, 0.76]),
            "_c39": np.nan,
        })
        df_claims.to_csv(RAW_CLAIMS_PATH, index=False)

    if not os.path.exists(RAW_FRAUD_PATH):
        n_f = 1500
        df_fraud = pd.DataFrame({
            "Month": rng.choice(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], size=n_f),
            "WeekOfMonth": rng.integers(1, 6, size=n_f),
            "DayOfWeek": rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], size=n_f),
            "Make": rng.choice(["Honda", "Toyota", "Ford", "Mazda", "Chevrolet", "Pontiac", "Accura"], size=n_f),
            "AccidentArea": rng.choice(["Urban", "Rural"], size=n_f, p=[0.85, 0.15]),
            "DayOfWeekClaimed": rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], size=n_f),
            "MonthClaimed": rng.choice(["Jan", "Feb", "Mar", "Apr", "May", "Jun"], size=n_f),
            "WeekOfMonthClaimed": rng.integers(1, 6, size=n_f),
            "Sex": rng.choice(["Male", "Female"], size=n_f),
            "MaritalStatus": rng.choice(["Married", "Single", "Divorced", "Widow"], size=n_f),
            "Age": rng.integers(18, 80, size=n_f),
            "Fault": rng.choice(["Policy Holder", "Third Party"], size=n_f),
            "PolicyType": rng.choice(["Sedan - Liability", "Sedan - Collision", "Sedan - All Perils", "Sport - Collision"], size=n_f),
            "VehicleCategory": rng.choice(["Sedan", "Sport", "Utility"], size=n_f),
            "VehiclePrice": rng.choice(list(VEHICLE_PRICE_MAP.keys()), size=n_f),
            "FraudFound_P": rng.choice([0, 1], size=n_f, p=[0.94, 0.06]),
            "PolicyNumber": np.arange(1, n_f + 1),
            "RepNumber": rng.integers(1, 17, size=n_f),
            "Deductible": rng.choice([300, 400, 500, 700], size=n_f),
            "DriverRating": rng.integers(1, 5, size=n_f),
            "Days_Policy_Accident": rng.choice(list(DAYS_POLICY_MAP.keys()), size=n_f),
            "Days_Policy_Claim": rng.choice(list(DAYS_POLICY_MAP.keys()), size=n_f),
            "PastNumberOfClaims": rng.choice(list(PAST_CLAIMS_MAP.keys()), size=n_f),
            "AgeOfVehicle": rng.choice(list(AGE_OF_VEHICLE_MAP.keys()), size=n_f),
            "AgeOfPolicyHolder": rng.choice(["26 to 30", "31 to 35", "36 to 40", "41 to 50", "51 to 65"], size=n_f),
            "PoliceReportFiled": rng.choice(["No", "Yes"], size=n_f),
            "WitnessPresent": rng.choice(["No", "Yes"], size=n_f),
            "AgentType": rng.choice(["External", "Internal"], size=n_f),
            "NumberOfSuppliments": rng.choice(list(SUPPLIMENTS_MAP.keys()), size=n_f),
            "AddressChange_Claim": rng.choice(["no change", "4 to 8 years", "2 to 3 years", "1 year"], size=n_f),
            "NumberOfCars": rng.choice(["1 vehicle", "2 vehicles", "3 to 4"], size=n_f),
            "Year": rng.choice([1994, 1995, 1996], size=n_f),
            "BasePolicy": rng.choice(["Liability", "Collision", "All Perils"], size=n_f),
        })
        df_fraud.to_csv(RAW_FRAUD_PATH, index=False)


def parse_fraud_oracle(df_fraud: pd.DataFrame) -> pd.DataFrame:
    """Konwertuje przedziałowe zmienne tekstowe ze zbioru fraud_oracle.csv na wartości numeryczne."""
    df = df_fraud.copy()
    df["VehiclePrice_num"] = df["VehiclePrice"].map(VEHICLE_PRICE_MAP)
    df["AgeOfVehicle_num"] = df["AgeOfVehicle"].map(AGE_OF_VEHICLE_MAP)
    df["PastClaims_num"] = df["PastNumberOfClaims"].map(PAST_CLAIMS_MAP)
    df["DaysPolicyAccident_num"] = df["Days_Policy_Accident"].map(DAYS_POLICY_MAP)
    df["Suppliments_num"] = df["NumberOfSuppliments"].map(SUPPLIMENTS_MAP)
    df["BasePolicy_code"] = df["BasePolicy"].map({"Liability": 1, "Collision": 2, "All Perils": 3})
    df.to_csv(PROCESSED_FRAUD_PATH, index=False)
    return df


def run_numeric_eda() -> dict:
    """Przeprowadza pełną analizę eksploracyjną danych liczbowych na zbiorach insurance_claims.csv i fraud_oracle.csv."""
    generate_fallback_datasets()

    df_claims = pd.read_csv(RAW_CLAIMS_PATH)
    df_fraud_raw = pd.read_csv(RAW_FRAUD_PATH)

    # Usunięcie pustej kolumny technicznej _c39 ze zbioru insurance_claims.csv
    if "_c39" in df_claims.columns:
        df_claims = df_claims.drop(columns=["_c39"])

    # Obliczenie wieku pojazdu oraz kodowanie zmiennej fraud_reported
    max_year = int(df_claims["auto_year"].max()) if "auto_year" in df_claims.columns else 2015
    df_claims["vehicle_age"] = np.clip(max_year - df_claims["auto_year"], 0, None)
    df_claims["fraud_flag"] = df_claims["fraud_reported"].map({"Y": 1, "N": 0})

    # Wybór kluczowych kolumn numerycznych do analizy statystycznej (insurance_claims.csv)
    selected_claims_cols = [
        "total_claim_amount",
        "vehicle_claim",
        "injury_claim",
        "property_claim",
        "policy_annual_premium",
        "policy_deductable",
        "months_as_customer",
        "age",
        "vehicle_age",
        "number_of_vehicles_involved",
        "bodily_injuries",
        "witnesses",
        "fraud_flag",
    ]
    available_cols = [c for c in selected_claims_cols if c in df_claims.columns]

    # 1. Obliczenie statystyk opisowych oraz skośności rozkładu
    desc_claims = df_claims[available_cols].describe().T
    desc_claims["median"] = df_claims[available_cols].median()
    desc_claims["skewness"] = df_claims[available_cols].skew()
    desc_claims["kurtosis"] = df_claims[available_cols].kurtosis()

    # 2. Identyfikacja wartości odstających (Outliers) dla zmiennej celu total_claim_amount
    target_col = "total_claim_amount"
    q1 = df_claims[target_col].quantile(0.25)
    q3 = df_claims[target_col].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = max(0.0, q1 - 1.5 * iqr)
    outliers_mask = (df_claims[target_col] > upper_bound) | (df_claims[target_col] < lower_bound)
    outliers_count = int(outliers_mask.sum())
    outliers_pct = (outliers_count / len(df_claims)) * 100

    # 3. Wykres rozkładu kwoty roszczenia (total_claim_amount)
    plt.figure(figsize=(10, 5))
    sns.histplot(df_claims[target_col].dropna(), kde=True, bins=40, color="#1f77b4")
    plt.axvline(df_claims[target_col].mean(), color="red", linestyle="--", label=f"Średnia: {df_claims[target_col].mean():.2f}")
    plt.axvline(df_claims[target_col].median(), color="green", linestyle="-", label=f"Mediana: {df_claims[target_col].median():.2f}")
    plt.axvline(upper_bound, color="orange", linestyle=":", label=f"Górny próg IQR: {upper_bound:.2f}")
    plt.title(f"Rozkład kwoty roszczenia ({target_col}) — Skośność: {df_claims[target_col].skew():.2f}")
    plt.xlabel("Kwota roszczenia (total_claim_amount)")
    plt.ylabel("Liczba zgłoszeń")
    plt.legend()
    plt.tight_layout()
    dist_plot_path = os.path.join(FIGURES_DIR, "claim_amount_dist.png")
    plt.savefig(dist_plot_path, dpi=150)
    plt.close()

    # 4. Macierz korelacji dla zbioru insurance_claims.csv
    plt.figure(figsize=(11, 8))
    corr_claims = df_claims[available_cols].corr(method="pearson")
    sns.heatmap(corr_claims, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Macierz korelacji zmiennych numerycznych (insurance_claims.csv)")
    plt.tight_layout()
    corr_plot_path = os.path.join(FIGURES_DIR, "correlation_matrix.png")
    plt.savefig(corr_plot_path, dpi=150)
    plt.close()

    # 5. Parsowanie i analiza zbioru fraud_oracle.csv
    df_fraud = parse_fraud_oracle(df_fraud_raw)
    fraud_num_cols = [
        "Age",
        "Deductible",
        "DriverRating",
        "VehiclePrice_num",
        "AgeOfVehicle_num",
        "PastClaims_num",
        "DaysPolicyAccident_num",
        "Suppliments_num",
        "BasePolicy_code",
        "FraudFound_P",
    ]
    available_fraud_cols = [c for c in fraud_num_cols if c in df_fraud.columns]

    desc_fraud = df_fraud[available_fraud_cols].describe().T
    desc_fraud["median"] = df_fraud[available_fraud_cols].median()
    desc_fraud["skewness"] = df_fraud[available_fraud_cols].skew()

    plt.figure(figsize=(10, 7))
    corr_fraud = df_fraud[available_fraud_cols].corr(method="spearman")
    sns.heatmap(corr_fraud, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Macierz korelacji cech numerycznych i porządkowych (fraud_oracle.csv)")
    plt.tight_layout()
    fraud_corr_path = os.path.join(FIGURES_DIR, "fraud_oracle_corr.png")
    plt.savefig(fraud_corr_path, dpi=150)
    plt.close()

    summary = {
        "claims_rows": len(df_claims),
        "fraud_rows": len(df_fraud),
        "target_col": target_col,
        "target_mean": float(df_claims[target_col].mean()),
        "target_median": float(df_claims[target_col].median()),
        "target_skew": float(df_claims[target_col].skew()),
        "outliers_count": outliers_count,
        "outliers_pct": round(outliers_pct, 2),
        "claims_stats_table": desc_claims.reset_index().rename(columns={"index": "Zmienna"}).round(3),
        "fraud_stats_table": desc_fraud.reset_index().rename(columns={"index": "Zmienna"}).round(3),
        "dist_plot_path": dist_plot_path,
        "corr_plot_path": corr_plot_path,
        "fraud_corr_path": fraud_corr_path,
    }
    print(
        f"[EDA NUMERYCZNE] Zakończono analizę. "
        f"Wiersze insurance_claims: {len(df_claims)}, fraud_oracle: {len(df_fraud)}."
    )
    return summary


if __name__ == "__main__":
    run_numeric_eda()