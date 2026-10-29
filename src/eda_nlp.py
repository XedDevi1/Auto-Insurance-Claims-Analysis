import os
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from src.eda_numeric import generate_fallback_datasets, RAW_CLAIMS_PATH, RAW_FRAUD_PATH

PROCESSED_CLAIMS_PATH = os.path.join("data", "processed", "insurance_claims_clean.csv")
TFIDF_OUTPUT_PATH = os.path.join("data", "processed", "tfidf_top_terms.csv")
FIGURES_DIR = os.path.join("reports", "figures")

os.makedirs(os.path.dirname(PROCESSED_CLAIMS_PATH), exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Zbiór słów funkcyjnych (stop-words) do odfiltrowania podczas preprocessingu NLP
STOP_WORDS = {
    "i", "w", "na", "z", "do", "o", "że", "się", "nie", "to", "jest", "jak", "po",
    "ale", "za", "od", "przez", "przy", "oraz", "lub", "tym", "tego", "był", "była",
    "było", "są", "dla", "tak", "tylko", "bardzo", "kiedy", "gdy", "ponieważ", "podczas",
    "the", "and", "to", "of", "in", "was", "for", "on", "that", "with", "at", "by",
    "from", "as", "is", "were", "it", "an", "be", "this", "have", "had", "recorded",
    "reported", "involved", "insured", "policyholder", "claim", "incident", "vehicle"
}


def build_claim_narrative(row: pd.Series) -> str:
    """Buduje spójny opis tekstowy zdarzenia drogowego na podstawie atrybutów zgłoszenia w insurance_claims.csv."""
    # Jeżeli w kluczowych polach opisowych występują braki danych ('?'), zwracamy NaN dla części zgłoszeń
    if pd.isna(row.get("collision_type")) and pd.isna(row.get("property_damage")):
        return np.nan

    collision = row.get("collision_type") if pd.notna(row.get("collision_type")) else "unknown collision configuration"
    prop_dmg = row.get("property_damage") if pd.notna(row.get("property_damage")) else "unverified property damage"
    police = row.get("police_report_available") if pd.notna(row.get("police_report_available")) else "unconfirmed"

    narrative = (
        f"Zgłoszenie szkody komunikacyjnej typu {row.get('incident_type', 'collision')} "
        f"({collision}) o stopniu uszkodzeń {row.get('incident_severity', 'minor')} "
        f"w lokalizacji {row.get('incident_location', 'roadway')}, miasto {row.get('incident_city', 'unknown')}, "
        f"stan {row.get('incident_state', 'unknown')}. "
        f"Zdarzenie miało miejsce o godzinie {row.get('incident_hour_of_the_day', 12)} "
        f"z udziałem pojazdu marki {row.get('auto_make', 'auto')} {row.get('auto_model', '')} "
        f"(rok produkcji {row.get('auto_year', 2010)}). "
        f"Powiadomione służby ratunkowe: {row.get('authorities_contacted', 'none')}, "
        f"uszkodzenie mienia: {prop_dmg}, dostępność raportu policji: {police}, "
        f"liczba świadków na miejscu zdarzenia: {row.get('witnesses', 0)}, "
        f"obrażenia ciała: {row.get('bodily_injuries', 0)}. "
        f"Profil ubezpieczonego: zawód {row.get('insured_occupation', 'unknown')}, "
        f"wykształcenie {row.get('insured_education_level', 'unknown')}, "
        f"zainteresowania {row.get('insured_hobbies', 'none')}, "
        f"relacja rodzinna {row.get('insured_relationship', 'unknown')}."
    )
    return narrative


def build_fraud_oracle_narrative(row: pd.Series) -> str:
    """Buduje opis tekstowy z atrybutów tekstowych zbioru fraud_oracle.csv."""
    return (
        f"Kolizja drogowa w obszarze {row.get('AccidentArea', 'Urban')} "
        f"w miesiącu {row.get('Month', 'Jan')} ({row.get('DayOfWeek', 'Monday')}). "
        f"Kierowca pojazdu marki {row.get('Make', 'Vehicle')} kategorii {row.get('VehicleCategory', 'Sedan')} "
        f"o wartości {row.get('VehiclePrice', 'standard')} i wieku {row.get('AgeOfVehicle', 'unknown')}. "
        f"Orzeczona wina: {row.get('Fault', 'Policy Holder')}, typ polisy {row.get('PolicyType', 'Standard')} "
        f"(zakres bazowy {row.get('BasePolicy', 'Collision')}). "
        f"Raport policyjny: {row.get('PoliceReportFiled', 'No')}, obecność świadków: {row.get('WitnessPresent', 'No')}, "
        f"typ agenta: {row.get('AgentType', 'External')}, zmiana adresu: {row.get('AddressChange_Claim', 'no change')}."
    )


def preprocess_text(text: str) -> str:
    """Przeprowadza tokenizację, konwersję do małych liter oraz usunięcie interpunkcji, cyfr i stop-words."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}0-9]", " ", text)
    tokens = text.split()
    filtered_tokens = [tok for tok in tokens if tok not in STOP_WORDS and len(tok) > 2]
    return " ".join(filtered_tokens)


def run_nlp_eda() -> dict:
    """Wykonuje analizę danych tekstowych (NLP), identyfikację braków danych, imputację oraz ekstrakcję TF-IDF."""
    generate_fallback_datasets()

    df_claims = pd.read_csv(RAW_CLAIMS_PATH)
    df_fraud = pd.read_csv(RAW_FRAUD_PATH)

    if "_c39" in df_claims.columns:
        df_claims = df_claims.drop(columns=["_c39"])

    # Zamiana znaków zapytania '?' oznaczających brakujące dane w insurance_claims.csv na wartości NaN
    df_claims = df_claims.replace("?", np.nan)

    # Obliczenie odsetka brakujących danych w polach tekstowych przed imputacją
    text_attributes = ["collision_type", "property_damage", "police_report_available", "authorities_contacted"]
    existing_text_attrs = [c for c in text_attributes if c in df_claims.columns]
    total_text_cells = len(df_claims) * len(existing_text_attrs)
    missing_text_cells = int(df_claims[existing_text_attrs].isna().sum().sum())

    # Generowanie narracji tekstowych (lub użycie istniejącej kolumny tekstowej)
    if "narrative" not in df_claims.columns:
        df_claims["narrative"] = df_claims.apply(build_claim_narrative, axis=1)

    fraud_narratives = df_fraud.head(1000).apply(build_fraud_oracle_narrative, axis=1)

    # Analiza braków danych (NaN) w wygenerowanych opisach zdarzeń
    nan_narratives_count = int(df_claims["narrative"].isna().sum())
    nan_narratives_pct = (nan_narratives_count / len(df_claims)) * 100

    # Obliczenie długości opisów tekstowych (w słowach) przed imputacją
    valid_narratives = pd.concat([df_claims["narrative"].dropna(), fraud_narratives], ignore_index=True)
    word_counts = valid_narratives.astype(str).apply(lambda s: len(s.split()))
    mean_words = float(word_counts.mean())
    std_words = float(word_counts.std())

    # Wykres rozkładu długości opisów zdarzeń
    plt.figure(figsize=(9, 5))
    sns.histplot(word_counts, bins=25, kde=True, color="#2ca02c")
    plt.axvline(mean_words, color="red", linestyle="--", label=f"Średnia długość: {mean_words:.1f} słów")
    plt.title("Rozkład długości opisów tekstowych zgłoszeń (Analiza NLP)")
    plt.xlabel("Liczba słów w opisie zdarzenia")
    plt.ylabel("Liczba zgłoszeń")
    plt.legend()
    plt.tight_layout()
    wc_plot_path = os.path.join(FIGURES_DIR, "word_count_dist.png")
    plt.savefig(wc_plot_path, dpi=150)
    plt.close()

    # Imputacja brakujących wartości tekstowych oraz czyszczenie tekstu
    for col in existing_text_attrs:
        df_claims[col] = df_claims[col].fillna("brak_danych")
    df_claims["narrative"] = df_claims["narrative"].fillna("brak szczegółowego opisu zdarzenia drogowego")
    df_claims["clean_narrative"] = df_claims["narrative"].apply(preprocess_text)
    df_claims.to_csv(PROCESSED_CLAIMS_PATH, index=False)

    # Połączony korpus tekstowy z obu zbiorów do analizy częstotliwości słów TF-IDF
    combined_clean_corpus = pd.concat(
        [df_claims["clean_narrative"], fraud_narratives.apply(preprocess_text)],
        ignore_index=True,
    )

    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(combined_clean_corpus)
    mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    vocab = vectorizer.get_feature_names_out()

    tfidf_df = (
        pd.DataFrame({"Termin": vocab, "Średnia waga TF-IDF": np.round(mean_tfidf, 4)})
        .sort_values(by="Średnia waga TF-IDF", ascending=False)
        .head(15)
    )
    tfidf_df.to_csv(TFIDF_OUTPUT_PATH, index=False)

    # Wykres najważniejszych terminów według wagi TF-IDF
    plt.figure(figsize=(10, 6))
    sns.barplot(data=tfidf_df, x="Średnia waga TF-IDF", y="Termin", hue="Termin", palette="viridis", legend=False)
    plt.title("Najważniejsze terminy i frazy w opisach szkód (TF-IDF)")
    plt.xlabel("Średnia waga TF-IDF")
    plt.ylabel("Termin / Fraza")
    plt.tight_layout()
    tfidf_plot_path = os.path.join(FIGURES_DIR, "tfidf_top_words.png")
    plt.savefig(tfidf_plot_path, dpi=150)
    plt.close()

    summary = {
        "total_narratives": len(combined_clean_corpus),
        "missing_text_cells": missing_text_cells,
        "missing_cells_pct": round((missing_text_cells / max(1, total_text_cells)) * 100, 2),
        "nan_narratives_count": nan_narratives_count,
        "nan_narratives_pct": round(nan_narratives_pct, 2),
        "mean_words": round(mean_words, 2),
        "std_words": round(std_words, 2),
        "tfidf_table": tfidf_df,
        "wc_plot_path": wc_plot_path,
        "tfidf_plot_path": tfidf_plot_path,
    }
    print(
        f"[EDA NLP] Zakończono przetwarzanie tekstu. "
        f"Średnia długość opisu: {mean_words:.1f} słów, Braki NaN w narracjach: {nan_narratives_pct:.2f}%."
    )
    return summary


if __name__ == "__main__":
    run_nlp_eda()