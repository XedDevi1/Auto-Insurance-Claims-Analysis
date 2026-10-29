from src.eda_numeric import run_numeric_eda
from src.eda_nlp import run_nlp_eda


def main():
    """Uruchamia pełny potok analizy eksploracyjnej danych (EDA) dla Etapu 1."""
    print("=== Rozpoczęcie potoku analitycznego EDA (Etap 1) ===")
    numeric_results = run_numeric_eda()
    nlp_results = run_nlp_eda()
    print("=== Potok EDA zakończony pomyślnie. Wykresy zapisano w katalogu reports/figures/ ===")
    return numeric_results, nlp_results


if __name__ == "__main__":
    main()