# Uczenie maszynowe - Projekt 1: Auto Insurance Claims Analysis

## Opis Projektu
Projekt skupia się na analizie danych z ubezpieczeń komunikacyjnych (auto insurance). Celem jest połączenie analizy danych tekstowych (opisy zdarzeń drogowych/roszczeń) z danymi liczbowymi (wiek kierowcy, kwota roszczenia, wiek pojazdu), aby znaleźć zależności pomiędzy charakterem wypadku a ostatecznymi kosztami ubezpieczeniowymi. Projekt wykorzystuje metody statystyczne, NLP oraz klasyczne modele uczenia maszynowego (nadzorowanego i nienadzorowanego).

## Zespół (Grupa)
*   **Pavel Andreyenka (7188517):** Lead Programmer - data parsing, EDA, skrypty ML.
*   **Vadzim Presman (71913):** Lead Programmer - NLP preprocessing, konfiguracja repozytorium.
*   **Mykyta Dushyn (72347):** Data Analyst - statystyka, pytania badawcze.
*   **Dariia Lapina (72850):** Data Analyst - wizualizacja, dokumentacja.
*   **Oleksandr Holota (73939):** Project Manager - raportowanie, projekt interfejsu (Gradio).

## Struktura Repozytorium
*   `/data/` - folder zawierający pobrane zbiory danych (lub skrypty do ich pobrania).
*   `/notebooks/` - notatniki Jupyter z eksploracyjną analizą danych (EDA).
*   `/src/` - kod źródłowy modułów przetwarzania danych i modeli.
*   `/reports/` - wygenerowane raporty w formacie PDF (w tym Raport - Etap 1).
*   `app.py` - skrypt uruchamiający interfejs użytkownika (Gradio).
*   `requirements.txt` - lista wymaganych bibliotek Python.

## Zbiory Danych
1.  **Dane tekstowe (Zbiór 1):** *Automobile Insurance Claims Narratives*. Zbiór zawiera opisy zdarzeń w języku naturalnym z formularzy zgłoszeniowych.
2.  **Dane liczbowe (Zbiór 2):** *Auto Insurance Claims Data*. Zbiór numeryczny zawierający atrybuty polisy, kierowcy oraz pojazdu.
*Uwaga: W projekcie wykorzystywane są pełne zbiory danych, bez wyodrębniania podzbiorów.*

## Instrukcja Uruchomienia (Odtworzenie wyników)

1. **Sklonuj repozytorium:**
   ```bash
   git clone [TWOJ_LINK_DO_GITHUB]
   cd nazwa-repozytorium
   ```

2. **Zainstaluj wymagane biblioteki:**
   Zalecane jest użycie wirtualnego środowiska (np. venv lub conda).
   ```bash
   pip install -r requirements.txt
   ```

3. **Uruchom analizę EDA:**
   Przejdź do folderu `/notebooks/` i uruchom pliki Jupyter Notebook, aby prześledzić statystyki i wykresy.

4. **Uruchom interfejs użytkownika (Gradio):**
   Aby skorzystać z interfejsu pozwalającego na analizę i testowanie wyników naszych modeli, uruchom:
   ```bash
   python app.py
   ```
   Aplikacja uruchomi się lokalnie i będzie dostępna w przeglądarce pod adresem `http://127.0.0.1:7860/`.
