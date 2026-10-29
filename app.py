import gradio as gr
from src.eda_numeric import run_numeric_eda
from src.eda_nlp import run_nlp_eda


def load_dashboard_data():
    num_summary = run_numeric_eda()
    nlp_summary = run_nlp_eda()

    num_md = (
        f"### Podsumowanie Zbioru 2 (Dane Liczbowe)\n"
        f"- **Liczba obserwacji:** `{num_summary['claims_rows']}`\n"
        f"- **Zmienna celu:** `{num_summary['target_col']}`\n"
        f"- **Średnia kwota roszczenia:** `{num_summary['target_mean']:.2f}` | **Mediana:** `{num_summary['target_median']:.2f}`\n"
        f"- **Skośność rozkładu (Skewness):** `{num_summary['target_skew']:.2f}` (rozkład silnie prawostronnie skośny)\n"
        f"- **Wartości odstające (Outliers > Q3 + 1.5*IQR):** `{num_summary['outliers_count']}` (`{num_summary['outliers_pct']}%`)"
    )

    nlp_md = (
        f"### Podsumowanie Zbioru 1 (Dane Tekstowe - NLP)\n"
        f"- **Liczba zgłoszeń:** `{nlp_summary['total_narratives']}`\n"
        f"- **Średnia długość zgłoszenia:** `{nlp_summary['mean_words']}` słów (odchylenie std: `{nlp_summary['std_words']}`)\n"
        f"- **Braki danych (NaN):** `{nlp_summary['nan_narratives_count']}` (`{nlp_summary['nan_narratives_pct']}%` zbioru — poddane imputacji)"
    )

    return (
        num_md,
        num_summary["claims_stats_table"],
        num_summary["dist_plot_path"],
        num_summary["corr_plot_path"],
        nlp_md,
        nlp_summary["tfidf_table"],
        nlp_summary["wc_plot_path"],
        nlp_summary["tfidf_plot_path"],
    )


with gr.Blocks(title="Auto Insurance Claims Analysis - Etap 1") as demo:
    gr.Markdown(
        "# Auto Insurance Claims Analysis — Panel Analityczny (Etap 1)\n"
        "Interfejs prezentujący wyniki analizy eksploracyjnej danych (EDA) dla zbioru tekstowego (NLP) oraz liczbowego."
    )

    refresh_btn = gr.Button("Odśwież / Przelicz statystyki EDA", variant="primary")

    with gr.Tabs():
        with gr.TabItem("1. Analiza Danych Liczbowych (Zbiór 2)"):
            num_info = gr.Markdown()
            num_table = gr.Dataframe(label="Statystyki opisowe zmiennych numerycznych")
            with gr.Row():
                num_dist_plot = gr.Image(label="Rozkład zmiennej celu (Claim Amount)", type="filepath")
                num_corr_plot = gr.Image(label="Macierz korelacji", type="filepath")

        with gr.TabItem("2. Analiza Danych Tekstowych - NLP (Zbiór 1)"):
            nlp_info = gr.Markdown()
            with gr.Row():
                nlp_wc_plot = gr.Image(label="Rozkład długości zgłoszeń", type="filepath")
                nlp_tfidf_plot = gr.Image(label="Najczęstsze terminy (TF-IDF)", type="filepath")
            nlp_table = gr.Dataframe(label="Top 15 terminów według wagi TF-IDF")

        with gr.TabItem("3. Pytania Badawcze i Zespół"):
            gr.Markdown(
                "### Pytania Badawcze\n"
                "1. W jaki sposób występowanie konkretnych słów kluczowych i poziom złożoności opisu wypadku (dane tekstowe) "
                "koreluje z ostateczną wartością przyznanego odszkodowania (dane liczbowe)?\n"
                "2. Czy na podstawie sentymentu i struktury narracji zgłoszenia (NLP) można z wyprzedzeniem przewidzieć "
                "anomalie liczbowe i potencjalne próby wyłudzenia ubezpieczenia?\n"
                "3. Które cechy numeryczne (np. wiek kierowcy, staż polisy) determinują charakter opisywanych szkód "
                "(klastrowanie nienadzorowane tekstów dla konkretnych grup ryzyka)?\n\n"
                "### Zespół Projektowy\n"
                "- **Pavel Andreyenka (71885)** — Lead Programmer (Dane numeryczne, EDA, modele ML)\n"
                "- **Vadzim Presman (71913)** — Lead Programmer (NLP, integracja, GitHub)\n"
                "- **Mykyta Dushyn (72347)** — Data Analyst (Analiza statystyczna, pytania badawcze)\n"
                "- **Dariia Lapina (72850)** — Data Analyst (Weryfikacja źródeł, wizualizacje, dokumentacja)\n"
                "- **Oleksandr Holota (73939)** — Project Manager / UI (Koordynacja, raporty, interfejs Gradio)"
            )

    demo.load(
        fn=load_dashboard_data,
        inputs=[],
        outputs=[
            num_info, num_table, num_dist_plot, num_corr_plot,
            nlp_info, nlp_table, nlp_wc_plot, nlp_tfidf_plot
        ],
    )
    refresh_btn.click(
        fn=load_dashboard_data,
        inputs=[],
        outputs=[
            num_info, num_table, num_dist_plot, num_corr_plot,
            nlp_info, nlp_table, nlp_wc_plot, nlp_tfidf_plot
        ],
    )

if __name__ == "__main__":
    demo.launch()