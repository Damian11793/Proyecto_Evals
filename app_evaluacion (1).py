"""
App de Streamlit para evaluar el agente de TechModa con ragas.

Ejecutar con:
    streamlit run app_evaluacion.py

Requiere:
    pip install streamlit ragas datasets langchain-google-genai "langchain-community<0.4.2" pandas plotly
"""

import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Evaluación del agente TechModa", page_icon="🛍️", layout="wide")

# ============================================
# Sidebar — configuración
# ============================================
st.sidebar.title("Configuración")

api_key = st.sidebar.text_input("GOOGLE_API_KEY", type="password", help="AQ.Ab8RN6KoMR0ltqydiXl54XQNSeCVAoC9d2NdlOAing58Zqczmw")

modelo_llm = st.sidebar.selectbox(
    "Modelo LLM evaluador",
    ["gemini-2.5-flash", "gemini-2.5-pro"],
    index=0,
)

modelo_embeddings = st.sidebar.text_input(
    "Modelo de embeddings",
    value="models/gemini-embedding-001",
)

metricas_disponibles = {
    "Answer Relevancy": "answer_relevancy",
    "Faithfulness": "faithfulness",
    "Context Precision": "context_precision",
    "Context Recall": "context_recall",
}
metricas_seleccionadas = st.sidebar.multiselect(
    "Métricas a calcular",
    list(metricas_disponibles.keys()),
    default=["Answer Relevancy"],
    help="Faithfulness / Context Precision / Context Recall solo son fiables si "
         "'contexts' contiene los fragmentos reales recuperados del catálogo, no la respuesta ya generada.",
)

st.sidebar.markdown("---")
archivo = st.sidebar.file_uploader("Dataset (.jsonl)", type=["jsonl"])

# ============================================
# Header
# ============================================
st.title("🛍️ Evaluación del agente de TechModa")
st.caption("Métricas ragas sobre el dataset question / ground_truth / contexts")

# ============================================
# Cargar dataset
# ============================================
if archivo is None:
    st.info("Sube tu archivo `dataset.jsonl` en la barra lateral para comenzar.")
    st.stop()

filas = [json.loads(linea) for linea in archivo.read().decode("utf-8").splitlines() if linea.strip()]
st.success(f"{len(filas)} casos cargados")

# ============================================
# Sección: preguntas cargadas
# ============================================
st.subheader("📋 Preguntas del dataset")
df_preguntas = pd.DataFrame({
    "#": range(1, len(filas) + 1),
    "Pregunta": [f.get("question", "") for f in filas],
    "# contexts": [len(f.get("contexts", [])) for f in filas],
})
st.dataframe(df_preguntas, use_container_width=True, hide_index=True)

with st.expander("Ver dataset completo (question / ground_truth / contexts)"):
    st.dataframe(pd.DataFrame(filas), use_container_width=True)

# ============================================
# Modo de evaluación: ¿evaluar el prompt o solo el dataset ya generado?
# ============================================
st.markdown("---")
st.subheader("⚙️ Qué se va a evaluar")

modo = st.radio(
    "Modo de evaluación",
    [
        "Generar respuestas en vivo con mi SYSTEM_PROMPT (evalúa el prompt)",
        "Usar ground_truth como respuesta ya generada (evalúa solo el dataset)",
    ],
    index=0,
)

system_prompt = st.text_area(
    "SYSTEM_PROMPT del agente",
    value=(
        "Eres el asistente de compras de TechModa, una tienda de moda. Respondé en español neutral, "
        "amable y conciso. Recomendá ÚNICAMENTE productos del CATÁLOGO que se te entrega como "
        "contexto; si nada encaja, decirlo con honestidad y sugerir refinar la búsqueda. No "
        "inventes productos, precios ni características que no estén en el contexto. Se amable en todo momento con el cliente"
    ),
    height=140,
    disabled=modo.startswith("Usar ground_truth"),
    help="Este prompt se usa como system prompt real al generar cada respuesta. Solo aplica en el modo 'Generar respuestas en vivo'.",
)

# ============================================
# Ejecutar evaluación
# ============================================
ejecutar = st.sidebar.button("▶️ Ejecutar evaluación", type="primary", disabled=not api_key)

if not api_key:
    st.sidebar.warning("Ingresa tu API key para habilitar la evaluación.")

if ejecutar:
    os.environ["GOOGLE_API_KEY"] = api_key

    with st.spinner("Configurando LLM y embeddings evaluadores..."):
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        evaluator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=modelo_llm, temperature=0))
        evaluator_embeddings = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(model=modelo_embeddings))

    generar_en_vivo = modo.startswith("Generar respuestas en vivo")

    if generar_en_vivo:
        with st.spinner("Generando respuestas del agente con tu SYSTEM_PROMPT..."):
            modelo_agente = ChatGoogleGenerativeAI(model=modelo_llm, temperature=0.3)
            respuestas_generadas = []
            barra = st.progress(0.0, text="Generando respuestas...")
            for i, fila in enumerate(filas):
                contexto_str = "\n".join(fila.get("contexts", [])) or "No hay productos en el catálogo para esta búsqueda."
                prompt_completo = f"{system_prompt}\n\nCATÁLOGO:\n{contexto_str}\n\nCliente: {fila['question']}"
                respuesta = modelo_agente.invoke(prompt_completo).content
                respuestas_generadas.append(respuesta)
                barra.progress((i + 1) / len(filas), text=f"Generando respuestas... ({i + 1}/{len(filas)})")
            barra.empty()
    else:
        respuestas_generadas = [fila["ground_truth"] for fila in filas]

    with st.expander("Ver respuestas usadas en la evaluación"):
        st.dataframe(
            pd.DataFrame({"Pregunta": [f["question"] for f in filas], "Respuesta evaluada": respuestas_generadas}),
            use_container_width=True,
        )

    with st.spinner("Construyendo dataset de evaluación..."):
        from ragas import EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample

        samples = []
        for fila, respuesta in zip(filas, respuestas_generadas):
            samples.append(
                SingleTurnSample(
                    user_input=fila["question"],
                    retrieved_contexts=fila.get("contexts", []),
                    response=respuesta,
                    reference=fila["ground_truth"],
                )
            )
        dataset = EvaluationDataset(samples=samples)

    with st.spinner(f"Calculando métricas ({', '.join(metricas_seleccionadas)})... esto puede tardar un momento"):
        from ragas import evaluate
        from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall

        mapa_metricas = {
            "Answer Relevancy": ResponseRelevancy(),
            "Faithfulness": Faithfulness(),
            "Context Precision": LLMContextPrecisionWithReference(),
            "Context Recall": LLMContextRecall(),
        }
        metricas_obj = [mapa_metricas[m] for m in metricas_seleccionadas]

        try:
            resultado = evaluate(
                dataset=dataset,
                metrics=metricas_obj,
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
            )
            df = resultado.to_pandas()
        except Exception as e:
            st.error(f"Error al evaluar: {e}")
            st.stop()

    st.session_state["df_resultados"] = df

# ============================================
# Mostrar resultados (persisten entre reruns)
# ============================================
if "df_resultados" in st.session_state:
    df = st.session_state["df_resultados"]
    columnas_metricas = [metricas_disponibles[m] for m in metricas_seleccionadas if metricas_disponibles[m] in df.columns]

    st.markdown("---")
    st.subheader("Resumen")

    cols = st.columns(len(columnas_metricas) if columnas_metricas else 1)
    for i, col_metrica in enumerate(columnas_metricas):
        promedio = df[col_metrica].mean()
        cols[i].metric(col_metrica.replace("_", " ").title(), f"{promedio:.2f}" if pd.notna(promedio) else "N/A")

    st.markdown("---")
    st.subheader("Resultados por caso")

    import plotly.express as px

    for col_metrica in columnas_metricas:
        fig = px.bar(
            df,
            x="user_input",
            y=col_metrica,
            title=f"{col_metrica.replace('_', ' ').title()} por pregunta",
            range_y=[0, 1],
        )
        fig.update_layout(xaxis_title="Pregunta", yaxis_title="Score", xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Tabla detallada")
    columnas_mostrar = ["user_input"] + columnas_metricas + (["response"] if "response" in df.columns else [])
    st.dataframe(df[columnas_mostrar], use_container_width=True)

    with st.expander("Ver casos con score bajo (< 0.6)"):
        for col_metrica in columnas_metricas:
            bajos = df[df[col_metrica] < 0.6]
            if not bajos.empty:
                st.markdown(f"**{col_metrica}**")
                for _, fila in bajos.iterrows():
                    st.write(f"- ({fila[col_metrica]:.2f}) {fila['user_input']}")

    st.markdown("---")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar resultados (CSV)",
        data=csv,
        file_name="resultados_ragas.csv",
        mime="text/csv",
    )
else:
    st.info("Configura la API key y presiona **▶️ Ejecutar evaluación** en la barra lateral.")
