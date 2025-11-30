import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import io
import os
from datetime import datetime

# Конфигурация страницы
st.set_page_config(
    page_title="Анализ тональности",
    layout="wide",
)

# Заголовок приложения
st.markdown("### Автоматическая классификация текстов по эмоциональной окраске")

# Информационная строка под заголовком
st.markdown("""📋 Классификация: 😐 Нейтральные (0) | 😊 Позитивные (1) | 😞 Негативные (2)
""", unsafe_allow_html=True)

# Пример данных для справки
EXAMPLE_TEXTS = [
    "Это просто прекрасный продукт! Очень доволен покупкой.",
    "Ужасное качество, никогда больше не куплю.",
    "Нормальный товар, но есть недостатки.",
    "Отлично! Рекомендую всем знакомым.",
    "Разочарован. Не оправдало ожиданий.",
    "Обычный продукт, ничего особенного.",
    "Восхитительно! Лучшее что я покупал.",
    "Очень плохой сервис, не советую.",
    "Всё устроило, хорошее соотношение цены и качества.",
    "Ужасная доставка, товар пришёл поврежденным."
]


def create_example_data():
    """Создание примера данных для справки"""
    example_df = pd.DataFrame({
        'text': EXAMPLE_TEXTS,
        'sentiment': [1, 2, 0, 1, 2, 0, 1, 2, 1, 2],  # 0-neu, 1-pos, 2-neg
        'sentiment_label': ['positive', 'negative', 'neutral', 'positive',
                            'negative', 'neutral', 'positive', 'negative',
                            'neutral', 'negative']
    })
    return example_df


# Основной контент
st.markdown("---")

# Секция загрузки файлов
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📁 Загрузка данных")
    uploaded_file = st.file_uploader(
        "Загрузите CSV файл с текстами",
        type=['csv'],
        help="Файл должен содержать колонку 'text'"
    )

with col2:
    st.markdown("### 🎯 Валидация модели")
    validation_file = st.file_uploader(
        "Загрузите CSV для оценки (опционально)",
        type=['csv'],
        help="Файл должен содержать колонки 'text' и 'sentiment'"
    )

# Проверка статуса бэкенда
def check_backend_status():
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('model_ready', False), "✅ Бэкенд доступен"
        else:
            return False, "❌ Бэкенд недоступен"
    except:
        return False, "❌ Не удалось подключиться к бэкенду"

# Отображение статуса
model_ready, status_message = check_backend_status()
if not model_ready:
    st.warning(f"{status_message}. Убедитесь, что бэкенд запущен и модель обучена.")

# Обработка загруженного файла
if uploaded_file is not None:
    st.markdown("---")
    st.markdown("## 📋 Предпросмотр данных")

    try:
        df = pd.read_csv(uploaded_file)

        # Информация о данных
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Общее количество", len(df))
        with col2:
            st.metric("📝 Колонки", len(df.columns))
        with col3:
            avg_length = df['text'].str.len().mean() if 'text' in df.columns else 0
            st.metric("📏 Средняя длина", f"{avg_length:.0f} симв.")

        st.dataframe(df.head(10), use_container_width=True)

        # Кнопка для запуска анализа
        if st.button("🎯 Запустить анализ тональности", type="primary", use_container_width=True):
            if not model_ready:
                st.error("Модель не готова. Убедитесь, что бэкенд запущен и обучен.")
            else:
                with st.spinner("Анализируем тексты..."):
                    try:
                        # Отправка файла на бэкенд
                        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
                        response = requests.post(
                            "http://localhost:8000/predict",
                            files=files
                        )

                        if response.status_code == 200:
                            result = response.json()

                            # Отображение статистики
                            st.markdown("---")
                            st.markdown("## 📈 Результаты анализа")
                            col1, col2, col3 = st.columns(3)

                            stats = result['statistics']
                            with col1:
                                st.metric("😊 Позитивные", stats.get('positive', 0))
                            with col2:
                                st.metric("😐 Нейтральные", stats.get('neutral', 0))
                            with col3:
                                st.metric("😞 Негативные", stats.get('negative', 0))

                            # Визуализация
                            fig_col1, fig_col2 = st.columns(2)

                            with fig_col1:
                                fig_pie = px.pie(
                                    values=list(stats.values()),
                                    names=list(stats.keys()),
                                    title="Распределение тональности",
                                    color=list(stats.keys()),
                                    color_discrete_map={
                                        'positive': '#2E8B57',
                                        'neutral': '#FFD700',
                                        'negative': '#DC143C'
                                    }
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)

                            with fig_col2:
                                fig_bar = px.bar(
                                    x=list(stats.keys()),
                                    y=list(stats.values()),
                                    title="Количество текстов по тональности",
                                    color=list(stats.keys()),
                                    color_discrete_map={
                                        'positive': '#2E8B57',
                                        'neutral': '#FFD700',
                                        'negative': '#DC143C'
                                    }
                                )
                                st.plotly_chart(fig_bar, use_container_width=True)

                            # Кнопка скачивания
                            try:
                                download_response = requests.get(
                                    f"http://localhost:8000/download/{result['results_file']}")
                                if download_response.status_code == 200:
                                    st.download_button(
                                        label="📥 Скачать результаты",
                                        data=download_response.content,
                                        file_name=result['results_file'],
                                        mime="text/csv",
                                        use_container_width=True
                                    )

                                    # Сохраняем результаты для фильтрации
                                    st.session_state['last_result'] = result
                                    st.session_state['results_df'] = pd.read_csv(
                                        io.StringIO(download_response.content.decode('utf-8')))
                            except Exception as download_error:
                                st.error(f"Ошибка при скачивании: {download_error}")

                        else:
                            st.error(f"Ошибка сервера: {response.text}")

                    except Exception as e:
                        st.error(f"Ошибка при анализе: {e}")

    except Exception as e:
        st.error(f"Ошибка при чтении файла: {str(e)}")

# Валидация модели
if validation_file is not None and uploaded_file is not None and 'last_result' in st.session_state:
    st.markdown("---")
    st.markdown("## 📊 Оценка модели")

    if st.button("Вычислить метрики качества", use_container_width=True):
        with st.spinner("Вычисляем метрики..."):
            try:
                files = {
                    'predictions_file': (
                        'predictions.csv',
                        requests.get(
                            f"http://localhost:8000/download/{st.session_state['last_result']['results_file']}").content,
                        'text/csv'
                    ),
                    'ground_truth_file': (
                        validation_file.name,
                        validation_file.getvalue(),
                        'text/csv'
                    )
                }

                response = requests.post(
                    "http://localhost:8000/evaluate",
                    files=files
                )

                if response.status_code == 200:
                    metrics = response.json()

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Macro-F1 Score", f"{metrics['macro_f1']:.3f}")
                    with col2:
                        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")

                    # Детальный отчет
                    st.subheader("Детальный отчет")
                    report_df = pd.DataFrame(metrics['detailed_report']).transpose()
                    st.dataframe(report_df, use_container_width=True)
                else:
                    st.error(f"Ошибка при валидации: {response.text}")

            except Exception as e:
                st.error(f"Ошибка валидации: {e}")

# Поиск и фильтрация
if 'results_df' in st.session_state:
    st.markdown("---")
    st.markdown("## 🔍 Поиск и фильтрация")

    search_col1, search_col2 = st.columns([2, 1])

    with search_col1:
        search_term = st.text_input("Поиск по текстам")

    with search_col2:
        sentiment_filter = st.multiselect(
            "Фильтр по тональности",
            options=['negative', 'neutral', 'positive'],
            default=['negative', 'neutral', 'positive']
        )

    filtered_df = st.session_state['results_df'][
        (st.session_state['results_df']['text'].str.contains(search_term, case=False, na=False) if search_term else True) &
        (st.session_state['results_df']['sentiment_label'].isin(sentiment_filter))
        ]

    st.dataframe(filtered_df, use_container_width=True)

# Информационная панель
st.markdown("---")

col1, col2 = st.columns([2, 4])
with col1:
    with st.expander("📁 Формат файлов"):
        st.markdown("""

            **Формат обучающих данных:**
            ```csv
            ID,text,src,label
            1,Текст примера...,источник,0
            2,Другой текст...,источник,1
            ```
        
            **Основной файл для анализа:**
            ```csv
            text
            Ваш текст здесь...
            ```
        
            **Валидационный файл:**
            ```csv
            text,label
            Текст,0
            Текст,1  
            Текст,2
            ```
        
            **Метки тональности:**
            - 0 - Нейтральная
            - 1 - Позитивная  
            - 2 - Негативная
            """)

with col2:
    with st.expander("📋 Пример данных"):
        example_df = create_example_data()
        st.dataframe(example_df, use_container_width=True)
        st.caption("Пример структуры данных для анализа")

st.markdown("---")