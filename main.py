from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
from model import RuBertSentimentClassifier
import io
import json
import os
from datetime import datetime
import uuid

app = FastAPI(title="Sentiment Analysis API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Создание директорий
os.makedirs("data", exist_ok=True)
os.makedirs("shared_data", exist_ok=True)

# Инициализация RuBERT модели
classifier = RuBertSentimentClassifier()
MODEL_PATH = "data"
TRAINING_DATA_PATH = "data/training_data.csv"


def ensure_model_exists():
    """Убедиться, что модель существует, иначе обучить на CSV данных"""
    try:
        classifier.load(MODEL_PATH)
        print("✅ RuBERT модель загружена успешно")
        return True
    except Exception as e:
        print(f"⚠️ Модель не найдена, обучаем на CSV данных...")
        from train_model import train_and_save_model
        success = train_and_save_model(TRAINING_DATA_PATH)
        if not success:
            print("❌ Не удалось обучить модель. Убедитесь, что файл data/training_data.csv существует.")
            return False

        # Пытаемся загрузить заново после обучения
        try:
            classifier.load(MODEL_PATH)
            print("✅ RuBERT модель загружена успешно после обучения")
            return True
        except Exception as load_error:
            print(f"❌ Ошибка при загрузке модели после обучения: {load_error}")
            return False


# Гарантируем, что модель будет готова при запуске
model_ready = ensure_model_exists()


@app.get("/")
async def root():
    status = "ready" if model_ready else "training_required"
    return {
        "message": "Sentiment Analysis API with RuBERT",
        "status": status,
        "model_ready": model_ready,
        "model_type": "RuBERT"
    }


@app.get("/health")
async def health_check():
    status = "healthy" if model_ready else "needs_training"
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "model_ready": model_ready,
        "model_type": "RuBERT"
    }


@app.post("/predict")
async def predict_sentiment(file: UploadFile = File(...)):
    if not model_ready:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready. Please ensure training data exists and model is trained."
        )

    try:
        # Генерация уникального имени файла
        file_id = str(uuid.uuid4())
        output_filename = f"results_{file_id}.csv"
        output_path = f"shared_data/{output_filename}"

        # Чтение CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Проверка наличия колонки 'text'
        if 'text' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV файл должен содержать колонку 'text'")

        # Предсказание с помощью RuBERT
        texts = df['text'].tolist()
        print(f"🔮 Анализируем {len(texts)} текстов с помощью RuBERT...")
        predictions = classifier.predict(texts)

        # Создаем новый DataFrame только с двумя колонками
        result_df = pd.DataFrame({
            'text': texts,
            'label': predictions
        })

        # Сохранение результата
        result_df.to_csv(output_path, index=False)

        # Статистика для отчета
        stats = result_df['label'].value_counts().to_dict()
        stats_named = {
            'negative': stats.get(0, 0),
            'neutral': stats.get(1, 0),
            'positive': stats.get(2, 0)
        }

        return {
            "message": "Prediction completed with RuBERT",
            "statistics": stats_named,
            "results_file": output_filename,
            "file_id": file_id,
            "total_records": len(result_df),
            "model_type": "RuBERT"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain")
async def retrain_model(file: UploadFile = File(...)):
    """Переобучить модель на новых данных"""
    try:
        # Чтение CSV с новыми данными
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Проверяем необходимые колонки - теперь используем 'label' вместо 'sentiment'
        required_columns = ['text', 'label']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"CSV файл должен содержать колонки {required_columns}. Отсутствуют: {missing_columns}"
            )

        # Обучаем модель
        texts = df['text'].tolist()
        labels = df['label'].astype(int).tolist()

        classifier.train(texts, labels)
        classifier.save(MODEL_PATH)

        global model_ready
        model_ready = True

        return {
            "message": "Model retrained successfully",
            "records_used": len(df),
            "label_distribution": df['label'].value_counts().to_dict()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/status")
async def model_status():
    """Получить статус модели"""
    training_data_exists = os.path.exists(TRAINING_DATA_PATH)
    model_exists = os.path.exists(MODEL_PATH)

    # Проверим структуру training data если файл существует
    training_data_info = {}
    if training_data_exists:
        try:
            df = pd.read_csv(TRAINING_DATA_PATH)
            training_data_info = {
                "columns": list(df.columns),
                "records_count": len(df),
                "has_text_column": 'text' in df.columns,
                "has_label_column": 'label' in df.columns
            }
        except Exception as e:
            training_data_info = {"error": str(e)}

    return {
        "model_ready": model_ready,
        "training_data_exists": training_data_exists,
        "model_file_exists": model_exists,
        "training_data_path": TRAINING_DATA_PATH,
        "model_path": MODEL_PATH,
        "training_data_info": training_data_info
    }


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"shared_data/{filename}"
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type='text/csv',
            filename=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    else:
        raise HTTPException(status_code=404, detail="File not found")


@app.post("/evaluate")
async def evaluate_model(
        predictions_file: UploadFile = File(...),
        ground_truth_file: UploadFile = File(...)
):
    """Оценка модели по метрикам"""
    try:
        print(f"🔍 Начало evaluate: {predictions_file.filename}, {ground_truth_file.filename}")

        # Читаем содержимое файлов
        predictions_content = await predictions_file.read()
        ground_truth_content = await ground_truth_file.read()

        predictions_text = predictions_content.decode('utf-8')
        ground_truth_text = ground_truth_content.decode('utf-8')

        # Читаем CSV
        pred_df = pd.read_csv(io.StringIO(predictions_text))
        true_df = pd.read_csv(io.StringIO(ground_truth_text))

        print(f"📊 Колонки predictions: {list(pred_df.columns)}")
        print(f"📊 Колонки ground_truth: {list(true_df.columns)}")
        print(f"📏 Размер predictions: {len(pred_df)}")
        print(f"📏 Размер ground_truth: {len(true_df)}")

        # ГИБКАЯ ПРОВЕРКА КОЛОНОК

        # Для predictions ищем колонку с sentiment
        sentiment_col = None
        for col in ['sentiment', 'label', 'sentiment_label', 'target']:
            if col in pred_df.columns:
                sentiment_col = col
                break

        if sentiment_col is None:
            available_cols = list(pred_df.columns)
            raise HTTPException(
                status_code=400,
                detail=f"Predictions file must contain sentiment column. Available columns: {available_cols}"
            )

        # Для ground truth ищем колонку с истинными метками
        truth_col = None
        for col in ['label', 'sentiment', 'target', 'true_label']:
            if col in true_df.columns:
                truth_col = col
                break

        if truth_col is None:
            available_cols = list(true_df.columns)
            raise HTTPException(
                status_code=400,
                detail=f"Ground truth file must contain label column. Available columns: {available_cols}"
            )

        print(f"🎯 Используем колонку predictions: '{sentiment_col}'")
        print(f"🎯 Используем колонку ground_truth: '{truth_col}'")

        # Вычисление метрик
        from sklearn.metrics import f1_score, classification_report, accuracy_score

        y_true = true_df[truth_col]
        y_pred = pred_df[sentiment_col]

        # Проверяем совместимость данных
        if len(y_true) != len(y_pred):
            print(f"⚠️ Разная длина: y_true={len(y_true)}, y_pred={len(y_pred)}")
            # Берем минимум для совместимости
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]

        print(f"📊 Уникальные значения y_true: {sorted(y_true.unique())}")
        print(f"📊 Уникальные значения y_pred: {sorted(y_pred.unique())}")

        macro_f1 = f1_score(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)

        return {
            "macro_f1": macro_f1,
            "accuracy": accuracy,
            "detailed_report": report,
            "columns_used": {
                "predictions": sentiment_col,
                "ground_truth": truth_col
            },
            "data_info": {
                "samples_used": len(y_true),
                "true_labels_distribution": y_true.value_counts().to_dict(),
                "pred_labels_distribution": y_pred.value_counts().to_dict()
            }
        }

    except Exception as e:
        print(f"❌ Ошибка в evaluate: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))