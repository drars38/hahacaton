import pandas as pd
import torch
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import f1_score
import joblib


class RuBertSentimentClassifier:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.classifier = None
        self.device = 0 if torch.cuda.is_available() else -1
        self.is_trained = False

    def clean_text(self, text):
        """Очистка текста"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\S+@\S+', '', text)
        return text.strip()

    def load_pretrained(self, model_path):
        """Загрузка предобученной модели"""
        try:
            print(f"🔍 Загружаем модель из {model_path}")

            # Проверяем наличие необходимых файлов
            required_files = ['config.json', 'model.safetensors']
            for file in required_files:
                if not os.path.exists(os.path.join(model_path, file)):
                    print(f"❌ Отсутствует файл: {file}")
                    return False

            # Загружаем токенизатор
            if os.path.exists(os.path.join(model_path, 'tokenizer_config.json')):
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                print("✅ Токенизатор загружен")
            else:
                # Если нет токенизатора, используем базовый
                self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
                print("⚠️  Используем базовый токенизатор")

            # Загружаем модель
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            print("✅ Модель загружена")

            # Создаем pipeline для предсказаний
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                truncation=True,
                max_length=128,
                batch_size=32
            )

            self.is_trained = True
            print("✅ RuBERT модель успешно загружена")
            return True

        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False

    def train(self, texts, labels, model_name="DeepPavlov/rubert-base-cased-sentence"):
        """Fine-tuning модели RuBERT (если понадобится дообучение)"""
        try:
            print("🔄 Запуск fine-tuning RuBERT...")
            # ... ваш существующий код обучения ...
            # Этот метод можно оставить для возможности дообучения
            pass
        except Exception as e:
            print(f"❌ Ошибка обучения RuBERT: {e}")
            raise

    def predict(self, texts):
        """Предсказание тональности"""
        if not self.is_trained or self.classifier is None:
            raise ValueError("Модель не загружена. Сначала выполните загрузку модели.")

        # Очищаем тексты
        cleaned_texts = [self.clean_text(text) for text in texts]

        try:
            # Предсказание
            results = self.classifier(cleaned_texts)

            # Преобразуем в числовые метки (LABEL_0 -> 0, LABEL_1 -> 1, LABEL_2 -> 2)
            predictions = [int(r["label"].replace("LABEL_", "")) for r in results]

            return predictions

        except Exception as e:
            print(f"❌ Ошибка при предсказании: {e}")
            # Возвращаем нейтральные предсказания в случае ошибки
            return [1] * len(texts)

    def save(self, path):
        """Сохранение модели (для будущего дообучения)"""
        if self.is_trained and self.model is not None:
            model_path = os.path.join(path, "rubert_model")
            os.makedirs(model_path, exist_ok=True)
            self.model.save_pretrained(model_path)
            if self.tokenizer:
                self.tokenizer.save_pretrained(model_path)
            print(f"✅ Модель сохранена в {model_path}")

    def load(self, path):
        """Загрузка модели"""
        model_path = os.path.join(path, "rubert_model")
        if os.path.exists(model_path):
            return self.load_pretrained(model_path)
        else:
            print(f"❌ Модель не найдена по пути: {model_path}")
            return False