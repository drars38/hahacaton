import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, precision_score, recall_score
import os
import pickle
from django.conf import settings

# Загружаем необходимые данные NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class TextNormalizer:
    """Класс для нормализации текста"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Добавляем русские стоп-слова если доступны
        try:
            russian_stopwords = set(stopwords.words('russian'))
            self.stop_words.update(russian_stopwords)
        except:
            pass
    
    def normalize(self, text):
        """Нормализует текст: токенизация, лемматизация, удаление стоп-слов"""
        if not text or not isinstance(text, str):
            return ""
        
        # Приводим к нижнему регистру
        text = text.lower()
        
        # Удаляем специальные символы, оставляем только буквы и пробелы
        text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ\s]', ' ', text)
        
        # Токенизация
        tokens = word_tokenize(text)
        
        # Лемматизация и удаление стоп-слов
        normalized_tokens = []
        for token in tokens:
            if len(token) > 2 and token not in self.stop_words:
                lemmatized = self.lemmatizer.lemmatize(token)
                normalized_tokens.append(lemmatized)
        
        return ' '.join(normalized_tokens)


class SentimentClassifier:
    """Простой классификатор тональности на основе правил и ML"""
    
    def __init__(self):
        self.normalizer = TextNormalizer()
        self.model = None
        self.vectorizer = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Инициализация простой модели на основе правил"""
        # Простой классификатор на основе ключевых слов
        self.positive_words = [
            'хороший', 'отлично', 'прекрасно', 'замечательно', 'люблю', 
            'нравится', 'рад', 'счастлив', 'отличный', 'великолепно',
            'good', 'great', 'excellent', 'wonderful', 'love', 'like', 'happy'
        ]
        self.negative_words = [
            'плохой', 'ужасно', 'ненавижу', 'не нравится', 'грустно', 
            'злой', 'плохо', 'отвратительно', 'ужасный',
            'bad', 'terrible', 'hate', 'awful', 'sad', 'angry', 'horrible'
        ]
    
    def predict(self, text):
        """Предсказывает тональность текста (0-нейтральная, 1-положительная, 2-отрицательная)"""
        if not text:
            return 0, 0.5  # Нейтральная по умолчанию
        
        normalized = self.normalizer.normalize(text)
        text_lower = text.lower()
        normalized_lower = normalized.lower()
        
        # Подсчет положительных и отрицательных слов
        positive_count = sum(1 for word in self.positive_words if word in text_lower or word in normalized_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower or word in normalized_lower)
        
        # Определение тональности
        # 0 = Нейтральная, 1 = Положительная, 2 = Отрицательная
        if positive_count > negative_count and positive_count > 0:
            sentiment = 1  # Положительная
            confidence = min(0.5 + (positive_count - negative_count) * 0.1, 0.95)
        elif negative_count > positive_count and negative_count > 0:
            sentiment = 2  # Отрицательная
            confidence = min(0.5 + (negative_count - positive_count) * 0.1, 0.95)
        else:
            sentiment = 0  # Нейтральная
            confidence = 0.6
        
        return sentiment, confidence
    
    def predict_batch(self, texts):
        """Предсказывает тональность для списка текстов"""
        results = []
        for text in texts:
            sentiment, confidence = self.predict(text)
            results.append((sentiment, confidence))
        return results


class DataProcessor:
    """Класс для обработки загруженных данных"""
    
    def __init__(self):
        self.normalizer = TextNormalizer()
        self.classifier = SentimentClassifier()
    
    def process_csv(self, file_path):
        """Обрабатывает CSV файл и возвращает DataFrame"""
        try:
            # Пробуем разные кодировки
            encodings = ['utf-8', 'cp1251', 'latin-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Не удалось прочитать файл с доступными кодировками")
            
            return df
        except Exception as e:
            raise ValueError(f"Ошибка при чтении CSV: {str(e)}")
    
    def extract_texts(self, df):
        """Извлекает тексты из DataFrame (оптимизированная версия)"""
        # Ищем колонку с текстом
        text_columns = ['text', 'текст', 'comment', 'комментарий', 'review', 'отзыв', 'content', 'содержание', 'comment_text']
        source_columns = ['source', 'источник', 'url', 'author', 'автор', 'id']
        
        text_col = None
        source_col = None
        
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in text_columns and text_col is None:
                text_col = col
            if col_lower in source_columns and source_col is None:
                source_col = col
        
        # Если не нашли, берем первую колонку как текст
        if text_col is None:
            text_col = df.columns[0]
        
        # Оптимизация: используем vectorized операции вместо iterrows
        texts = df[text_col].fillna('').astype(str).tolist()
        if source_col:
            sources = df[source_col].fillna('').astype(str).tolist()
        else:
            sources = [''] * len(texts)
        
        return texts, sources
    
    def process_texts(self, texts, sources=None, batch_size=1000):
        """Обрабатывает список текстов: нормализация и классификация (оптимизированная версия)"""
        if sources is None:
            sources = [""] * len(texts)
        
        results = []
        total = len(texts)
        
        # Обрабатываем батчами для лучшей производительности
        for i in range(0, total, batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_sources = sources[i:i+batch_size]
            
            # Используем batch prediction если доступно
            batch_results = self.classifier.predict_batch(batch_texts)
            
            for text, source, (sentiment, confidence) in zip(batch_texts, batch_sources, batch_results):
                normalized = self.normalizer.normalize(text)
                results.append({
                    'original_text': text,
                    'normalized_text': normalized,
                    'text': text,
                    'source': source,
                    'sentiment_score': sentiment,
                    'confidence': confidence
                })
        
        return results


class ValidationService:
    """Сервис для валидации результатов"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Вычисляет метрики: macro-F1, precision, recall"""
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        return {
            'macro_f1_score': float(macro_f1),
            'precision': float(precision),
            'recall': float(recall)
        }
    
    def validate_predictions(self, validation_df, predictions):
        """Валидирует предсказания модели"""
        # Ищем колонку с истинными метками
        label_columns = ['sentiment', 'label', 'target', 'tonality', 'тональность', 'оценка']
        
        true_label_col = None
        for col in validation_df.columns:
            if col.lower() in label_columns:
                true_label_col = col
                break
        
        if true_label_col is None:
            # Пробуем последнюю колонку
            true_label_col = validation_df.columns[-1]
        
        y_true = validation_df[true_label_col].astype(int).tolist()
        y_pred = [p['sentiment_score'] for p in predictions]
        
        # Убеждаемся, что длины совпадают
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        return self.calculate_metrics(y_true, y_pred)

