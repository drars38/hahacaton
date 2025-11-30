from django.db import models
from django.utils import timezone


class TextDocument(models.Model):
    """Модель для хранения загруженных документов"""
    name = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(default=timezone.now)
    file_path = models.CharField(max_length=500, blank=True)
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return self.name


class TextEntry(models.Model):
    """Модель для хранения отдельных текстовых записей"""
    SENTIMENT_CHOICES = [
        (0, 'Нейтральная'),
        (1, 'Положительная'),
        (2, 'Отрицательная'),
    ]
    
    document = models.ForeignKey(TextDocument, on_delete=models.CASCADE, related_name='entries')
    text = models.TextField()
    source = models.CharField(max_length=255, blank=True, null=True)
    original_text = models.TextField(blank=True)  # Оригинальный текст до нормализации
    normalized_text = models.TextField(blank=True)  # Нормализованный текст
    sentiment_score = models.IntegerField(choices=SENTIMENT_CHOICES, null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    is_manually_corrected = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['document', 'sentiment_score']),
            models.Index(fields=['source']),
        ]
    
    def __str__(self):
        return f"{self.text[:50]}... ({self.get_sentiment_score_display()})"


class ValidationResult(models.Model):
    """Модель для хранения результатов валидации"""
    document = models.ForeignKey(TextDocument, on_delete=models.CASCADE, related_name='validation_results')
    macro_f1_score = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    created_at = models.DateTimeField(default=timezone.now)
    validation_file_path = models.CharField(max_length=500, blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Validation for {self.document.name} - F1: {self.macro_f1_score:.3f}"

