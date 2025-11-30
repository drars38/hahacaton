from django.contrib import admin
from .models import TextDocument, TextEntry, ValidationResult


@admin.register(TextDocument)
class TextDocumentAdmin(admin.ModelAdmin):
    list_display = ['name', 'uploaded_at', 'entries_count']
    search_fields = ['name']
    
    def entries_count(self, obj):
        return obj.entries.count()
    entries_count.short_description = 'Количество записей'


@admin.register(TextEntry)
class TextEntryAdmin(admin.ModelAdmin):
    list_display = ['text_preview', 'source', 'sentiment_score', 'is_manually_corrected', 'created_at']
    list_filter = ['sentiment_score', 'is_manually_corrected', 'document']
    search_fields = ['text', 'source']
    
    def text_preview(self, obj):
        return obj.text[:50] + '...' if len(obj.text) > 50 else obj.text
    text_preview.short_description = 'Текст'


@admin.register(ValidationResult)
class ValidationResultAdmin(admin.ModelAdmin):
    list_display = ['document', 'macro_f1_score', 'precision', 'recall', 'created_at']
    list_filter = ['created_at']


