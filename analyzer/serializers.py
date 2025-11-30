from rest_framework import serializers
from .models import TextDocument, TextEntry, ValidationResult


class TextEntrySerializer(serializers.ModelSerializer):
    sentiment_label = serializers.CharField(source='get_sentiment_score_display', read_only=True)
    
    class Meta:
        model = TextEntry
        fields = ['id', 'text', 'source', 'original_text', 'normalized_text', 
                  'sentiment_score', 'sentiment_label', 'confidence', 
                  'is_manually_corrected', 'created_at']
        read_only_fields = ['id', 'created_at']


class TextDocumentSerializer(serializers.ModelSerializer):
    entries_count = serializers.SerializerMethodField()
    
    class Meta:
        model = TextDocument
        fields = ['id', 'name', 'uploaded_at', 'entries_count']
        read_only_fields = ['id', 'uploaded_at']
    
    def get_entries_count(self, obj):
        return obj.entries.count()


class ValidationResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = ValidationResult
        fields = ['id', 'document', 'macro_f1_score', 'precision', 'recall', 
                  'created_at', 'validation_file_path']
        read_only_fields = ['id', 'created_at']


