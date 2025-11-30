from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.db.models import Q
import pandas as pd
import os
from django.conf import settings

from .models import TextDocument, TextEntry, ValidationResult
from .serializers import TextDocumentSerializer, TextEntrySerializer, ValidationResultSerializer
from .services import DataProcessor, ValidationService


def index(request):
    """Главная страница приложения"""
    return render(request, 'analyzer/index.html')


class TextDocumentViewSet(viewsets.ModelViewSet):
    """ViewSet для работы с документами"""
    queryset = TextDocument.objects.all()
    serializer_class = TextDocumentSerializer
    parser_classes = [MultiPartParser, FormParser]
    
    @action(detail=False, methods=['post'])
    def upload(self, request):
        """Загрузка CSV файла с текстами"""
        if 'file' not in request.FILES:
            return Response({'error': 'Файл не найден'}, status=status.HTTP_400_BAD_REQUEST)
        
        file = request.FILES['file']
        
        # Сохраняем файл
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        file_path = os.path.join(settings.MEDIA_ROOT, file.name)
        
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        
        # Создаем документ
        document = TextDocument.objects.create(
            name=file.name,
            file_path=file_path
        )
        
        # Обрабатываем файл
        processor = DataProcessor()
        try:
            df = processor.process_csv(file_path)
            texts, sources = processor.extract_texts(df)
            results = processor.process_texts(texts, sources, batch_size=1000)
            
            # Сохраняем результаты в БД батчами для ускорения
            entries = []
            batch_size = 1000  # Размер батча для bulk_create
            
            for result in results:
                entry = TextEntry(
                    document=document,
                    text=result['text'],
                    source=result['source'],
                    original_text=result['original_text'],
                    normalized_text=result['normalized_text'],
                    sentiment_score=result['sentiment_score'],
                    confidence=result['confidence']
                )
                entries.append(entry)
                
                # Сохраняем батчами
                if len(entries) >= batch_size:
                    TextEntry.objects.bulk_create(entries, ignore_conflicts=True)
                    entries = []
            
            # Сохраняем оставшиеся записи
            if entries:
                TextEntry.objects.bulk_create(entries, ignore_conflicts=True)
            
            serializer = self.get_serializer(document)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        
        except Exception as e:
            document.delete()
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=True, methods=['get'])
    def export_csv(self, request, pk=None):
        """Экспорт результатов в CSV"""
        document = self.get_object()
        entries = document.entries.all()
        
        data = []
        for entry in entries:
            data.append({
                'text': entry.text,
                'source': entry.source,
                'normalized_text': entry.normalized_text,
                'sentiment_score': entry.sentiment_score,
                'sentiment_label': entry.get_sentiment_score_display(),
                'confidence': entry.confidence,
                'is_manually_corrected': entry.is_manually_corrected
            })
        
        df = pd.DataFrame(data)
        response = HttpResponse(content_type='text/csv; charset=utf-8')
        response['Content-Disposition'] = f'attachment; filename="{document.name}_results.csv"'
        
        df.to_csv(response, index=False, encoding='utf-8-sig')
        return response
    
    @action(detail=False, methods=['post'])
    def upload_from_folder(self, request):
        """Загрузка файлов из указанной папки"""
        folder_path = request.data.get('folder_path', '')
        if not folder_path:
            return Response({'error': 'Путь к папке не указан'}, status=status.HTTP_400_BAD_REQUEST)
        
        if not os.path.exists(folder_path):
            return Response({'error': 'Папка не найдена'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Ищем CSV файлы в папке
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        if not csv_files:
            return Response({'error': 'CSV файлы не найдены в папке'}, status=status.HTTP_400_BAD_REQUEST)
        
        processor = DataProcessor()
        uploaded_documents = []
        errors = []
        
        for csv_file in csv_files:
            file_path = os.path.join(folder_path, csv_file)
            try:
                # Создаем документ
                document = TextDocument.objects.create(
                    name=csv_file,
                    file_path=file_path
                )
                
                # Обрабатываем файл
                df = processor.process_csv(file_path)
                texts, sources = processor.extract_texts(df)
                results = processor.process_texts(texts, sources)
                
                # Сохраняем результаты в БД батчами
                entries = []
                batch_size = 1000
                
                for result in results:
                    entry = TextEntry(
                        document=document,
                        text=result['text'],
                        source=result['source'],
                        original_text=result['original_text'],
                        normalized_text=result['normalized_text'],
                        sentiment_score=result['sentiment_score'],
                        confidence=result['confidence']
                    )
                    entries.append(entry)
                    
                    if len(entries) >= batch_size:
                        TextEntry.objects.bulk_create(entries, ignore_conflicts=True)
                        entries = []
                
                if entries:
                    TextEntry.objects.bulk_create(entries, ignore_conflicts=True)
                
                uploaded_documents.append({
                    'id': document.id,
                    'name': document.name,
                    'entries_count': len(results)
                })
                
            except Exception as e:
                errors.append({
                    'file': csv_file,
                    'error': str(e)
                })
        
        return Response({
            'uploaded': uploaded_documents,
            'errors': errors,
            'total_files': len(csv_files),
            'success_count': len(uploaded_documents)
        }, status=status.HTTP_201_CREATED)
    
    @action(detail=True, methods=['post'])
    def validate(self, request, pk=None):
        """Валидация результатов с помощью тестовой выборки"""
        document = self.get_object()
        
        if 'file' not in request.FILES:
            return Response({'error': 'Валидационный файл не найден'}, 
                          status=status.HTTP_400_BAD_REQUEST)
        
        file = request.FILES['file']
        
        # Сохраняем файл
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        file_path = os.path.join(settings.MEDIA_ROOT, f"validation_{file.name}")
        
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        
        # Обрабатываем валидационный файл
        processor = DataProcessor()
        try:
            validation_df = processor.process_csv(file_path)
            
            # Получаем предсказания для документа
            entries = document.entries.all().order_by('id')
            predictions = []
            for entry in entries:
                predictions.append({
                    'sentiment_score': entry.sentiment_score
                })
            
            # Вычисляем метрики
            validation_service = ValidationService()
            metrics = validation_service.validate_predictions(validation_df, predictions)
            
            # Сохраняем результат валидации
            validation_result = ValidationResult.objects.create(
                document=document,
                macro_f1_score=metrics['macro_f1_score'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                validation_file_path=file_path
            )
            
            serializer = ValidationResultSerializer(validation_result)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class TextEntryViewSet(viewsets.ModelViewSet):
    """ViewSet для работы с текстовыми записями"""
    queryset = TextEntry.objects.all()
    serializer_class = TextEntrySerializer
    
    def get_queryset(self):
        queryset = TextEntry.objects.all()
        
        # Фильтрация по документу
        document_id = self.request.query_params.get('document', None)
        if document_id:
            queryset = queryset.filter(document_id=document_id)
        
        # Фильтрация по тональности
        sentiment = self.request.query_params.get('sentiment', None)
        if sentiment is not None:
            queryset = queryset.filter(sentiment_score=int(sentiment))
        
        # Фильтрация по источнику
        source = self.request.query_params.get('source', None)
        if source:
            queryset = queryset.filter(source__icontains=source)
        
        # Поиск по тексту
        search = self.request.query_params.get('search', None)
        if search:
            queryset = queryset.filter(
                Q(text__icontains=search) | 
                Q(normalized_text__icontains=search) |
                Q(source__icontains=search)
            )
        
        return queryset
    
    @action(detail=True, methods=['patch'])
    def update_sentiment(self, request, pk=None):
        """Ручная корректировка разметки"""
        entry = self.get_object()
        sentiment_score = request.data.get('sentiment_score')
        
        if sentiment_score is None or sentiment_score not in [0, 1, 2]:
            return Response({'error': 'Неверное значение sentiment_score'}, 
                          status=status.HTTP_400_BAD_REQUEST)
        
        entry.sentiment_score = sentiment_score
        entry.is_manually_corrected = True
        entry.save()
        
        serializer = self.get_serializer(entry)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def statistics(self, request):
        """Статистика по записям"""
        queryset = self.get_queryset()
        
        total = queryset.count()
        sentiment_counts = {
            'neutral': queryset.filter(sentiment_score=0).count(),  # 0 = Нейтральная
            'positive': queryset.filter(sentiment_score=1).count(),  # 1 = Положительная
            'negative': queryset.filter(sentiment_score=2).count()  # 2 = Отрицательная
        }
        
        sources = queryset.exclude(source__isnull=True).exclude(source='').values_list('source', flat=True).distinct()
        
        return Response({
            'total': total,
            'sentiment_distribution': sentiment_counts,
            'unique_sources': list(sources)[:20]  # Первые 20 источников
        })


class ValidationResultViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet для просмотра результатов валидации"""
    queryset = ValidationResult.objects.all()
    serializer_class = ValidationResultSerializer

