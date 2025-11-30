from django.core.management.base import BaseCommand
from analyzer.models import TextDocument, TextEntry
from analyzer.services import DataProcessor
import os
from django.conf import settings


class Command(BaseCommand):
    help = 'Загружает CSV файлы из указанной папки в базу данных'

    def add_arguments(self, parser):
        parser.add_argument(
            'folder_path',
            type=str,
            help='Путь к папке с CSV файлами'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=1000,
            help='Размер батча для обработки (по умолчанию 1000)'
        )

    def handle(self, *args, **options):
        folder_path = options['folder_path']
        batch_size = options['batch_size']
        
        if not os.path.exists(folder_path):
            self.stdout.write(self.style.ERROR(f'Папка не найдена: {folder_path}'))
            return
        
        # Ищем CSV файлы в папке
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        if not csv_files:
            self.stdout.write(self.style.WARNING('CSV файлы не найдены в папке'))
            return
        
        self.stdout.write(self.style.SUCCESS(f'Найдено {len(csv_files)} CSV файлов'))
        
        processor = DataProcessor()
        total_entries = 0
        
        for csv_file in csv_files:
            file_path = os.path.join(folder_path, csv_file)
            self.stdout.write(f'Обработка файла: {csv_file}...')
            
            try:
                # Создаем документ
                document, created = TextDocument.objects.get_or_create(
                    name=csv_file,
                    defaults={'file_path': file_path}
                )
                
                if not created:
                    self.stdout.write(self.style.WARNING(f'  Файл {csv_file} уже загружен, пропускаем'))
                    continue
                
                # Обрабатываем файл
                df = processor.process_csv(file_path)
                texts, sources = processor.extract_texts(df)
                self.stdout.write(f'  Найдено {len(texts)} записей')
                
                results = processor.process_texts(texts, sources, batch_size=batch_size)
                
                # Сохраняем результаты в БД батчами
                entries = []
                saved_count = 0
                
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
                        saved_count += len(entries)
                        self.stdout.write(f'  Сохранено {saved_count}/{len(results)} записей...', ending='\r')
                        entries = []
                
                if entries:
                    TextEntry.objects.bulk_create(entries, ignore_conflicts=True)
                    saved_count += len(entries)
                
                total_entries += saved_count
                self.stdout.write(self.style.SUCCESS(f'  ✓ Файл {csv_file} обработан: {saved_count} записей'))
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'  ✗ Ошибка при обработке {csv_file}: {str(e)}'))
                if document and document.id:
                    document.delete()
        
        self.stdout.write(self.style.SUCCESS(f'\nВсего загружено записей: {total_entries}'))

