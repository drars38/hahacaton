import pandas as pd
import os


def debug_training_data(csv_file_path='training_data.csv'):
    """Подробная диагностика обучающих данных"""

    print("🔍 ЗАПУСК ДИАГНОСТИКИ ДАННЫХ")
    print("=" * 50)

    try:
        # Проверка существования файла
        if not os.path.exists(csv_file_path):
            print(f"❌ Файл {csv_file_path} не существует!")
            return False

        print(f"✅ Файл найден: {csv_file_path}")
        print(f"📏 Размер файла: {os.path.getsize(csv_file_path) / 1024 / 1024:.2f} MB")

        # Загрузка данных
        print("\n📁 Загружаем данные...")
        df = pd.read_csv(csv_file_path)

        print(f"✅ Загружено записей: {len(df):,}")
        print(f"📊 Колонки: {list(df.columns)}")

        # Проверка необходимых колонок
        required_columns = ['text', 'label']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"❌ Отсутствуют колонки: {missing_columns}")
            return False

        print("✅ Все необходимые колонки присутствуют")

        # Анализ колонки 'text'
        print("\n📝 АНАЛИЗ КОЛОНКИ 'text':")
        print(f"   Тип данных: {df['text'].dtype}")
        print(f"   Пропуски: {df['text'].isnull().sum()}")
        print(f"   Пустые строки: {(df['text'] == '').sum()}")

        # Примеры текстов
        sample_texts = df['text'].head(3).tolist()
        print(f"   Примеры текстов:")
        for i, text in enumerate(sample_texts):
            print(f"     {i + 1}. {str(text)[:100]}...")

        # Анализ колонки 'label'
        print("\n🏷️ АНАЛИЗ КОЛОНКИ 'label':")
        print(f"   Тип данных: {df['label'].dtype}")
        print(f"   Пропуски: {df['label'].isnull().sum()}")

        # Статистика по меткам
        label_stats = df['label'].value_counts().sort_index()
        print(f"   Распределение меток:")
        for label, count in label_stats.items():
            print(f"     {label}: {count:,} ({count / len(df) * 100:.1f}%)")

        # Проверка уникальных значений
        unique_labels = df['label'].unique()
        print(f"   Уникальные значения: {sorted(unique_labels)}")

        # Проверка на нечисловые значения
        non_numeric = df[~df['label'].apply(lambda x: str(x).isdigit())]['label'].unique()
        if len(non_numeric) > 0:
            print(f"❌ Обнаружены нечисловые значения: {non_numeric}")
            return False

        # Преобразование в int
        print("\n🔧 Преобразование label в int...")
        try:
            df['label'] = df['label'].astype(int)
            unique_int_labels = df['label'].unique()
            print(f"✅ Уникальные int значения: {sorted(unique_int_labels)}")
        except Exception as e:
            print(f"❌ Ошибка преобразования: {e}")
            return False

        # Проверка диапазона меток
        invalid_labels = [label for label in unique_int_labels if label not in [0, 1, 2]]
        if invalid_labels:
            print(f"❌ Обнаружены невалидные метки: {invalid_labels}")
            return False

        print("✅ Все метки в допустимом диапазоне [0, 1, 2]")

        # Проверка памяти
        print(f"\n💾 Использование памяти: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

        print("\n🎉 ДИАГНОСТИКА ЗАВЕРШЕНА УСПЕШНО!")
        return True

    except Exception as e:
        print(f"❌ ОШИБКА ДИАГНОСТИКИ: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    debug_training_data('training_data.csv')