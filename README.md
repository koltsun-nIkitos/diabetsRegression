# diabetsRegression

## О проекте

Проект по предсказанию прогрессирования диабета через год на основе медицинских показателей. Реализованы и сравнены модели машинного обучения для решения задачи регрессии.

## Данные
Используется встроенный датасет `sklearn.datasets.load_diabetes`:

### Признаки:
- **age** - возраст
- **sex** - пол
- **bmi** - индекс массы тела
- **bp** - среднее артериальное давление
- **s1-s6** - шесть различных анализов крови (tc, ldl, hdl, tch, ltg, glu)

### Целевая переменная:
- **target** - количественная мера прогрессирования диабета через год

## Технологии
- **Python 3.8+**
- **pandas** - анализ и обработка данных
- **numpy** - численные вычисления
- **matplotlib** - визуализация
- **scikit-learn** - машинное обучение
- **pickle** - сериализация моделей

### 1. Клонирование репозитория
```
bash:
git clone https://github.com/koltsun-nIkitos/diabetsRegression.git
cd diabetsRegression
```

### 2. Установка зависимостей 
```
bash:
pip install -r requirements.txt
```

### 3. Запуск проекта
```
bash:
python diabetes_regression.py
```


## Сравнение моделей:
- Модель              MAE	        MSE	        R²
- Random Forest	      42.95	        2903.58	    0.452
- Linear Regression	  44.28	        3108.57	    0.414

## Важность признаков (Random Forest):
- **bmi** (индекс массы тела) - 0.301
- **s5** (уровень глюкозы) - 0.138
- **bp** (артериальное давление) - 0.117
- **s6** - 0.102
- **s4** - 0.091

# Ключевые особенности
## 1. Предобработка данных
Стандартизация признаков с помощью StandardScaler
Правильное разделение на train/test (80/20)
Исследование корреляций между признаками

## 2. Модели машинного обучения
- * Linear Regression - базовая линейная модель
- * Random Forest - ансамблевый метод на основе деревьев

## 3. Оценка качества
- * MAE (Mean Absolute Error) - средняя абсолютная ошибка
- * MSE (Mean Squared Error) - средняя квадратичная ошибка
- * R² (R-squared) - коэффициент детерминации

## 4. Сериализация модели
Сохранение лучшей модели с помощью pickle
Сохранение scaler для корректного предсказания
Пример загрузки и использования модели

# Использование модели
## Загрузка и предсказание
```
import pickle
import numpy as np
```

# Загрузка модели
```
with open('best_diabetes_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
feature_names = model_data['feature_names']

# Новые данные для предсказания
new_patient = np.array([[0.05, 0.05, 0.05, 0.05, -0.05, 0.01, 0.02, 0.03, 0.04, 0.05]])

# Трансформация и предсказание
scaled_data = scaler.transform(new_patient)
prediction = model.predict(scaled_data)

print(f"Прогноз прогрессирования диабета: {prediction[0]:.1f}")
```

# Структура проекта
- diabetsRegression/
- ├── diabetes_regression.py    # Основной скрипт
- ├── best_diabetes_model.pkl   # Сохраненная модель
- ├── requirements.txt          # Зависимости
- ├── README.md                 # Документация
- └── images/                   # Графики и визуализации
-     ├── target_distribution.png
-     ├── models_comparison.png
-     └── random_forest.png


# Возможные улучшения
- *Добавление кросс-валидации
- *Подбор гиперпараметров (GridSearchCV)
- *Эксперименты с другими алгоритмами (XGBoost, SVM)
- *Feature engineering
- *Создание веб-интерфейса