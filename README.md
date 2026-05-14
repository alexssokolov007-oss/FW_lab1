# Лабораторная работа №1 — ML Pipeline

**Студент:** Соколов Александр Сергеевич  
**Дисциплина:** Фреймворки машинного обучения  
**Датасет:** [Student Depression and Lifestyle — 100k](https://www.kaggle.com/datasets/aldinwhyudii/student-depression-and-lifestyle-100k-data)

## Задачи

1. Разведочный анализ данных (EDA) с визуализацией
2. Предобработка: пропуски, выбросы, кодирование, масштабирование
3. Отбор признаков и Feature Engineering
4. Разделение данных на train / val / test без утечки данных
5. Linear Regression — прогноз CGPA (шкала 0–4)
6. Logistic Regression — классификация Depression, эксперименты с lr и epochs
7. Доказательство отсутствия переобучения
8. Оценка качества моделей (метрики, ROC, confusion matrix)
9. Выводы о качестве и влиянии предобработки

## Запуск

```bash
jupyter notebook lab1_ml_pipeline.ipynb
```

## Структура

```
lab1/
├── lab1_ml_pipeline.ipynb
├── student_lifestyle_100k.csv
└── README.md
```
