import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1200

age        = rng.integers(18, 27, n).astype(float)
gender     = rng.choice(["Male", "Female"], n)
department = rng.choice(
    ["Science", "Engineering", "Medical", "Business", "Arts"], n,
    p=[0.25, 0.25, 0.20, 0.20, 0.10]
)
sleep      = rng.normal(6.5, 1.2, n)
study      = rng.normal(5.0, 1.5, n)
physical   = rng.normal(150, 60, n)
social     = rng.normal(3.0, 1.8, n)
stress     = np.clip(rng.normal(5.5, 2.0, n), 2, 10)  # шкала 2–10

cgpa = np.clip(
    5.0 + 0.35*study + 0.20*sleep - 0.10*(stress / 2) - 0.05*social
    + rng.normal(0, 0.4, n),
    4.0, 10.0
)

log_odds = (
    -0.8
    + 0.35 * (stress / 5)
    - 0.30 * sleep
    + 0.20 * social
    + 0.15 * (department == "Medical").astype(float)
    + rng.normal(0, 0.3, n)
)
depression = rng.random(n) < 1 / (1 + np.exp(-log_odds))

df = pd.DataFrame({
    "Student_ID":         np.arange(1001, 1001 + n),
    "Age":                age,
    "Gender":             gender,
    "Department":         department,
    "CGPA":               cgpa,
    "Sleep_Duration":     sleep,
    "Study_Hours":        study,
    "Social_Media_Hours": social,
    "Physical_Activity":  physical,
    "Stress_Level":       stress,
    "Depression":         depression,
})

# Намеренно вносим типичные проблемы реальных данных

# Несогласованный формат Gender
mask = rng.choice(n, 80, replace=False)
df.loc[mask[:30], "Gender"] = "male"
df.loc[mask[30:60], "Gender"] = "FEMALE"
df.loc[mask[60:], "Gender"] = "M"

# Пропущенные значения
for col, frac in [("Sleep_Duration", 0.06), ("Study_Hours", 0.05),
                  ("Physical_Activity", 0.07), ("Department", 0.04),
                  ("Stress_Level", 0.03), ("Social_Media_Hours", 0.04)]:
    df.loc[rng.choice(n, int(n * frac), replace=False), col] = np.nan

# Логические ошибки
df.loc[rng.choice(n, 5, replace=False), "Age"]              = rng.choice([-3, -5, 150, 200], 5)
df.loc[rng.choice(n, 5, replace=False), "Sleep_Duration"]   = rng.choice([-1, 0, 28, 30], 5)
df.loc[rng.choice(n, 5, replace=False), "CGPA"]             = rng.choice([-2, 0.5, 11, 15], 5)
df.loc[rng.choice(n, 5, replace=False), "Study_Hours"]      = rng.choice([-1, 25, 30], 5)
df.loc[rng.choice(n, 5, replace=False), "Physical_Activity"] = rng.choice([-10, -50, 1500, 2000], 5)
df.loc[rng.choice(n, 5, replace=False), "Social_Media_Hours"] = rng.choice([-2, -1, 30, 28], 5)

# Дубликаты
df = pd.concat([df, df.iloc[rng.choice(n, 30, replace=False)]], ignore_index=True)

# Мусорные столбцы
df["CGPA_copy"]      = df["CGPA"] + rng.normal(0, 0.001, len(df))
df["Useless_Column"] = "A"
df["Random_Noise"]   = rng.random(len(df))

df.sample(frac=1, random_state=42).reset_index(drop=True).to_csv("student_data.csv", index=False)
print(f"Датасет сохранён: {len(df)} строк, {df.shape[1]} столбцов")
