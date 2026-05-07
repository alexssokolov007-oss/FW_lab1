import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1200

age        = rng.integers(18, 28, n).astype(float)
gender     = rng.choice(["Male", "Female"], n)
sleep      = rng.normal(6.5, 1.2, n)
study      = rng.normal(5.0, 1.5, n)
physical   = rng.normal(3.0, 1.5, n)
diet       = rng.choice(["Healthy", "Moderate", "Unhealthy"], n, p=[0.3, 0.45, 0.25])
social     = rng.normal(3.0, 1.8, n)
academic   = rng.integers(1, 6, n).astype(float)
fin_stress = rng.choice(["Yes", "No"], n, p=[0.4, 0.6])
fam_hist   = rng.choice(["Yes", "No"], n, p=[0.3, 0.7])

cgpa = np.clip(
    5.0 + 0.35*study + 0.20*sleep - 0.15*academic - 0.10*social + rng.normal(0, 0.4, n),
    4.0, 10.0
)

log_odds = (
    -0.8
    + 0.4 * academic
    - 0.3 * sleep
    + 0.2 * social
    + 0.5 * (fam_hist == "Yes").astype(float)
    + rng.normal(0, 0.3, n)
)
depression = (rng.random(n) < 1 / (1 + np.exp(-log_odds))).astype(int)

df = pd.DataFrame({
    "Student_ID":         np.arange(1001, 1001 + n),
    "Age":                age,
    "Gender":             gender,
    "Sleep_Duration":     sleep,
    "Study_Hours":        study,
    "Physical_Activity":  physical,
    "Dietary_Habits":     diet,
    "Social_Media_Hours": social,
    "Academic_Pressure":  academic,
    "Financial_Stress":   fin_stress,
    "Family_History":     fam_hist,
    "CGPA":               cgpa,
    "Depression":         depression,
})

# Намеренно вносим типичные проблемы реальных данных

mask = rng.choice(n, 80, replace=False)
df.loc[mask[:30], "Gender"] = "male"
df.loc[mask[30:60], "Gender"] = "FEMALE"
df.loc[mask[60:], "Gender"] = "M"

for col, frac in [("Sleep_Duration", 0.06), ("Study_Hours", 0.05),
                  ("Physical_Activity", 0.07), ("Dietary_Habits", 0.04),
                  ("Academic_Pressure", 0.03), ("Financial_Stress", 0.04)]:
    df.loc[rng.choice(n, int(n * frac), replace=False), col] = np.nan

df.loc[rng.choice(n, 5, replace=False), "Age"]          = rng.choice([-3, -5, 150, 200, 999], 5)
df.loc[rng.choice(n, 5, replace=False), "Sleep_Duration"] = rng.choice([-1, 0, 28, 30], 5)
df.loc[rng.choice(n, 5, replace=False), "CGPA"]         = rng.choice([-2, 0.5, 11, 15], 5)
df.loc[rng.choice(n, 5, replace=False), "Study_Hours"]  = rng.choice([-1, 25, 30], 5)

df = pd.concat([df, df.iloc[rng.choice(n, 30, replace=False)]], ignore_index=True)

df["CGPA_copy"]      = df["CGPA"] + rng.normal(0, 0.001, len(df))
df["Useless_Column"] = "A"
df["Random_Noise"]   = rng.random(len(df))

df.sample(frac=1, random_state=42).reset_index(drop=True).to_csv("student_data.csv", index=False)
print(f"Датасет сохранён: {len(df)} строк, {df.shape[1]} столбцов")
