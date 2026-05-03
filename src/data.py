import pandas as pd
import numpy as np

np.random.seed(42)

n = 500

df = pd.DataFrame({"age": np.random.randint(18,70,n), "tenure": np.random.randint(1,10,n), "monthly_charges": np.random.randint(20,120,n), "total_spent": np.random.randint(100,20000,n), "contract_type": np.random.choice(["monthly", "yearly"], n), "internet_service": np.random.choice(["dsl", "fiber", "none"], n)})

# Create target (churn logic)
df["churn"] = ((df["monthly_charges"] > 80) & (df["tenure"] < 4)).astype(int)

df.to_csv("data/churn.csv", index=False)

print("Advanced dataset created!")