import pandas as pd

titanic_df = pd.read_csv("./datasets/titanic.csv")
print(titanic_df["Survived"].value_counts())
heart_df = pd.read_csv("./datasets/heart.csv")
print(heart_df["target"].value_counts())