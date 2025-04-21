import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(file_path="ice_cream_sales.csv", test_size=0.3, random_state=1):
    df = pd.read_csv(file_path)
    X = df[["temperature_celsius"]]
    y = df["units_sold"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
