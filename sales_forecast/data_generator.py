import pandas as pd
import numpy as np

def generate_ice_cream_data(output_path="ice_cream_sales.csv", num_samples=150, temp_min=10, temp_max=40, seed=42):
    np.random.seed(seed)
    temperatures = np.random.uniform(temp_min, temp_max, num_samples)
    sales = 30 + temperatures * 12 + np.random.normal(0, 25, num_samples)

    df = pd.DataFrame({
        "temperature_celsius": temperatures,
        "units_sold": sales
    })
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    generate_ice_cream_data()
