from data_generator import generate_ice_cream_data
from train_model import train_and_log_model
from feature_importance_plot import plot_feature_importance

def main():
    output_csv = "ice_cream_sales.csv"
    generate_ice_cream_data(output_path=output_csv, num_samples=200, temp_min=12, temp_max=38)

    model = train_and_log_model(file_path=output_csv, max_depth=6)

    # Only makes sense with more than one feature, but included here for future extensibility
    plot_feature_importance(model, feature_names=["temperature_celsius"])

if __name__ == "__main__":
    main()
