from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from data_loader import load_and_split_data

def train_and_log_model(file_path="ice_cream_sales.csv", max_depth=5):
    X_train, X_test, y_train, y_test = load_and_split_data(file_path)

    model = DecisionTreeRegressor(max_depth=max_depth, random_state=1)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    mlflow.set_experiment("IceCreamSalesDecisionTree")
    with mlflow.start_run():
        mlflow.log_param("model_type", "DecisionTreeRegressor")
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "decision_tree_model")

    print(f"Model trained. MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
    return model

if __name__ == "__main__":
    train_and_log_model()
