import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    plt.figure(figsize=(6, 4))
    plt.bar(feature_names, importance)
    plt.title("Feature Importance")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.show()
