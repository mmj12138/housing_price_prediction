from src.data_preprocessing import load_data, preprocess_data
from src.train import train_baseline, train_mlp
from src.feature_analysis import feature_correlation, correlation_heatmap, plot_feature_coeffs
from src.evaluate import evaluate_model

def main():
    train_df, test_df = load_data()
    preprocessor, X_train, X_val, y_train, y_val, features = preprocess_data(train_df)

    print(test_df)

    print(features)

    print("\n=== Baseline Model: Ridge ===")
    ridge_pipe = train_baseline(preprocessor, X_train, y_train, X_val, y_val, model_type="ridge")

    print("\n=== Baseline Model: Lasso ===")
    lasso_pipe = train_baseline(preprocessor, X_train, y_train, X_val, y_val, model_type="lasso")

    # Feature analysis on original train dataframe (limited to selected features + target)
    print("\n=== Feature Correlation Analysis ===")
    feature_correlation(train_df[features + ["SalePrice"]], target="SalePrice", top_n=20)
    correlation_heatmap(train_df[features + ["SalePrice"]], target="SalePrice", top_n=20)

    # Show Ridge coefficients (feature importance)
    plot_feature_coeffs(ridge_pipe, preprocessor, feature_names=features, top_n=20)

    print("\n=== Neural Network (MLP) ===")
    mlp_model, fitted_preproc = train_mlp(preprocessor, X_train, X_val, y_train, y_val)

    print("\n✅ All done.")


if __name__ == '__main__':
    main()
