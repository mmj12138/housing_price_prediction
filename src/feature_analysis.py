import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
from sklearn.compose import ColumnTransformer

def feature_correlation(df, target='SalePrice', top_n=20, save_path=None):
    # compute correlations with target (numeric-only)
    corr = df.corr(numeric_only=True)[target].sort_values(ascending=False)
    top = corr.head(top_n)
    print('📊 Top correlated features with', target)
    print(top)
    plt.figure(figsize=(8,6))
    # sns.barplot(x=top.values, y=top.index)
    plt.xlabel('Correlation coefficient')
    plt.title(f'Top {top_n} features correlated with {target}')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def correlation_heatmap(df, target='SalePrice', top_n=20):
    corr_all = df.corr(numeric_only=True)
    # pick features whose absolute correlation with target is largest
    top_features = corr_all[target].abs().sort_values(ascending=False).head(top_n).index.tolist()
    sub = df[top_features].corr()
    plt.figure(figsize=(11,9))
    # sns.heatmap(sub, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title(f'Correlation heatmap (Top {top_n} variables by |corr| to {target})')
    plt.tight_layout()
    plt.show()

def plot_feature_coeffs(pipe, preprocessor, feature_names, top_n=20):
    # extract numeric and categorical feature names after preprocessor
    # This function assumes preprocessor is a ColumnTransformer with ('num', transformer, numeric_cols) and ('cat', ohe, cat_cols)
    numeric_cols = []
    cat_cols = []
    for name, trans, cols in preprocessor.transformers_:
        if name == 'num':
            numeric_cols = list(cols)
        elif name == 'cat':
            cat_cols = list(cols)

    # try to fetch one-hot names if available
    ohe = None
    for name, trans, cols in preprocessor.transformers_:
        if name == "cat":
            # ColumnTransformer stores fitted transformer in trans.named_steps if it's a Pipeline; else trans
            if hasattr(trans, "named_steps") and "ohe" in trans.named_steps:
                ohe = trans.named_steps["ohe"]
            elif hasattr(trans, "categories_") or hasattr(trans, "categories"):
                ohe = trans

    feature_expanded = []
    feature_expanded += numeric_cols

    # === safe one-hot name extraction ===
    ohe_names = []
    if ohe is not None and hasattr(ohe, "get_feature_names_out"):
        try:
            ohe_names = list(ohe.get_feature_names_out(cat_cols))
        except Exception:
            # sklearn <1.0 fallback
            cats_attr = getattr(ohe, "categories_", None)
            if cats_attr is None:
                cats_attr = getattr(ohe, "categories", [])
            for i, cats in enumerate(cats_attr):
                if i < len(cat_cols):  # ✅ 防止越界
                    prefix = cat_cols[i]
                    ohe_names += [f"{prefix}_{c}" for c in cats]
    if ohe_names:
        feature_expanded += ohe_names
    else:
        feature_expanded += cat_cols

    # get coefficients from the final estimator
    try:
        coef = pipe.named_steps['regressor'].coef_
    except Exception:
        print('Could not extract coefficients from pipeline.')
        return

    if len(coef) != len(feature_expanded):
        # mismatch; try to reduce to numeric feature names only
        print('Warning: feature-expanded length != coef length. Showing numeric feature coefficients only.')
        feature_expanded = numeric_cols
        coef = coef[:len(feature_expanded)]

    coef_series = pd.Series(coef.flatten(), index=feature_expanded).sort_values(key=np.abs, ascending=False)
    top = coef_series.head(top_n)
    print('\n🔎 Top features by absolute coefficient:')
    print(top)
    plt.figure(figsize=(8,6))
    # sns.barplot(x=top.values, y=top.index)
    plt.xlabel('Coefficient value (signed)')
    plt.title('Top feature coefficients (by absolute value)')
    plt.tight_layout()
    plt.show()
