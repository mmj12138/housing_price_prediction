from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_model(model, X_test, y_test, is_torch=False):
    if is_torch:
        # model should be a torch model and X_test a tensor
        preds = model(X_test).detach().numpy().reshape(-1)
        y_true = y_test.numpy().reshape(-1)
    else:
        preds = model.predict(X_test)
        y_true = y_test
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
    return rmse, mae, r2
