import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso
from .models import get_baseline_model, MLP
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import clone

from joblib import dump

from sklearn.exceptions import NotFittedError

from sklearn.compose import ColumnTransformer

from .config import CONFIG

def _to_array(X):
    # helper: handle sparse matrices
    try:
        return X.toarray()
    except Exception:
        return np.asarray(X)

def train_baseline(preprocessor, X_train, y_train, X_val, y_val, model_type='ridge'):
    # pipeline
    model = Pipeline([('preprocessor', preprocessor), ('regressor', get_baseline_model(model_type))])
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    print(f"{model_type.upper()} → RMSE: {rmse:.2f}, R²: {r2:.4f}")
    # save pipeline
    dump(model, f'model_{model_type}.joblib')
    return model

def train_mlp(preprocessor, X_train, X_val, y_train, y_val):
    # fit preprocessor first
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    X_train_np = _to_array(X_train_proc).astype('float32')
    X_val_np = _to_array(X_val_proc).astype('float32')

    X_train_t = torch.tensor(X_train_np)
    y_train_t = torch.tensor(y_train.values.reshape(-1,1)).float()
    X_val_t = torch.tensor(X_val_np)
    y_val_t = torch.tensor(y_val.values.reshape(-1,1)).float()

    input_dim = X_train_t.shape[1]
    mlp = MLP(
        input_dim=input_dim,
        hidden_dims=CONFIG['mlp_params']['hidden_dims'],
        dropout=CONFIG['mlp_params']['dropout']
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=CONFIG['mlp_params']['lr'])

    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=CONFIG['mlp_params']['batch_size'], shuffle=True)

    for epoch in range(CONFIG['mlp_params']['epochs']):
        mlp.train()
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = mlp(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
        if epoch % 10 == 0 or epoch == CONFIG['mlp_params']['epochs']-1:
            mlp.eval()
            with torch.no_grad():
                val_pred = mlp(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
            print(f"Epoch {epoch}: Train Loss={running/len(loader):.4f}, Val Loss={val_loss:.4f}")
    # save model state dict
    torch.save(mlp.state_dict(), 'mlp_state.pth')
    return mlp, preprocessor
