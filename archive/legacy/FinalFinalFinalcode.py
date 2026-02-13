#import packages
import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler #MinMaxScaler #Normalizer #StandardScaler 
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm
import time

start_time = time.time()

##########################################################################

#functions

#seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#definitions of models

#standard RNN, not so great with long term memory
class StockRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

#LSTM, better with long term memory
class StockLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

#GRU, less parameters
class StockGRU(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

#create sequences aligns all the data
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len][0])  # Predict Close/Last
    return np.array(X), np.array(y)

#function for preparing data
def prepare_data(df, features, seq_len, scale, device, train_year, test_year):

    df_train = df[df["Date"].dt.year <= train_year].copy()
    df_test = df[df["Date"].dt.year == test_year].copy()

    # Scale only on training
    scale.fit(df_train[features])
    scaled_train = scale.transform(df_train[features])
    scaled_test = scale.transform(df_test[features])

    # Create sequences separately
    X_train, y_train = create_sequences(scaled_train, seq_len)
    X_test, y_test = create_sequences(scaled_test, seq_len)

    # Dates for plotting
    #dates_train = df_train["Date"].iloc[seq_len:].reset_index(drop=True)
    dates_test = df_test["Date"].iloc[seq_len:].reset_index(drop=True)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    return dates_test, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

#function for training
def train_model(model, train_loader, scaler, optimizer, loss_fn, n_epochs, device):
    train_rmse_scores = []
    for epoch in range(n_epochs):
        model.train()
        preds_epoch = []
        actuals_epoch = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

            preds_epoch.append(preds.detach().cpu().numpy())
            actuals_epoch.append(yb.detach().cpu().numpy())

        # Calculate RMSE
        preds_epoch = np.concatenate(preds_epoch)
        actuals_epoch = np.concatenate(actuals_epoch)

        preds_unscaled = scaler.inverse_transform(
            np.column_stack([preds_epoch.squeeze(), np.zeros((len(preds_epoch), 3))])
        )[:, 0]
        actuals_unscaled = scaler.inverse_transform(
            np.column_stack([actuals_epoch.squeeze(), np.zeros((len(actuals_epoch), 3))])
        )[:, 0]

        rmse = root_mean_squared_error(actuals_unscaled, preds_unscaled)
        train_rmse_scores.append(rmse)
    
    return train_rmse_scores

############################################################################

#reading data
#df = pd.read_csv("S&P500 5Y.csv")
#df = pd.read_csv("NASDAQ-100 (NDX) Historical Data 5Y.csv")
df = pd.read_csv("Dow Jones Industrial Average Historical Data 5Y.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
df = df.sort_values("Date")
features = ["Close/Last", "Open", "High", "Low"]

##########################################################################

#hyperparameters

seed_num=20
set_seed(seed_num)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

norm=RobustScaler()
loss_fn = nn.MSELoss()

MAX_EPOCHS = 50
seq_len = [50]
batch_size = [64]

#for tuning
hidden_sizes = [32, 64, 128, 256]
learning_rates = [0.001, 0.005, 0.01]
num_layers_list = [1, 2, 3]

gen = torch.Generator().manual_seed(seed_num)
grid = list(product(hidden_sizes, learning_rates, num_layers_list))

for SEQ_LEN in seq_len:
    print(f"\n Running with Sequence Length: {SEQ_LEN}")
    
    results = {}

    #preparation of data
    dates, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = \
        prepare_data(df, features, SEQ_LEN, norm, device, train_year=2023, test_year=2024)
    
    for BATCH_SIZE in batch_size:
        print(f"\n Running with Batch Size: {BATCH_SIZE}")
        #load data
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), BATCH_SIZE, shuffle=True, generator=gen)
        
        ###########################################################################
        
        #training
        
        for name, ModelClass in [("GRU", StockGRU), ("LSTM", StockLSTM), ("RNN", StockRNN)]:
            results[name] = {}
            
            for HIDDEN_SIZE, LEARNING_RATE, NUM_LAYERS in tqdm(grid, desc=name):
        
                #setup the model and create label for plot
                model = ModelClass(input_size=4, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                
                #training        
                train_rmse_scores = train_model(model, train_loader, norm, optimizer, loss_fn, MAX_EPOCHS, device)
        
                #predictions/evaluation
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_test_tensor.to(device)).squeeze().cpu().numpy()
                    y_actual = y_test_tensor.squeeze().cpu().numpy()
                    y_pred_unscaled = norm.inverse_transform(np.column_stack([y_pred, np.zeros((len(y_pred), 3))]))[:, 0]
                    y_actual_unscaled = norm.inverse_transform(np.column_stack([y_actual, np.zeros((len(y_actual), 3))]))[:, 0]
        
        
                #store label and results
                
                label = f"H: {HIDDEN_SIZE}, LR: {LEARNING_RATE}, L: {NUM_LAYERS}, M: {name}"
                results[name][label] = {
                    "train_rmse": train_rmse_scores,
                    "pred": y_pred_unscaled,
                    "actual": y_actual_unscaled,
                    "config": {
                        "seq_len": SEQ_LEN,
                        "batch_size": BATCH_SIZE,
                        "hidden_size": HIDDEN_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "num_layers": NUM_LAYERS,
                        "model_name": name
                    }
                }
        
############################################################################

#plot all

fig, axs = plt.subplots(2, 3, figsize=(18, 10))

for idx, (name, model_results) in enumerate(results.items()):
    ax1 = axs[0, idx]
    ax2 = axs[1, idx]

    for label, res in model_results.items():
        #plot predictions
        ax1.plot(dates.values, res["pred"], alpha=0.7)
        ax2.plot(res["train_rmse"])    

    #plot actuals 
    actual = next(iter(model_results.values()))["actual"]
    ax1.plot(dates.values, actual, label="Actual", color="black", linewidth=2)
    
    ax1.set_title(f"{name}: Predicted vs Actual (All)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.grid(True)

    #training loss
    ax2.set_title(f"{name}: Training RMSE per epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("RMSE")
    ax2.grid(True)

plt.tight_layout()
plt.show()

##########################################################################

#pick and plot best model

fig, axs = plt.subplots(2, 3, figsize=(18, 10))

best_configs = {}

for idx, (name, model_results) in enumerate(results.items()):
    ax1 = axs[0, idx]
    ax2 = axs[1, idx]

    # Find best config by final epoch RMSE
    best_label, best_res = min(model_results.items(), key=lambda item: item[1]["train_rmse"][-1])
    best_rmse = best_res["train_rmse"][-1]
    best_conf = best_res["config"]

    legend_label = f"H: {best_conf['hidden_size']}, LR: {best_conf['learning_rate']}, L: {best_conf['num_layers']}, SL: {best_conf['seq_len']}, BS: {best_conf['batch_size']}, RMSE={best_rmse:.4f}"

    best_configs[name] = {
        "ModelClass": {"GRU": StockGRU, "LSTM": StockLSTM, "RNN": StockRNN}[name],
        "hidden_size": best_conf["hidden_size"],
        "learning_rate": best_conf["learning_rate"],
        "num_layers": best_conf["num_layers"],
        "seq_len": best_conf["seq_len"],
        "batch_size": best_conf["batch_size"],
        "rmse": best_rmse
    }

    ax1.plot(dates.values, best_res["actual"], label="Actual", color="black", linewidth=2)
    ax1.plot(dates.values, best_res["pred"], label=legend_label, alpha=0.8)
    ax1.set_title(f"{name}: Predicted vs Actual (Best Model)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.grid(True)
    ax1.legend(fontsize='small')

    ax2.plot(best_res["train_rmse"], label=legend_label, color="blue")
    ax2.set_title(f"{name}: Training RMSE per epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("RMSE")
    ax2.grid(True)
    ax2.legend(fontsize='small')

plt.tight_layout()
plt.show()

##########################################################################################################################

#retrain each best model config and validate on 2025
for name, conf in best_configs.items():
    
    #re-prep because sequence length might be different
    val_norm = RobustScaler()
    
    dates_all, X_trainval_tensor, y_trainval_tensor, X_val_tensor, y_val_tensor = \
        prepare_data(df, features, conf["seq_len"], val_norm, device, train_year=2024, test_year=2025)
    
    model = conf["ModelClass"](input_size=4, hidden_size=conf["hidden_size"], num_layers=conf["num_layers"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf["learning_rate"]) 
    train_loader = DataLoader(TensorDataset(X_trainval_tensor, y_trainval_tensor), batch_size=conf["batch_size"], shuffle=True, generator=gen)
    
    # Retrain
    train_model(model, train_loader, val_norm, optimizer, loss_fn, MAX_EPOCHS, device)

    # Validate on 2025
    model.eval()
    with torch.no_grad():
        y_pred = model(X_val_tensor.to(device)).squeeze().cpu().numpy()
        y_actual = y_val_tensor.squeeze().cpu().numpy()

        y_pred_unscaled = val_norm.inverse_transform(
            np.column_stack([y_pred, np.zeros((len(y_pred), 3))])
        )[:, 0]
        y_actual_unscaled = val_norm.inverse_transform(
            np.column_stack([y_actual, np.zeros((len(y_actual), 3))])
        )[:, 0]

        val_rmse = root_mean_squared_error(y_actual_unscaled, y_pred_unscaled)

        results[name]["final_val_rmse_2025"] = val_rmse
        results[name]["pred_2025"] = y_pred_unscaled
        results[name]["actual_2025"] = y_actual_unscaled
        

########################################################################################

plt.figure(figsize=(12, 6))

# Plot actual values
plt.plot(dates_all.values, results["GRU"]["actual_2025"], label="Actual 2025", color="black", linewidth=2)

# Add model predictions with hyperparams in legend
for name, color in zip(["GRU", "LSTM", "RNN"], ["blue", "green", "red"]):
    conf = best_configs[name]
    rmse = results[name]["final_val_rmse_2025"]

    H = conf["hidden_size"]
    LR = conf["learning_rate"]
    L = conf["num_layers"]
    
    legend_label = (
    f"{name}: H={conf['hidden_size']}, LR={conf['learning_rate']}, "
    f"L={conf['num_layers']}, SL={conf['seq_len']}, BS={conf['batch_size']}, "
    f"RMSE={rmse:.2f}"
)
    
    plt.plot(
        dates_all.values,
        results[name]["pred_2025"],
        label=legend_label,
        color=color,
        alpha=0.8
    )

plt.title("Final Validation: Predictions vs Actual (2025)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

end_time = time.time()
elapsed = end_time - start_time
print(f"\nTotal Runtime: {elapsed // 60:.0f} minutes {elapsed % 60:.2f} seconds")
