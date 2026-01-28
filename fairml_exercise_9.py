import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sklearn as skl
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# PART A

# prepare data
df = pd.read_csv("data.csv")
x1 = df['x1']
x2 = df['x2']
y = df['outcome']
groups = df['group']

# set up model
model = LogisticRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# train and test model
def train_and_test(X, y):
    accuracies = []
    disparities = []

    print(f"{'Fold':<10} | {'Accuracy':<12} | {'Demographic Disparity':<20}")
    print("-" * 50)

    for i, (train_index, val_index) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        group_val = groups.iloc[val_index]
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        
        # Calculate Accuracy
        acc = accuracy_score(y_val, y_pred)
        accuracies.append(acc)
        
        prob_g0 = y_pred[group_val == 0].mean()
        prob_g1 = y_pred[group_val == 1].mean()
        
        disparity = abs(prob_g0 - prob_g1)
        disparities.append(disparity)
        
        print(f"{i+1:<10} | {acc:<12.4f} | {disparity:<20.4f}")

    print("-" * 40)
    print(f"{'AVERAGE':<10} | {np.mean(accuracies):<12.4f} | {np.mean(disparities):<20.4f}")


train_and_test(pd.concat([x1, x2], axis=1), y)

print("=" * 50)
# PART B
corr_x1 = df['x1'].corr(df['group'])
corr_x2 = df['x2'].corr(df['group'])
print(f"Correlation between x1 and group: {corr_x1}")
print(f"Correlation between x2 and group: {corr_x2}")

train_and_test(x1.to_frame(), y)
print("=" * 50)

#PART C
# discontinuities when on decision boundary?
# sigmoid function on disparity term
# \frac{1}{1+e^{-x}}


# PART D





#minimize Logistic Regression loss function (log loss) and add disparity term
#output weights 

def sigmoid(x, reg=1):
    return reg * (1 / (1 + np.exp(-x)))

def fairness_loss(model, X, y_true, g_train, lam):
    logits = model(X).squeeze()
    bce = nn.BCEWithLogitsLoss()
    loss = bce(logits, y_true)
    
    y_pred = (torch.sigmoid(logits) >= 0.5).float()

    p_pos_g0 = y_pred[g_train == 0].mean()
    p_pos_g1 = y_pred[g_train == 1].mean()
    disparity = torch.abs(p_pos_g0 - p_pos_g1)
    
    return loss + sigmoid(disparity, lam)

def train_model(model, x_train, y_train, group_train, lam=1.0, lr=0.01, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = fairness_loss(model, x_train, y_train, group_train, lam)
        loss.backward()
        optimizer.step()
    return model

def train_pytorch(input, lam):
    accuracies = []
    disparities = []

    print(f"{'Fold':<10} | {'Accuracy':<12} | {'Demographic Disparity':<20}")
    print("-" * 50)
    
    X_tensor = torch.tensor(input.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    groups_tensor = torch.tensor(groups.values, dtype=torch.int64)

    for i, (train_idx, val_idx) in enumerate(kf.split(input)):
        X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
        y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
        group_train, group_val = groups_tensor[train_idx], groups_tensor[val_idx]
        
        model = nn.Linear(X_train.shape[1], 1)

        model = train_model(model, X_tensor[train_idx], y_train, group_train, lam=lam)
        
        with torch.no_grad():
            logits_val = model(X_val).squeeze()
            probs_val = torch.sigmoid(logits_val)
            y_pred = (probs_val >= 0.5).float()
            
            acc = (y_pred == y_val).float().mean().item()
            prob_g0 = y_pred[group_val == 0].mean().item()
            prob_g1 = y_pred[group_val == 1].mean().item()
            disparity = abs(prob_g0 - prob_g1)

            accuracies.append(acc)
            disparities.append(disparity)
            print(f"{i+1:<10} | {acc:<12.4f} | {disparity:<20.4f}")
    mean_acc = np.mean(accuracies)
    mean_disp = np.mean(disparities)
    print("-" * 40)
    print(f"{'AVERAGE':<10} | {np.mean(accuracies):<12.4f} | {np.mean(disparities):<20.4f}")
    return mean_acc, mean_disp
    
# Running part D for plot
accuracies_list = []
disparities_list = []
regs = np.arange(0.0, 1.05, 0.05)

for reg in regs:
    acc, disp = train_pytorch(pd.concat([x1, x2], axis=1), reg)
    if reg == 0:
        print("Disparity 0: accuracy is", disp)
    accuracies_list.append(acc)
    disparities_list.append(disp)

plt.plot(disparities_list, accuracies_list)
plt.xlabel('Demographic Disparity')
plt.ylabel('Accuracy')
plt.title('Accuracy-Disparity Tradeoff')
plt.grid(True)
plt.show()

# PART E
#train_pytorch(pd.concat([x1, x2]), 1)
