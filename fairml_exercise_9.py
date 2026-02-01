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
    # print("-" * 50)

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
        
        # print(f"{i+1:<10} | {acc:<12.4f} | {disparity:<20.4f}")

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

def calc_disparity(X_val, y_val, group_val, model):
    weights_list = []
    with torch.no_grad():
        logits_val = model(X_val).squeeze()
        probs_val = torch.sigmoid(logits_val)
        y_pred = (probs_val >= 0.5).float()
        
        acc = (y_pred == y_val).float().mean().item()
        prob_g0 = y_pred[group_val == 0].mean().item()
        prob_g1 = y_pred[group_val == 1].mean().item()
        disparity = abs(prob_g0 - prob_g1)
        w = model.weight.squeeze().cpu().numpy()
        weights_list.append(w)
    # print(f"{i+1:<10} | {acc:<12.4f} | {disparity:<20.4f}")
    return acc, disparity, weights_list
    

# PART D

#minimize Logistic Regression loss function (log loss) and add disparity term
#output weights 

def fairness_loss(model, X, y_true, g_train, lam):
    logits = model(X).squeeze()
    bce = nn.BCEWithLogitsLoss()
    loss = bce(logits, y_true)
    
    y_pred = torch.sigmoid(logits)
    p_pos_g0 = y_pred[g_train == 0].mean()
    p_pos_g1 = y_pred[g_train == 1].mean()
    disparity = (p_pos_g0 - p_pos_g1)**2
    
    return loss + lam * disparity

def train_model(model, x_train, y_train, group_train, lam=1.0, lr=0.01, epochs=200):
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
    all_weights = []

    # print(f"{'Fold':<10} | {'Accuracy':<12} | {'Demographic Disparity':<20}")
    # print("-" * 50)
    
    X_tensor = torch.tensor(input.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    groups_tensor = torch.tensor(groups.values, dtype=torch.int64)
    for i, (train_idx, val_idx) in enumerate(kf.split(input)):
        X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
        y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
        group_train, group_val = groups_tensor[train_idx], groups_tensor[val_idx]
        
        model = nn.Linear(X_train.shape[1], 1)

        model = train_model(model, X_train, y_train, group_train, lam=lam)
        
        accuracy, disparity, weights = calc_disparity(X_val, y_val, group_val, model)
        accuracies.append(accuracy)
        disparities.append(disparity)
        all_weights.append(weights)

    all_weights = np.stack(all_weights, axis=0)
    mean_weights = np.mean(all_weights, axis=0)
    mean_acc = np.mean(accuracies)
    mean_disp = np.mean(disparities)
    #print("-" * 40)
    #print(f"{'AVERAGE':<10} | {np.mean(accuracies):<12.4f} | {np.mean(disparities):<20.4f}")
    return mean_acc, mean_disp, mean_weights
    
# Running part D for plot
def run_plot(X, title):
    accuracies_list = []
    disparities_list = []
    weights_list = []
    regs = [0, 0.1, 0.5, 1, 5, 10, 20, 50]
    
    for reg in regs:
        acc, disp, w = train_pytorch(X, reg)
        if reg == 0:
            print("When disparity strength is 0: Disparity: ", disp, "  Accuracy: ", acc)
        accuracies_list.append(acc)
        disparities_list.append(disp)
        weights_list.append(w)
    weights_array = np.stack(weights_list, axis=0)
    plt.scatter(disparities_list, accuracies_list)
    plt.xlabel('Demographic Disparity')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.grid(True)
    plt.savefig(title)
    plt.figure()
    for j in range(weights_array.shape[1]):
        plt.plot(
            regs,
            weights_array[:, j],
            marker='o',
            label=f"Feature {j}"
        )

    plt.xscale("log")
    plt.xlabel("regularization λ")
    plt.ylabel("Average weight value")
    plt.title("Feature weights vs disparity regularization")
    plt.legend()
    plt.grid(True)
    plt.show()

    return accuracies_list, disparities_list

run_plot(pd.concat([x1, x2], axis=1), "Accuracy-Disparity Tradeoff (x1, x2)")
         
# PART E
run_plot(pd.concat([x1, x2, groups], axis=1), "Accuracy-Disparity Tradeoff (x1, x2, group)")


"""
PART F

In parts (D) and (E), how do the coefficients of x1, x2, and group change as the strength 
of the disparity term in the loss function increases? Give an intuitive explanation for why 
the coefficients change the way they do.

# TODO

PART G

Comparing your accuracy-vs-disparity curves in parts (D) and (E), which option gives a 
better tradeoff: using group or not using group?

Using group gives a better tradeoff as seen by the graph which has more points with 
higher accuracy for lower disparity. Without group the model can only reduce disparity 
by becoming less sensitive to whatever correlates with group but with group the model 
can reduce disparity more directly and effectively, allowing us to keep a higher accuracy
at the same disparity level. 


PART H

Describe a decision-making scenario that might have led to this toy problem. Specifically, 
state what the outcome, x1, x2, and group variables are. In this scenario, describe which 
classifier (if any) would you use and what factors would you consider in making your choice

Let's look at the problem of whether to give a criminal pretrial bail or to not give them bail. 
The outcome of this would be whether the person appears in court for their hearing or if they 
fail to appear in court or are rearrested before then which is binary. x1 would be a continuous
factor such as number of prior missed court appearances. x2 could be a continuous measure of 
stability or support such as employment stability calculated using number of years of past
continual employment. Group could be a binary of race, such as 'black' being 1 and 'not black'
being 0. You could use logistic regression classifier with a fairness regularization term to
prevent producing large demographic disparities to get a higher accuracy. We can tune the 
fairness regularized classifier to allow a choice between more accuracy or less demographic
disparity. Factors that we should consider when making this choice is of course accuracy of
predictions since pretrial bail has important consequences on a person's life. We also consider
demographic disparity and the transparency of the algorithm biases to judges in order for
them to make more informed decisions. 


"""
