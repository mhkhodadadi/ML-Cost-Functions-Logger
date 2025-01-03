import numpy as np
import matplotlib.pyplot as plt
import mglearn

from matplotlib.animation import FuncAnimation
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

import warnings
# Suppress convergence warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Generate Dataset and Split into Training and Test Pairs
X, y = make_blobs(n_samples=500, n_features=2, centers=2, cluster_std=1.0, random_state=0)

plt.figure(figsize=(9, 3))
plt.subplot(121)
plt.title("Histogram of Data with stratify=None")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)
plt.hist(y_train, bins=3)
plt.hist(y_test, bins=3)


plt.subplot(122)
plt.title("Histogram of Data with stratify=Y")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
plt.hist(y_train, bins=3)
plt.hist(y_test, bins=3)
plt.show()

classes = np.unique(y)
feature_number = X.shape[1]
print(f'Number of Classes: {len(classes)}, Number of features: {feature_number}')

# Plotting Decision Boundaries and Class Separation for Logistic Regression
iterations = 200
model_internal_loop = LogisticRegression(solver='saga', max_iter=iterations)
model_internal_loop.fit(X_train, y_train)

mglearn.plots.plot_2d_separator(model_internal_loop, X, fill=False, eps=0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1"])
plt.show()

# Train Models Externally and measure scores
# Function to update weights for misclassified samples
def update_weights(model, X, y):
    y_pred = model.predict(X)
    y0_inc = np.where((y == 0) & (y_pred != y), 2, 1)
    y1_inc = np.where((y == 1) & (y_pred != y), 2, 1)
    sample_weights = np.where((y_pred != y), 2, 1)
    class_weights = {0: np.mean(y0_inc), 1: np.mean(y1_inc)}
    return class_weights, sample_weights


# Defining and Training models into externally loop
model_external_loop = LogisticRegression(solver='saga', max_iter=1, warm_start=True)
model_external_loop_weighted = LogisticRegression(solver='saga', max_iter=1, warm_start=True, class_weight=None)
model_external_loop_weighted.fit(X_train, y_train)

costs_normal = []
costs_weighted = []
for i in range(iterations):
    # Training normal model
    model_external_loop.fit(X_train, y_train)
    y_pred_prob_normal = model_external_loop.predict_proba(X_train)
    cost_normal = log_loss(y_train, y_pred_prob_normal)
    costs_normal.append(cost_normal)

    # Training weighted model
    class_weights, sample_weights = update_weights(model_external_loop_weighted, X_train, y_train)

    model_external_loop_weighted.set_params(class_weight=class_weights)
    model_external_loop_weighted.fit(X_train, y_train, sample_weight=sample_weights)

    # Probability estimates
    y_pred_prob_weighted = model_external_loop_weighted.predict_proba(X_train)
    # Log loss, aka logistic loss or cross-entropy loss.
    cost_weighted = log_loss(y_train, y_pred_prob_weighted)
    costs_weighted.append(cost_weighted)

print(f'Costs of model          (with internal training iteration): {model_internal_loop.score(X_test, y_test)}')
print(f'Costs of model          (with external training iteration): {model_external_loop.score(X_test, y_test)}')
print(f'Costs of weighted model (with external training iteration): {model_external_loop_weighted.score(X_test, y_test)}')

# Animated Plot: Comparing Cost Functions of External Training Models
# Initializing plots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].set_title("Normal Training")
axes[1].set_title("Weighted Training")

ylim = np.max(costs_normal)
ylim = np.max([ylim, np.max(costs_weighted)])
for ax in axes:
    ax.set_xlim(0, iterations)
    ax.set_ylim(0, ylim * 2)
    ax.set_xlabel("iterations")
    ax.set_ylabel("Cost")
    ax.grid()

lines_normal, = axes[0].plot([], [], color='blue')
lines_weighted, = axes[1].plot([], [], color='orange')

iter = 0
cycles = 1
def fun_animation(frame):
    global iter
    iter += 1
    # Exit condition after specified cycles
    if iter >= iterations * cycles:  # Stop after the permitted cycles
        # plt.close(fig)  # Closes the plot window
        return lines_normal, lines_weighted
    # Update the lines with new data
    lines_normal.set_data(range(frame + 1), costs_normal[:frame + 1])
    lines_weighted.set_data(range(frame + 1), costs_weighted[:frame + 1])
    return lines_normal, lines_weighted


# Running animation
ani = FuncAnimation(fig, fun_animation, frames=iterations, interval=20, blit=True)
plt.tight_layout()
plt.show()