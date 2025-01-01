# ML-Cost-Functions-Logger  
## Logistic Regression Costs Visualization  

This project demonstrates the training of logistic regression models using two different approaches: normal training and weighted training. It visualizes the costs associated with each model over a series of iterations using matplotlib animations.  

## Project Explanation  

In machine learning, training models effectively is crucial for accuracy and performance. This project addresses the challenges of misclassifications in logistic regression by introducing a weighted training approach. The experiment compares two methods of model training:  

1. **Normal Logistic Regression**: This model is trained without any adjustments to class weights, serving as a baseline for comparison.  
  
2. **Weighted Logistic Regression**: In this approach, we increase the weight of misclassified samples during training. This aims to give more importance to difficult-to-classify examples, potentially improving the model's ability to capture complex decision boundaries.  

The project generates synthetic data using `make_blobs` and trains both models over 200 iterations, capturing the log loss (cost) at each step. The resulting costs are visualized in real-time through animated plots, allowing for an intuitive understanding of how costs decrease as training progresses in both scenarios.  

## Table of Contents  

- [Installation](#installation)  
- [Project Structure](#project-structure)  
- [Results](#results)  
- [License](#license)  

## Installation  

To run this project, ensure you have Python and the necessary libraries installed. You can install the required libraries using pip:  

```bash  
pip install numpy matplotlib scikit-learn
```
## Project Structure
ML-Cost-Functions-Logger.py: The main Python script containing the implementation of logistic regression training and visualization. In this sample, a LogisticRegression model is trained with max_iter=1 and warm_start=True, continuing training for a specified number of iterations both without class_weight and with class_weight. The costs are recorded for animation plots.
* **Model Initialization**
```python
model_normal = LogisticRegression(solver='saga', max_iter=1, warm_start=True)  
model_weighted = LogisticRegression(solver='saga', max_iter=1, warm_start=True, class_weight=None)
```
Training Loop
```python
for i in range(iterations):  
    # Training normal model  
    model_normal.fit(X_train, y_train)  
    y_pred_prob_normal = model_normal.predict_proba(X_train)  
    cost_normal = log_loss(y_train, y_pred_prob_normal)  
    costs_normal.append(cost_normal)  

    # Training weighted model  
    if i == 0:  
        model_weighted.fit(X_train, y_train)  
    class_weight = update_weights(model_weighted, X_train, y_train, weights)  
    model_weighted.set_params(class_weight=class_weight)  
    model_weighted.fit(X_train, y_train)  
    y_pred_prob_weighted = model_weighted.predict_proba(X_train)  
    cost_weighted = log_loss(y_train, y_pred_prob_weighted)  
    costs_weighted.append(cost_weighted)
```
## Results

After training, the scores for the three models are measured:

```python
model_Base = LogisticRegression(solver='saga', max_iter=iterations)  
model_Base.fit(X_train, y_train)  

print(f'Costs of model          (with internal train iteration): {model_Base.score(X_test, y_test)}')  
print(f'Costs of normal model   (with external train iteration): {model_normal.score(X_test, y_test)}')  
print(f'Costs of weighted model (with external train iteration): {model_weighted.score(X_test, y_test)}')  
```
Then, animation plots are generated:

- For Python
```python
import matplotlib.pyplot as plt  
from matplotlib.animation import FuncAnimation  
...  
# Running animation  
ani = FuncAnimation(fig, fun_update, frames=iterations, interval=20, blit=True)  
plt.tight_layout()  
plt.show()
```  
- For Jupyter Notebook
```python
import matplotlib.pyplot as plt  
from matplotlib.animation import FuncAnimation  
from IPython.display import HTML  
...  
ani = FuncAnimation(fig, fun_update, frames=iterations, interval=20, blit=True)  
plt.tight_layout()  
display(HTML(ani.to_jshtml()))  
plt.close()  
```
## License
This project is licensed under the MIT License. See the LICENSE file for more details.
![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen) ![License](https://img.shields.io/badge/License-MIT-blue)

