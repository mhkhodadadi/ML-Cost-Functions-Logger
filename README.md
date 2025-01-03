# Machine Learning - Cost Functions Logger  
### Methods for Training Logistic Regression: Internal vs. External Loops

In this Code, we will train a logistic regression model on a classification dataset using the LogisticRegression algorithm in three different ways:

1. **Internal Iteration Train Loop:**
In this approach, we train the logistic regression model within a loop, updating the model's parameters with each iteration based on the training data. The algorithm uses gradient descent to minimize the loss function step by step until it converges or reaches a set number of iterations.

2. **External Iteration Train Loop:**
In this method, we set the model with `max_iter = 1 and warm_start = True`, which allows the model to train once internally and keep the results for the next cycles. We then control the training process externally, enabling us to adjust the parameters and log the cost functions throughout the process. We do this in two ways:

- **Without using the `class_weight` parameter**.
- **With the `class_weight` parameter**, applying sample weights.

Finally, use the **FuncAnimation** class from **matplotlib.animation** to create an animated plot of the cost function.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen) ![License](https://img.shields.io/badge/License-MIT-blue)

