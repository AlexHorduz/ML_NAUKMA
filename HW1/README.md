# Homework 1

You are given two Python files with either blank spaces that need to be filled in or bullet points outlining what needs to be implemented.

## Part 1:
- Implement your own Linear Regression model class:
    - Initialize the weights
    - Verbose training process: log training statistics (e.g., cost)
    - Implement the `fit` method
    - Implement the `predict` method
    - The plot of your model's performance should closely resemble that of Sklearn. If it does not, try tuning the hyperparameters (don't hesitate to set `num_iterations` to 1,000,000).
- Write a function for the Normal Equation: estimate the weights without training the model.

## Part 2:
You are given a small subset of the Flickr30k dataset.
1. Load the data.
2. Convert the images of shape (3, H, W) into N-dimensional vectors and explain how you did it. A few ideas include:
    - Convert to grayscale and flatten.
    - Take the mean across the RGB channels and flatten.
    - Use any pre-trained embedding model (e.g., CLIP, SigLIP, etc.).
    - Any other method you find suitable.
3. Convert the text labels to integers: 0 for humans, 1 for animals.
4. Create a train/test split. Use a test size of 0.2 to evaluate your final model.
5. Train the following models:
    - LogisticRegression (implement)
    - KNN
    - DecisionTree
6. Train the model using the following validation strategies:
    - **Simple train/test split**: Further divide the training data into training and validation subsets for small-scale hyperparameter optimization based on the validation set's performance.
    - **K-fold validation**: Train a separate model on each fold and evaluate the model using K-fold cross-validation.
    - **Stratified K-fold**: Similar to K-fold validation but ensures that class proportions are maintained across the splits.
7. Make predictions on the test set and measure accuracy.
    - For the simple train/test split, you will have only one model, so no additional modifications are needed.
    - In the K-fold case, you will have K models, so you can experiment with methods like majority voting, absolute voting, or average probability aggregation with a threshold.
8. Perform error analysis:
    - Examine the samples where the model predicted the wrong label and provide an explanation for why you think it happened.
    - Visualize the confusion matrix, showing counts of correctly classified classes and misclassified ones.
    - Try to improve your model: data cleaning, hyperparameters, model choice, etc.
