# ASL_recognizer

## What is all about

* Build a sign language recognizer. 
* Input was coordinates of hand motion. 
* Built features based on cartesian and polar coordinate systems.
* Tried various model selectors:
  * Log likelihood using cross-validation folds (CV)
  * Bayesian Information Criterion (BIC)
  * Discriminative Information Criterion (DIC)
* Trained HMM models

## Results 

WER rate was varying between 48 to 63 depending on the set of features and the model selection method.

## Adding NLP model to improve WER

In the optional Part 4, designed and implemented a hybrid HMM/NLP n-gram model. The final WER was 21% for a test set.

# Contributions

This was a learning project, not a development. 

# License

This was a part of Udacity course, act accordingly. Reminding: plagiarism not recommended. But look and learn if find it useful. 
