# Repository for machine learning methods

Methods mostly learned through the [FYSSTK3150/4150 course](https://compphysics.github.io/MachineLearning/doc/web/course.html).


#### Folder structure
```
lib/
|--- neuralnetwork.py
|--- logistic_regression.py
|--- regression.py
|--- resampling/
     |--- bootstrap.py
     |--- cross_validation.py
|--- utils/
     |--- metrics.py
     |--- optimizers.py
     |--- math_tools.py
```


#### TODO-list:
Current todo-list before starting on goals(not necessarily in that order):
[ ] Derive MLP backprop-algorithms properly and write pdf for it.
[ ] Parallelize(and jit) Cross-Validation, cross_validation.py
[ ] Jit optimize.py
[ ] Jit metrics.py
[ ] Jit math_tools.py
[ ] Jit logistic_regression.py
[ ] Jit the MLP
[ ] Parallelize the MLP, neuralnetwork.py
[ ] Decide on if I am going to implement optimizers in logistic regression.


#### Future work:
My goal is to implement the more machine learning and statistical methods in order to learn them myself. Following methods with examples will be implemented:
[ ] kNN classification
[ ] Rocchio classifier(Centroid classifier)
[ ] BoW models
[ ] Hidden Markov Models(Viterbi algorithm)
[ ] Boltzmann machines
[ ] Decision trees
[ ] Bayesian theory
[ ] Support Vector Machines(SVM)
[ ] Genetic algorithms
[ ] Blocking
[ ] Jackknifing
[ ] Convolutional Neural Networks (CNN)