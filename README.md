# Repository for machine learning methods

Methods mostly learned through the [FYSSTK3150/4150 course](https://compphysics.github.io/MachineLearning/doc/web/course.html) and some [INF4820(now called INF4080)](https://www.uio.no/studier/emner/matnat/ifi/IN4080/index.html). 

Goal is to use this repository as way of refreshing and teaching myself new methods.


#### Program folder structure
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
- [x] Derive MLP backprop-algorithms properly and write pdf for it.
- [x] Jit math_tools.py
- [x] Jit logistic_regression.py
- [x] Implemented optimizers in logistic regression(far inferior to the default one).
- [ ] Parallelize(and jit) Cross-Validation, cross_validation.py
- [ ] Jit the MLP, neuralnetwork.py.
- [ ] Vectorize the MLP, neuralnetwork.py.
- [ ] Parallelize the MLP, neuralnetwork.py


#### Future/aspirational work:
My goal is to implement the more machine learning and statistical methods in order to learn them myself. Following methods with examples will be implemented:
- [ ] kNN classification
- [ ] Rocchio classifier(Centroid classifier)
- [ ] BoW models
- [ ] Hidden Markov Models(Viterbi algorithm)
- [ ] Boltzmann machines
- [ ] Decision trees
- [ ] Bayesian theory
- [ ] Support Vector Machines(SVM)
- [ ] Genetic algorithms
- [ ] Blocking
- [ ] Jackknifing
- [ ] Convolutional Neural Networks (CNN)