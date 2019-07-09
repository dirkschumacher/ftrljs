# Dense FTRL-Proximal online learning algorithm for logistic regression

An _experimental_ implementation of the FTRL online learning algorithm for logistic regression in plain javascript.

Implemented as in this
[paper](https://www.eecs.tufts.edu/%7Edsculley/papers/ad-click-prediction.pdf).


```js
const optimizer = ftrl(
  3, // the number of features in your training examples 
  10, // lambda1 - use this parameter for l1 regularitzation
  2, // lambda2 - use this parameter for l2 regularitzation
  0.1, // alpha - parameters that control the learning rate (see Eq. 2)
  1 // beta - parameters that control the learning rate (see Eq. 2). 1 seems to be a good default value here
)

// pass one training example (an array) and the result (a number) at a time
optimizer.fit([1, 2, 3], 1)
optimizer.fit([1, 4, 3], 1)
optimizer.fit([1, 7, 3], 0)

// you can then predict the result for unseen examples
optimizer.predict([2, 3, 4])

// or just get the weights
optimizer.weights()

// ore store everything for later
const checkpoint = optimizer.save()
// ...
const newOptimizer = ftrl()
newOptimizer.load(checkpoint)
```

## API

* `ftrl()` creates a new optimizer/model
* `fit()` fits a single training example (array + result)
* `predict()` takes an array and returns the prediction (number between 0 and 1)
* `weights()` computes and returns the weights
* `save()` export the current state of the optimizer to an object
* `load()` restore a optimizer from a previously saved state


## Memory

The algorithm stores three dense arrays in the size of the features. If your number of features is too large, a sparse implementation of the algorithm can be used.

## References

McMahan, H. Brendan, et al. “Ad click prediction: a view from the
trenches.” Proceedings of the 19th ACM SIGKDD international conference
on Knowledge discovery and data mining. ACM, 2013.
