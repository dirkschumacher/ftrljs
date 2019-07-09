"use strict"

const test = require("tape")
const ftrl = require(".")
const rand = require("prob.js")

test("fit small data", (t) => {
  const optimizer = ftrl(3)
  t.equal(optimizer.weights()[0], 0)
  t.equal(optimizer.weights()[1], 0)
  t.equal(optimizer.weights()[2], 0)
  optimizer.fit([1, 2, 3], 1)
  t.equal(optimizer.weights()[0], 0.03333333333333333)
  t.equal(optimizer.weights()[1], 0.05)
  t.equal(optimizer.weights()[2], 0.06)
  t.end()
})

test("fit random data", (t) => {
  // here we learn if the first component of the training vector is negative
  // the other two components are irrelevant
  const rnorm = rand.normal(0, 1)
  const optimizer = ftrl(3, 10, 1)
  for (let i = 0; i < 10000; i++) {
    const x = rnorm()
    optimizer.fit([x, rnorm(), rnorm()], x < 0 ? 1 : 0)
  }
  t.true(optimizer.predict([1, 0, 3]) < 0.2)
  t.true(optimizer.predict([-1, 0, 3]) > 0.8)
  t.end()
})

test("save/load model", (t) => {
  const optimizer = ftrl(3, 1, 2, 0.5, 2)
  optimizer.fit([1, 2, 3], 1)
  const weights = optimizer.weights()
  const saved = optimizer.save()
  const newOptimizer = ftrl(10, 20, 30, 40, 50)
  newOptimizer.load(saved)
  const newWeights = newOptimizer.weights()
  for (let i = 0; i < 3; i++) {
    t.equal(weights[i], newWeights[i])
  }
  t.equal(optimizer.predict([1, 2, 3]), newOptimizer.predict([1, 2, 3]))
  const newSaved = newOptimizer.save()
  t.equal(newSaved.config.lambda1, saved.config.lambda1)
  t.equal(newSaved.config.lambda2, saved.config.lambda2)
  t.equal(newSaved.config.alpha, saved.config.alpha)
  t.equal(newSaved.config.beta, saved.config.beta)
  t.equal(newSaved.config.n_features, saved.config.n_features)
  for (let i = 0; i < 3; i++) {
    t.equal(newSaved.z[i], saved.z[i])
    t.equal(newSaved.n[i], saved.n[i])
    t.equal(newSaved.lastWeights[i], saved.lastWeights[i])
  }
  t.end()
})