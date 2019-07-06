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
  for (let i = 0; i < 100000; i++) {
    const x = rnorm()
    optimizer.fit([x, rnorm(), rnorm()], x < 0 ? 1 : 0)
  }
  t.true(optimizer.predict([1, 0, 3]) < 0.2)
  t.true(optimizer.predict([-1, 0, 3]) > 0.8)
  t.end()
})