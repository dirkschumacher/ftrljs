"use strict"

const zeros = (n) => {
  let val = []
  for (let i = 0; i < n; i++) {
    val.push(0)
  }
  return val
}

const ftrl = (n_features, lambda1 = 0, lambda2 = 0, alpha = 0.1, beta = 1) => {
  let z = zeros(n_features)
  let n = zeros(n_features)
  let lastWeights = zeros(n_features)

  const gradient = (p, y, x) => {
    let result = []
    for (let i = 0; i < n_features; i++) {
      result.push((p - y) * x[i])
    }
    return result
  }

  const predict = (x) => {
    let accum = 0
    for (let i = 0; i < n_features; i++) {
      accum += lastWeights[i] * x[i]
    }
    return 1.0 / (1.0 + Math.exp(-1.0 * accum))
  }

  const computeWeights = () => {
    let newWeights = zeros(n_features)
    for (let i = 0; i < n_features; i++) {
      const updateCoordinate = Math.abs(z[i]) > lambda1
      if (updateCoordinate) {
        let w = z[i] - Math.sign(z[i]) * lambda1
        w = w / -((beta + Math.sqrt(n[i])) / alpha + lambda2)
        newWeights[i] = w
      }
    }
    lastWeights = newWeights
    return lastWeights
  }

  const fit = (x, y) => {
    computeWeights()
    const p = predict(x)
    const g = gradient(p, y, x)

    for (let i = 0; i < n_features; i++) {
      const nn = n[i] + Math.pow(g[i], 2.0)
      const s = (Math.sqrt(nn) - Math.sqrt(n[i])) / alpha
      z[i] = z[i] + g[i] - s * lastWeights[i]
      n[i] = nn
    }
  }

  return {
    fit,
    predict,
    "weights": computeWeights
  }
}

module.exports = ftrl