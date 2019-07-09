"use strict"

const zeros = (n) => {
  let val = Array(n)
  for (let i = 0; i < n; i++) {
    val[i] = 0
  }
  return val
}

const inverseLogit = (x) => {
  return 1.0 / (1.0 + Math.exp(-x))
}

// Reference
// McMahan, H. Brendan, et al. “Ad click prediction: a view from the
// trenches.” Proceedings of the 19th ACM SIGKDD international conference
// on Knowledge discovery and data mining. ACM, 2013.
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
    return inverseLogit(accum)
  }

  const computeWeights = () => {
    for (let i = 0; i < n_features; i++) {
      const updateCoordinate = Math.abs(z[i]) > lambda1
      if (updateCoordinate) {
        let w = z[i] - Math.sign(z[i]) * lambda1
        w = w / -((beta + Math.sqrt(n[i])) / alpha + lambda2)
        lastWeights[i] = w
      } else {
        lastWeights[i] = 0
      }
    }
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

  const save = () => {
    return {
      config: {
        lambda1,
        lambda2,
        alpha,
        beta,
        n_features
      },
      n,
      z,
      lastWeights
    }
  }

  const load = (savedModel) => {
    lambda1 = savedModel.config.lambda1
    lambda2 = savedModel.config.lambda2
    alpha = savedModel.config.alpha
    beta = savedModel.config.beta
    n_features = savedModel.config.n_features
    n = savedModel.n
    z = savedModel.z
    lastWeights = savedModel.lastWeights
  }

  return {
    fit,
    predict,
    "weights": computeWeights,
    save,
    load
  }
}

module.exports = ftrl