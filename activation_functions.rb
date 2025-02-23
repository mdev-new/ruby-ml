ActivationFunctions = {

  :Linear => ->(x) { x },

  :Tanh => ->(x) { Math.tanh(x) },

  :ReLU => ->(x) { [0, x].max },

  :LeakyReLU => ->(layer) { layer.map { |x| x < 0 ? 0.1 * x : x } },

  :ELU => ->(layer) { layer.map { |x| x < 0 ? Math.exp(x) - 1 : x } },

  :Sigmoid => ->(x) { 1 / (1 + Math.exp(-x)) },

  :Softmax => ->(layer) {
    exp = layer.map { |x| Math.exp(x) }
    sum = exp.sum
    exp.map { |x| x / sum }
  },

  # Derivatives
  :dLinear => ->(x) { 1 },
  :dReLU => ->(x) { (x >= 0) ? 1 : 0 },
  :dTanh => ->(x) { 1 - Math.tanh(x) ** 2 },
  :dELU => ->(x) { x < 0 ? Math.exp(x) : 1 },
  :dSoftmax => ->(x) { x * (1 - x) },
  :dLeakyReLU => ->(x) { x < 0 ? 0.1 : 1 },
  :dSigmoid => ->(x) {
    sigmoid = 1 / (1 + Math.exp(-x))
    sigmoid * (1 - sigmoid)
  }

}
