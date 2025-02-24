ActivationFunctions = {

  :Tanh => ->(layer) { layer.map { |x| Math.tanh(x) } },

  :ReLU => ->(layer) { layer.map { |x| [0, x].max } },

  :Sigmoid => ->(layer) { layer.map { |x| 1 / (1 + Math.exp(-x)) } },

  :Softmax => ->(layer) do
    max = layer.max
    exps = layer.map { |x| Math.exp(x - max) }
    sum = exps.sum
    exps.map { |exp| exp / sum }
  end,

  # Derivatives
  :dReLU => ->(layer) { layer.map { |x| x >= 0 ? 1 : 0 } },
  
  :dTanh => ->(layer) { layer.map { |x| 1 - Math.tanh(x) ** 2 } },
  
  :dSigmoid => ->(layer) {
    layer.map do |x|
      sigmoid = 1 / (1 + Math.exp(-x))
      sigmoid * (1 - sigmoid)
    end
  },

  # :dSoftmax => ->(layer) { layer.map { |e| e*(1-e) } }

  :dSoftmax => ->(layer) do
    n = layer.size
    jacobian = Array.new(n) { Array.new(n, 0.0) }
    layer.each_with_index do |s, i|
      layer.each_with_index do |s_j, j|
        jacobian[i][j] = (i == j ? s * (1 - s) : -s * s_j)
      end
    end
    jacobian
  end

}
