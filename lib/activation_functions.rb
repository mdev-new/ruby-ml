
module ActivationFunctions
  Celu = ->(layer) {
    alpha = 1.0
    layer.map { |x| x >= 0 ? x : alpha * (Math.exp(x / alpha) - 1) }
  }

  Exponential = ->(layer) {
    layer.map { |x| Math.exp(x) }
  }

  Gelu = ->(layer) {
    # Gaussian Error Linear Unit: x * 0.5 * (1 + erf(x/sqrt(2)))
    layer.map { |x| x * 0.5 * (1 + Math.erf(x / Math.sqrt(2))) }
  }

  Glu = ->(layer) {
    # Gated Linear Unit: splits the input into two halves: A and B, then returns A * sigmoid(B)
    n = layer.size / 2
    a = layer[0...n]
    b = layer[n...layer.size]
    a.zip(b).map { |a_val, b_val| a_val * (1 / (1 + Math.exp(-b_val))) }
  }

  HardShrink = ->(layer) {
    lamb = 0.5
    layer.map { |x| (x > lamb || x < -lamb) ? x : 0 }
  }

  HardSigmoid = ->(layer) {
    # f(x)= min(1, max(0, 0.2*x + 0.5))
    layer.map do |x|
      y = 0.2 * x + 0.5
      y < 0 ? 0 : (y > 1 ? 1 : y)
    end
  }

  HardSilu = ->(layer) {
    # Approximate Hard SiLU as: x * hard_sigmoid(x)
    layer.map do |x|
      y = 0.2 * x + 0.5
      y = y < 0 ? 0 : (y > 1 ? 1 : y)
      x * y
    end
  }

  HardTanh = ->(layer) {
    # Clamp values between -1 and 1
    layer.map { |x| x < -1 ? -1 : (x > 1 ? 1 : x) }
  }

  Linear = ->(layer) { layer }

  LogSigmoid = ->(layer) {
    # log(sigmoid(x)) = -log(1 + exp(-x))
    layer.map { |x| -Math.log(1 + Math.exp(-x)) }
  }

  LogSoftmax = ->(layer) {
    # Numerically stable log softmax
    max = layer.max
    sum_exp = layer.map { |x| Math.exp(x - max) }.sum
    layer.map { |x| x - max - Math.log(sum_exp) }
  }

  Mish = ->(layer) {
    # mish(x)= x * tanh(softplus(x)) where softplus(x)=log(1 + exp(x))
    layer.map { |x| x * Math.tanh(Math.log(1 + Math.exp(x))) }
  }

  ReLU6 = ->(layer) {
    layer.map { |x| [[0, x].max, 6].min }
  }

  Selu = ->(layer) {
    lambda_sel = 1.0507
    alpha_sel = 1.67326
    layer.map { |x| x > 0 ? lambda_sel * x : lambda_sel * alpha_sel * (Math.exp(x) - 1) }
  }

  Silu = ->(layer) {
    # SiLU(x)= x * sigmoid(x)
    layer.map { |x| x / (1 + Math.exp(-x)) }
  }

  SoftShrink = ->(layer) {
    lamb = 0.5
    layer.map do |x|
      if x > lamb
        x - lamb
      elsif x < -lamb
        x + lamb
      else
        0
      end
    end
  }

  Softplus = ->(layer) {
    layer.map { |x| Math.log(1 + Math.exp(x)) }
  }

  Softsign = ->(layer) {
    layer.map { |x| x / (1 + x.abs) }
  }

  SparsePlus = ->(layer) {
    # An example sparsity-inducing activation: zero-out small-magnitude values.
    threshold = 0.5
    layer.map { |x| x.abs >= threshold ? x : 0 }
  }

  Sparsemax = ->(layer) {
    # Implements sparsemax activation (projection onto the simplex)
    z = layer
    sorted_z = z.sort.reverse
    k = 0
    cumulative = 0.0
    sorted_z.each_with_index do |z_val, i|
      cumulative += z_val
      if z_val + (1.0 / (i + 1)) * (1 - cumulative) > 0
        k = i + 1
      end
    end
    tau = (sorted_z.take(k).sum - 1) / k.to_f
    z.map { |x| [x - tau, 0].max }
  }

  Squareplus = ->(layer) {
    # Squareplus: 0.5*(x + sqrt(x^2 + 4))
    layer.map { |x| 0.5 * (x + Math.sqrt(x * x + 4)) }
  }

  TanhShrink = ->(layer) {
    # tanh_shrink: x - tanh(x)
    layer.map { |x| x - Math.tanh(x) }
  }

  Threshold = ->(layer) {
    threshold = 0
    layer.map { |x| x > threshold ? x : 0 }
  }

  Tanh = ->(layer) { layer.map { |x| Math.tanh(x) } }

  LeakyReLU = ->(layer) { layer.map { |x| x < 0 ? 0.1 * x : x } }

  ELU = ->(layer) { layer.map { |x| x < 0 ? Math.exp(x) - 1 : x } }

  ReLU = ->(layer) { layer.map { |x| [0, x].max } }

  Sigmoid = ->(layer) { layer.map { |x| 1 / (1 + Math.exp(-x)) } }

  Softmax = ->(layer) do
    max = layer.max
    exps = layer.map { |x| Math.exp(x - max) }
    sum = exps.sum
    exps.map { |exp| exp / sum }
  end,

  # Derivatives
  ReLU_Derivative = ->(layer) { layer.map { |x| x >= 0 ? 1 : 0 } }
  LeakyReLU_Derivative = ->(layer) { layer.map { |x| x < 0 ? 0.1 : 1 } }
  ELU_Derivative = ->(layer) { layer.map { |x| x < 0 ? Math.exp(x) : 1 } }

  Tanh_Derivative = ->(layer) { layer.map { |x| 1 - Math.tanh(x) ** 2 } }
  
  Sigmoid_Derivative = ->(layer) {
    layer.map do |x|
      sigmoid = 1 / (1 + Math.exp(-x))
      sigmoid * (1 - sigmoid)
    end
  }

  Softmax_Derivative = ->(layer) do
    n = layer.size
    jacobian = Array.new(n) { Array.new(n, 0.0) }
    layer.each_with_index do |s, i|
      layer.each_with_index do |s_j, j|
        jacobian[i][j] = (i == j ? s * (1 - s) : -s * s_j)
      end
    end
    jacobian
  end

  Celu_Derivative = ->(layer) {
    alpha = 1.0
    layer.map { |x| x >= 0 ? 1 : Math.exp(x / alpha) }
  }

  Exponential_Derivative = ->(layer) { 
    # derivative of exp(x) is exp(x)
    layer.map { |x| Math.exp(x) }
  }

  Gelu_Derivative = ->(layer) {
    # approximate derivative of Gelu:
    # 0.5 * (1 + erf(x/√2)) + (x / (Math.sqrt(2*Math::PI)))*Math.exp(-x*x/2)
    layer.map do |x|
      term1 = 0.5 * (1 + Math.erf(x / Math.sqrt(2)))
      term2 = (x / Math.sqrt(2 * Math::PI)) * Math.exp(-0.5 * x * x)
      term1 + term2
    end
  }

  Glu_Derivative = ->(layer) {
    # For Glu, the forward pass splits the input into two halves.
    # Assuming "layer" holds the concatenated [a, b] values,
    # we return an array of two-element arrays representing the partial derivatives:
    n = layer.size / 2
    a = layer[0...n]
    b = layer[n...layer.size]
    sigmoid = ->(z) { 1 / (1 + Math.exp(-z)) }
    a.zip(b).map do |a_val, b_val|
      d_da = sigmoid.call(b_val)
      d_db = a_val * sigmoid.call(b_val) * (1 - sigmoid.call(b_val))
      [d_da, d_db]
    end.flatten  # flatten if you prefer a one-dimensional array (length equals original input size)
  }

  HardShrink_Derivative = ->(layer) {
    lamb = 0.5
    layer.map { |x| (x > lamb || x < -lamb) ? 1 : 0 }
  }

  HardSigmoid_Derivative = ->(layer) {
    # derivative is 0.2 if 0 < (0.2*x+0.5) < 1, otherwise 0.
    layer.map do |x|
      linear = 0.2 * x + 0.5
      (linear > 0 && linear < 1) ? 0.2 : 0
    end
  }

  HardSilu_Derivative = ->(layer) {
    # derivative: hard_silu(x) = x * hard_sigmoid(x)
    # d/dx = hard_sigmoid(x) + x * d/dx[hard_sigmoid(x)]
    layer.map do |x|
      linear = 0.2 * x + 0.5
      hs = (linear < 0 ? 0 : (linear > 1 ? 1 : linear))
      d_hs = (linear > 0 && linear < 1) ? 0.2 : 0
      hs + x * d_hs
    end
  }

  HardTanh_Derivative = ->(layer) {
    # derivative is 1 if x is between -1 and 1, 0 otherwise.
    layer.map { |x| (x > -1 && x < 1) ? 1 : 0 }
  }

  Linear_Derivative = ->(layer) {
    layer.map { 1 }
  }

  LogSigmoid_Derivative = ->(layer) {
    # derivative of log_sigmoid, where f(x) = -log(1+exp(-x))
    # f'(x) = exp(-x) / (1+ exp(-x)) = 1/(1+exp(x)) = sigmoid(-x)
    layer.map { |x| 1 / (1 + Math.exp(x)) }
  }

  LogSoftmax_Derivative = ->(layer) do
    # returns the Jacobian matrix for log softmax.
    log_soft = ActivationFunctions[:LogSoftmax].call(layer)
    soft = log_soft.map { |v| Math.exp(v) }
    n = layer.size
    jacobian = Array.new(n) { Array.new(n, 0.0) }
    soft.each_with_index do |s, i|
      soft.each_with_index do |s_j, j|
        jacobian[i][j] = (i == j) ? (1 - s) : -s_j
      end
    end
    jacobian
  end

  Mish_Derivative = ->(layer) {
    # derivative of mish: approximate as:
    # tanh(softplus(x)) + x * sigmoid(x) * (1 - tanh(softplus(x))**2)
    layer.map do |x|
      softplus = Math.log(1 + Math.exp(x))
      tanh_sp = Math.tanh(softplus)
      sigmoid = 1 / (1 + Math.exp(-x))
      tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp)
    end
  }

  ReLU6_Derivative = ->(layer) {
    # derivative is 1 if x in (0,6), 0 otherwise.
    layer.map { |x| (x > 0 && x < 6) ? 1 : 0 }
  }

  Selu_Derivative = ->(layer) {
    lambda_sel = 1.0507
    alpha_sel = 1.67326
    layer.map { |x| x > 0 ? lambda_sel : lambda_sel * alpha_sel * Math.exp(x) }
  }

  Silu_Derivative = ->(layer) {
    # derivative of SiLU: sigmoid(x) + x * sigmoid(x)*(1-sigmoid(x))
    layer.map do |x|
      sig = 1 / (1 + Math.exp(-x))
      sig + x * sig * (1 - sig)
    end
  }

  SoftShrink_Derivative = ->(layer) {
    lamb = 0.5
    layer.map { |x| (x > lamb || x < -lamb) ? 1 : 0 }
  }

  Softplus_Derivative = ->(layer) {
    # derivative of softplus is sigmoid(x)
    layer.map { |x| 1 / (1 + Math.exp(-x)) }
  }

  Softsign_Derivative = ->(layer) {
    layer.map { |x| 1.0 / ((1 + x.abs) ** 2) }
  }

  SparsePlus_Derivative = ->(layer) {
    threshold = 0.5
    layer.map { |x| (x.abs >= threshold) ? 1 : 0 }
  }

  Sparsemax_Derivative = ->(layer) do
    # Compute sparsemax then its support.
    f_sparse = ActivationFunctions[:Sparsemax].call(layer)
    support = f_sparse.each_index.select { |i| f_sparse[i] > 0 }
    k = support.size
    n = layer.size
    jacobian = Array.new(n) { Array.new(n, 0.0) }
    support.each do |i|
      support.each do |j|
        jacobian[i][j] = (i == j ? 1 : 0) - (1.0 / k)
      end
    end
    jacobian
  end

  Squareplus_Derivative = ->(layer) {
    # derivative: 0.5*(1 + x/sqrt(x^2+4))
    layer.map { |x| 0.5 * (1 + x / Math.sqrt(x * x + 4)) }
  }

  TanhShrink_Derivative = ->(layer) {
    # derivative of tanh_shrink: d/dx (x - tanh(x)) = 1 - (1 - tanh(x)**2) = tanh(x)**2
    layer.map { |x| Math.tanh(x) ** 2 }
  }

  Threshold_Derivative = ->(layer) {
    threshold = 0
    layer.map { |x| x > threshold ? 1 : 0 }
  }

  Derivatives = {
    Celu => Celu_Derivative,
    Exponential => Exponential_Derivative,
    Gelu => Gelu_Derivative,
    Glu => Glu_Derivative,
    HardShrink => HardShrink_Derivative,
    HardSigmoid => HardSigmoid_Derivative,
    HardSilu => HardSilu_Derivative,
    HardTanh => HardTanh_Derivative,
    Linear => Linear_Derivative,
    LogSigmoid => LogSigmoid_Derivative,
    LogSoftmax => LogSoftmax_Derivative,
    Mish => Mish_Derivative,
    ReLU6 => ReLU6_Derivative,
    Selu => Selu_Derivative,
    Silu => Silu_Derivative,
    SoftShrink => SoftShrink_Derivative,
    Softplus => Softplus_Derivative,
    Softsign => Softsign_Derivative,
    SparsePlus => SparsePlus_Derivative,
    Sparsemax => Sparsemax_Derivative,
    Squareplus => Squareplus_Derivative,
    TanhShrink => TanhShrink_Derivative,
    Threshold => Threshold_Derivative,
    Tanh => Tanh_Derivative,
    LeakyReLU => LeakyReLU_Derivative,
    ELU => ELU_Derivative,
    ReLU => ReLU_Derivative,
    Sigmoid => Sigmoid_Derivative,
    Softmax => Softmax_Derivative
  }

end
