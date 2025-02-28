Epsilon = 1e-12

module LossFunctions
  MSE = ->(p, t) { p - t }

  BinaryCrossEntropy = ->(p, t) do
    p_clipped = [[p, Epsilon].max, 1 - Epsilon].min
    (-t / p_clipped) + ((1 - t) / (1 - p_clipped))
  end

  CategoricalCrossEntropy = ->(p, t) do
    p_clipped = [[p, Epsilon].max, 1 - Epsilon].min
    -t / p_clipped
  end

  KLDivergence = ->(p, t) do
    p_clipped = [[p, Epsilon].max, 1 - Epsilon].min
    -t / p_clipped
  end

  SparseCategoricalCrossEntropy = ->(p, t) do
    # Assume `p` is an array of probabilities and `t` is the target index.
    p_clipped = p.map { |pi| [[pi, Epsilon].max, 1 - Epsilon].min }
    p_clipped.each_with_index.map { |pi, i| i == t ? -1.0 / pi : 0.0 }
  end

  Poisson = ->(p, t) do
    p_clipped = [[p, Epsilon].max, 1 - Epsilon].min
    1 - t / p_clipped
  end

  # CTC = ->(p, t) do
  #   # Placeholder derivative for the CTC loss (actual CTC gradients are computed via a separate forward-backward algorithm).
  #   p - t
  # end
end
