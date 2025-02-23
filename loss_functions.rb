epsilon = 1e-12

LossFunctions = {
  :MSE => ->(p, t) { p - t },
  
  :BinaryCrossEntropy => ->(p, t) do
    p_clipped = [[p, epsilon].max, 1 - epsilon].min
    (-t / p_clipped) + ((1 - t) / (1 - p_clipped))
  end,
  
  :CategoricalCrossEntropy => ->(p, t) do
    p_clipped = [[p, epsilon].max, 1 - epsilon].min
    -t / p_clipped
  end
}
