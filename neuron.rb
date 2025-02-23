class Neuron
  attr_accessor :value, :weights, :bias, :delta

  def initialize
    @weights = []
    @bias = 0
    @value = 0
    @delta = 0
  end

  def randomize(no_inputs)
    @weights = Array.new(no_inputs) { rand(-1.0..1.0) }
    @bias = rand(-1.0..1.0)
  end
end
