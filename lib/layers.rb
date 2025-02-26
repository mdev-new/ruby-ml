require_relative 'neuron'
require_relative 'activation_functions'

module Layers

  class Input < Array
    def initialize(shape)
      super(shape[0]) { Neuron.new }
      # @shape = shape
      # TODO
    end

    def forward input
      self.each_with_index do |neuron, i|
        neuron.value = input[i]
      end
    end
  end

  class Dense < Array
    attr_reader :activation

    def initialize(size, activation = :Linear)
      super(size) { Neuron.new }
      @activation = activation
    end

    def forward input
      activation_fn = ActivationFunctions[@activation]

      z_values = self.map do |neuron|
        input.zip(neuron.weights).map { |i, w| i * w }.sum + neuron.bias
      end

      neuron_values = activation_fn.call z_values

      self.each_with_index do |neuron, i|
        neuron.value = neuron_values[i]
      end
    end
  end

  class Dropout < Array
    def initialize(rate)
      @rate = rate
    end

    def forward(input, training: true)
      return input unless training
      scale = (1.0 / (1 - @rate))

      # Create dropout mask and scale the kept units
      mask = input.map { rand > @rate ? 1.0 : 0.0 }
      input.zip(mask).map { |val, m| val * m * scale }
    end
  end

  class Conv2D
    
  end

end
