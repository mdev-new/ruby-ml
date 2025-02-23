require_relative 'neuron'
require_relative 'activation_functions'

module Layers

  class Input < Array
    def initialize(shape)
      super(shape[0]) { Neuron.new }
      # @shape = shape
      # TODO
    end
  end

  class Dense < Array
    attr_reader :activation

    def initialize(size, activation = :Linear)
      super(size) { Neuron.new }
      @activation = activation
    end
  end

  class Dropout < Array
    def initialize(rate)
      @rate = rate
    end
  end

  class Conv2D
    
  end

end
