module Optimizers
  class SGD
    def initialize(learning_rate)
      @learning_rate = learning_rate
    end
    
    def update(weight, gradient)
      weight - @learning_rate * gradient
    end
  end

  # class RMSprop
  #   def initialize(learning_rate, decay)
  #     @learning_rate = learning_rate
  #     @decay = decay
  #   end
  # end

  # class Adam
  #   def initialize(learning_rate, beta1, beta2)
  #     @learning_rate = learning_rate
  #     @beta1 = beta1
  #     @beta2 = beta2
  #   end
  # end
end
