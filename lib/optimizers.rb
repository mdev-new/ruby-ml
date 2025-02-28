module Optimizers
  class SGD
    def initialize(learning_rate)
      @learning_rate = learning_rate
    end
    
    def update(weight, gradient)
      weight - @learning_rate * gradient
    end
  end

  class RMSprop
    def initialize(learning_rate, decay = 0.9, epsilon = 1e-8)
      @learning_rate = learning_rate
      @decay = decay
      @epsilon = epsilon
      @r = 0
    end

    def update(weight, gradient)
      @r = @decay * @r + (1 - @decay) * gradient**2
      weight - @learning_rate * gradient / (Math.sqrt(@r) + @epsilon)
    end
  end

  class Adam
    def initialize(learning_rate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8)
      @learning_rate = learning_rate
      @beta1 = beta1
      @beta2 = beta2
      @epsilon = epsilon
      @m = 0
      @v = 0
      @t = 0
    end

    def update(weight, gradient)
      @t += 1
      @m = @beta1 * @m + (1 - @beta1) * gradient
      @v = @beta2 * @v + (1 - @beta2) * gradient**2
      m_hat = @m / (1 - @beta1**@t)
      v_hat = @v / (1 - @beta2**@t)
      weight - @learning_rate * m_hat / (Math.sqrt(v_hat) + @epsilon)
    end
  end

  class AdamW
    def initialize(learning_rate, weight_decay, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8)
      @learning_rate = learning_rate
      @weight_decay = weight_decay
      @beta1 = beta1
      @beta2 = beta2
      @epsilon = epsilon
      @m = 0
      @v = 0
      @t = 0
    end

    def update(weight, gradient)
      @t += 1
      @m = @beta1 * @m + (1 - @beta1) * gradient
      @v = @beta2 * @v + (1 - @beta2) * gradient**2
      m_hat = @m / (1 - @beta1**@t)
      v_hat = @v / (1 - @beta2**@t)
      weight_updated = weight - @learning_rate * m_hat / (Math.sqrt(v_hat) + @epsilon)
      # Apply decoupled weight decay
      weight_updated - @learning_rate * @weight_decay * weight
    end
  end

  class Adadelta
    def initialize(decay = 0.95, epsilon = 1e-6)
      @decay = decay
      @epsilon = epsilon
      @Eg = 0   # Accumulated gradient squared
      @Edx = 0  # Accumulated delta squared
    end

    def update(weight, gradient)
      @Eg = @decay * @Eg + (1 - @decay) * gradient**2
      update_val = Math.sqrt(@Edx + @epsilon) / Math.sqrt(@Eg + @epsilon) * gradient
      weight_updated = weight - update_val
      @Edx = @decay * @Edx + (1 - @decay) * update_val**2
      weight_updated
    end
  end

  class Adagrad
    def initialize(learning_rate, epsilon = 1e-8)
      @learning_rate = learning_rate
      @epsilon = epsilon
      @accum = 0
    end

    def update(weight, gradient)
      @accum += gradient**2
      weight - @learning_rate * gradient / (Math.sqrt(@accum) + @epsilon)
    end
  end

  class Adamax
    def initialize(learning_rate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8)
      @learning_rate = learning_rate
      @beta1 = beta1
      @beta2 = beta2
      @epsilon = epsilon
      @m = 0
      @u = 0
      @t = 0
    end

    def update(weight, gradient)
      @t += 1
      @m = @beta1 * @m + (1 - @beta1) * gradient
      @u = [@beta2 * @u, gradient.abs].max
      m_hat = @m / (1 - @beta1**@t)
      weight - (@learning_rate / (@u + @epsilon)) * m_hat
    end
  end

  class Adafactor
    # A simplified scalar version of Adafactor.
    def initialize(learning_rate, beta2 = 0.999, epsilon = 1e-8)
      @learning_rate = learning_rate
      @beta2 = beta2
      @epsilon = epsilon
      @v = 0
    end

    def update(weight, gradient)
      @v = @beta2 * @v + (1 - @beta2) * gradient**2
      weight - @learning_rate * gradient / (Math.sqrt(@v) + @epsilon)
    end
  end

  class Nadam
    def initialize(learning_rate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8)
      @learning_rate = learning_rate
      @beta1 = beta1
      @beta2 = beta2
      @epsilon = epsilon
      @m = 0
      @v = 0
      @t = 0
    end

    def update(weight, gradient)
      @t += 1
      m_old = @m
      @m = @beta1 * @m + (1 - @beta1) * gradient
      @v = @beta2 * @v + (1 - @beta2) * gradient**2
      m_hat = (@beta1 * m_old + (1 - @beta1) * gradient) / (1 - @beta1**@t)
      v_hat = @v / (1 - @beta2**@t)
      weight - @learning_rate * m_hat / (Math.sqrt(v_hat) + @epsilon)
    end
  end

  class Ftrl
    def initialize(alpha, beta, l1 = 0, l2 = 0)
      @alpha = alpha
      @beta = beta
      @l1 = l1
      @l2 = l2
      @z = 0
      @n = 0
    end

    def update(weight, gradient)
      @n += gradient**2
      sigma = (Math.sqrt(@n) - Math.sqrt(@n - gradient**2)) / @alpha
      @z += gradient - sigma * weight

      if @z.abs <= @l1
        0
      else
        sign = @z < 0 ? -1 : 1
        (sign * (@z.abs - @l1)) / ((@beta + Math.sqrt(@n)) / @alpha + @l2)
      end
    end
  end

  class Lion
    def initialize(learning_rate, beta = 0.9)
      @learning_rate = learning_rate
      @beta = beta
      @m = 0
    end

    def update(weight, gradient)
      @m = @beta * @m + (1 - @beta) * gradient
      # Use the sign of the momentum for the update.
      weight - @learning_rate * (@m <=> 0)
    end
  end

  class Lamb
    def initialize(learning_rate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-6)
      @learning_rate = learning_rate
      @beta1 = beta1
      @beta2 = beta2
      @epsilon = epsilon
      @m = 0
      @v = 0
      @t = 0
    end

    def update(weight, gradient)
      @t += 1
      @m = @beta1 * @m + (1 - @beta1) * gradient
      @v = @beta2 * @v + (1 - @beta2) * gradient**2
      m_hat = @m / (1 - @beta1**@t)
      v_hat = @v / (1 - @beta2**@t)
      step = m_hat / (Math.sqrt(v_hat) + @epsilon)
      w_norm = weight.abs
      step_norm = step.abs
      trust_ratio = (w_norm.zero? || step_norm.zero?) ? 1 : w_norm / step_norm
      weight - @learning_rate * trust_ratio * step
    end
  end

  class LossScaleOptimizer
    # Wraps another optimizer and rescales gradients.
    def initialize(optimizer, loss_scale)
      @optimizer = optimizer
      @loss_scale = loss_scale
    end

    def update(weight, gradient)
      scaled_gradient = gradient / @loss_scale
      @optimizer.update(weight, scaled_gradient)
    end
  end

end
