require_relative 'layers'
require_relative 'optimizers'
require_relative 'loss_functions'
require_relative 'activation_functions'

class NeuralNetwork
  attr_reader :layers

  def initialize(optimizer, layers, loss_function)
    @layers = layers
    @optimizer = optimizer
    @loss_function = loss_function
  end

  def randomize
    @layers.each_cons(2) do |prev, current|
      current.each do |neuron|
        neuron.randomize(prev.size)
      end
    end
  end

  def feed_forward(input)
    unless @layers.first.instance_of? Layers::Input
      fail "First layer must be an input layer"
    end

    @layers.each_with_index do |layer, index|

      prev =
        if index - 1 >= 0 then
          @layers[index - 1].map &:value
        else
          input
        end

      layer.forward prev
    end

  end

  def backpropagate(input, output)
    output_layer = @layers.last
    output_vals = output_layer.map(&:value)

    losses = output_layer.map.with_index { |neuron, i| @loss_function.call(neuron.value, output[i]) }

    activation_derivative = ActivationFunctions::Derivatives[output_layer.activation_fn]

    # Compute delta vector, using full Jacobian if activation is Softmax or Sparsemax.
    deltas =
      if [ActivationFunctions::Softmax, ActivationFunctions::Sparsemax].include?(output_layer.activation_fn)
        jacobian = activation_derivative.call(output_vals)
        jacobian.map { |row| row.zip(losses).map { |a, b| a * b }.sum }
      else
        derivs = activation_derivative.call(output_vals)
        losses.zip(derivs).map { |loss, deriv| loss * deriv }
      end

    output_layer.each_with_index { |neuron, i| neuron.delta = deltas[i] }

    # Compute hidden layers' deltas (skip input layer)
    ((@layers.size - 2)).downto(1) do |layer_index|
      current_layer = @layers[layer_index]
      current_vals = current_layer.map(&:value)
      next_layer = @layers[layer_index + 1]
      activation_derivative = ActivationFunctions::Derivatives[output_layer.activation_fn]

      activation_derivatives = activation_derivative.call(current_vals)

      current_layer.each_with_index do |neuron, j|
        error = next_layer.inject(0.0) do |sum, next_neuron|
          sum + next_neuron.weights[j] * next_neuron.delta
        end
        neuron.delta = error * activation_derivatives[j]
      end
    end

    # Update weights and biases (all layers except the input)
    (1...@layers.size).each do |layer_index|
      previous_layer = @layers[layer_index - 1]
      current_layer = @layers[layer_index]
      current_layer.each do |neuron|
        neuron.weights.each_with_index do |w, i|
          gradient = previous_layer[i].value * neuron.delta
          neuron.weights[i] = @optimizer.update(w, gradient)
        end
        neuron.bias = @optimizer.update(neuron.bias, neuron.delta)
      end
    end
  end

  def fit(input, output)
    feed_forward(input)
    backpropagate(input, output)
  end

  def predict(input)
    feed_forward(input)
    @layers.last.map { |neuron| neuron.value }
  end

  def validate(input, output)
    predictions = predict(input)
    error = predictions.zip(output).map { |p, o| @loss_function.call(p, o) }.sum / predictions.size.to_f
    error
  end

  def save filename
    File.open(filename, "wb") { |io| Marshal.dump(self, io) }
  end

  def self.load filename
    io = File.open(filename, "rb")
    return Marshal.load(io)
  end
end
