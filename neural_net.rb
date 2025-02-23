require 'yaml'
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

    @layers.first.each_with_index do |neuron, i|
      neuron.value = input[i]
    end

    @layers.each_cons(2) do |prev, current|

      activation_fn = ActivationFunctions[current.activation]

      current.each do |neuron|
        z = prev.zip(neuron.weights).map { |i, w| i.value * w }.sum + neuron.bias
        neuron.value = activation_fn.call z
      end
    end
  end

  # Todo: Implement this
  def validate(input, output)
    predictions = predict(input)
    error = predictions.zip(output).map { |p, o| (p - o)**2 }.sum / predictions.size.to_f
    puts "Validation error: #{error}"
  end

  def backpropagate(input, output)
    output_layer = @layers.last

    if @loss_function == :BinaryCrossEntropy
      activation_derivative = ActivationFunctions[:"d#{output_layer.activation}"]
      epsilon = 1e-12
      output_layer.each_with_index do |neuron, i|
        p = neuron.value
        p = [[p, epsilon].max, 1 - epsilon].min  # clip p between epsilon and 1 - epsilon
        grad_loss = - output[i] / p + (1 - output[i]) / (1 - p)
        neuron.delta = grad_loss * activation_derivative.call(neuron.value)
      end
    else
      # Default: using MSE
      activation_derivative = ActivationFunctions[:"d#{output_layer.activation}"]
      output_layer.each_with_index do |neuron, i|
        error = neuron.value - output[i]  # assuming MSE cost function
        neuron.delta = error * activation_derivative.call(neuron.value)
      end
    end

    # Compute hidden layers' deltas (skip input layer)
    ((@layers.size - 2)).downto(1) do |layer_index|
      current_layer = @layers[layer_index]
      next_layer = @layers[layer_index + 1]
      activation_derivative = ActivationFunctions[:"d#{current_layer.activation}"]
      current_layer.each_with_index do |neuron, j|
        error = next_layer.inject(0.0) do |sum, next_neuron|
          sum + next_neuron.weights[j] * next_neuron.delta
        end
        neuron.delta = error * activation_derivative.call(neuron.value)
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

  # Todo: Implement this
  def predict(input)
    feed_forward(input)
    @layers.last.map { |neuron| neuron.value }
  end


  # Serialization stuff
  def serialize
    {
      :layers => @layers.serialize,
      :optimizer => @optimizer.serialize
    }
  end

  def save_yaml(filename)
    File.write filename, (serialize).to_yaml
  end

  # Todo: Implement this
  def self.deserialize(serialized)
    layers = serialized[:layers]
    optimizer = serialized[:optimizer]

    NeuralNetwork.new layers, optimizer
  end

  # Todo: Implement this
  def self.load_yaml(filename)
    File.read filename
  end
end
