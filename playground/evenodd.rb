require 'neural_net'
require 'graphviz'

module Kernel
  def deep_clone()
    copy = clone()

    copy.instance_variables.each do |var|
      val = instance_variable_get(var)
      begin
        val = val.deep_clone(cache)
      rescue TypeError
        next
      end
      copy.instance_variable_set(var, val)
    end

    return copy
  end    
end


def progress_bar(current, total, length = 20, start_time = nil, addition_text = "")
  percent = (current.to_f / total) * 100
  progress = ((current.to_f / total) * length).to_i

  bar = "=" * progress + " " * (length - progress)
  percent_str = "#{(current != total) ? percent.round(1) : percent.round}%"
  start_index = [(length - percent_str.length) / 2, 0].max
  bar[start_index, percent_str.length] = percent_str

  eta_str = ""
  if start_time && current > 0
    elapsed = Time.now - start_time
    remaining = total - current
    eta = (elapsed / current) * remaining
    spp = percent > 0 ? elapsed / percent : 0
    eta_str = "ETA: #{eta.round}s, #{spp.round(1)} s/%"
  end

  # Return the bar and the ETA separately so the ETA can be displayed after the bar.
  "[#{bar}] #{eta_str} #{addition_text}"
end

def decompose_to_binary_array(num, size)
  binary_array = Array.new(size, 0)
  
  size.times do |i|
    binary_array[i] = (num >> i) & 1
  end

  binary_array
end

input = []
output = []

max_digits = 16
numbers = 2000

print "Vytvářím trénovací data"

numbers.times do |k|

  print "." if k % 100 == 0

  input.append(decompose_to_binary_array(k, max_digits))
  output.append([(k % 2 == 0) ? 1 : 0])
end

print "\n"

epochs = 1000

optimizer = Optimizers::SGD.new 0.1
layers = [
  Layers::Input.new([max_digits]),
#  Layers::Dense.new(128, ActivationFunctions::ReLU),  # Wider layer to capture pairwise interactions
#  Layers::Dense.new(128, ActivationFunctions::ReLU),
#  Layers::Dense.new(64, ActivationFunctions::ReLU),
Layers::Dense.new(8, ActivationFunctions::Tanh),
Layers::Dense.new(4, ActivationFunctions::Tanh),
Layers::Dense.new(4, ActivationFunctions::Tanh),
Layers::Dense.new(2, ActivationFunctions::Tanh),
Layers::Dense.new(1, ActivationFunctions::Sigmoid)
]


net = NeuralNetwork.new optimizer, layers, LossFunctions::MSE
net.randomize


max_val_loss_lines = 6
val_loss_line = 0

best_layers = []
min_loss = Float::INFINITY
min_loss_epoch = 0
start = Time.now

(0..epochs).each do |i|

  (0..(input.length - 1)).each do |j|
    print "\r\e[2K", (progress_bar (i * input.length + j), ((epochs + 1) * input.length), 35, start, ", epoch: #{i}, #{((Time.now - start)).round / (i + 1)} sec/epoch")

    net.fit input[j], output[j]
  end

  num_tests = 500
  total_loss = 0.0

  num_tests.times do
    number = rand(2000..64000).round
    target = [(number % 2 == 0) ? 1 : 0]
    total_loss += net.validate(decompose_to_binary_array(number, max_digits), target)
  end

  loss = total_loss / num_tests

  if loss.abs < min_loss.abs
    min_loss = loss
    min_loss_epoch = i
    best_layers = layers.deep_clone()
  end

  val_loss_line = (val_loss_line) % max_val_loss_lines + 1

  print "\e[#{val_loss_line}B\rAbsolute validation loss: #{loss.abs} - epoch: #{"#{i}".ljust 4}\e[#{val_loss_line}A"

end

print "\e[#{max_val_loss_lines}B\n"
puts "Min loss epoch: #{min_loss_epoch} -> #{min_loss}"

best_layers.each_with_index do |bl, i|
  layers[i] = bl
end

test_count = 20
(0..test_count).each_with_index do |inp, idx|

  number = rand(2000..64000)
  inp = decompose_to_binary_array(number, max_digits)
  outp = [(number % 2 == 0) ? 1 : 0]

  pred = (net.predict inp).map {|e| e.round(1) }

  puts "#{"#{number}".ljust 7} : #{pred[0] == 1.0 ? ("Sudé".ljust 5) : 'Liché'} #{pred == outp ? '✓' : '✗'}"
end

puts "Weights: ", layers[1][0].weights
puts "Bias: ", layers[1][0].bias
puts "Value: ", layers[1][0].value

# p (net.predict [1, 1]).map { |i| i.round 2 }

g = GraphViz.new(:G, type: :digraph)
g[:rankdir] = "LR"  # Set graph to go from left to right

neurons = {}

layers.each_with_index do |layer, layer_idx|
  layer.each_with_index do |neuron, neuron_idx|
    node_id = "#{layer_idx}_#{neuron_idx}"
    neurons[neuron] = g.add_nodes(node_id, label: neuron.bias.to_s)
  end
end

layers.each_cons(2) do |l1, l2|
  l1.each_with_index do |n1, idx1|
    l2.each_with_index do |n2, idx2|
      edge = g.add_edges(neurons[n1], neurons[n2])
      edge[:label] = n2.weights[idx1].to_s
    end
  end
end

g.output(png: "test_image.png")

