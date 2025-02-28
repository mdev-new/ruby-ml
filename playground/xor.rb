require 'parallel'
require_relative 'lib/neural_net'

def progress_bar(current, total, length = 20, start_time = nil)
  percent = (current.to_f / total) * 100
  progress = ((current.to_f / total) * length).to_i

  bar = "=" * progress + " " * (length - progress)
  percent_str = "#{percent.round(1)}%"
  start_index = [(length - percent_str.length) / 2, 0].max
  bar[start_index, percent_str.length] = percent_str

  eta_str = ""
  if start_time && current > 0
    elapsed = Time.now - start_time
    remaining = total - current
    eta = (elapsed / current) * remaining
    spp = percent > 0 ? elapsed / percent : 0
    eta_str = "ETA: #{eta.round}s, #{spp.round(1)} sec/perc"
  end

  # Return the bar and the ETA separately so the ETA can be displayed after the bar.
  "[#{bar}] #{current != total ? eta_str : " " * eta_str.length}"
end

epochs = 3000
input = [[0,0],[0,1],[1,0],[1,1]]
output = [[1, 0], [0, 1], [0, 1], [1, 0]]
# output = [[0],[1],[1],[0]]

optimizer = Optimizers::SGD.new 0.1
layers = [
  Layers::Input.new([2]),
  Layers::Dense.new(8, :ELU),
  Layers::Dense.new(2, :Softmax)
]

net = NeuralNetwork.new optimizer, layers, :CategoricalCrossEntropy
net.randomize

start = Time.now

# Parallel.each(0..epochs) do |i|
(0..epochs).each do |i|
  print "\r", (progress_bar i, epochs, 35, start)

  (0..(input.length - 1)).each do |j|
    net.fit input[j], output[j]     
  end

end

print "\n"

input.each_with_index do |inp, idx|
  p = (net.predict inp).map { |i| i.round 2 }

  if layers[-1].activation == :Softmax
    puts "#{inp} -> #{output[idx]} : #{p} ∑=#{p.sum} #{(p == output[idx] && p.sum == 1) ? '✓' : '✗'}"
  else
    puts "#{inp} -> #{output[idx]} : #{p} #{p == output[idx] ? '✓' : '✗'}"
  end

  # puts net.validate [1, 1], [1, 0]

end

# p (net.predict [1, 1]).map { |i| i.round 2 }

net.save 'xor.model'
