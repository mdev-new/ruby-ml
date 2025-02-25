require 'parallel'
require_relative 'lib/neural_net'

input = [[0,0],[0,1],[1,0],[1,1]]
output = [[1, 0],[0, 1],[0, 1],[1, 0]]
# output = [[0],[1],[1],[0]]

optimizer = Optimizers::SGD.new 0.1
layers = [
  Layers::Input.new([2]),
  Layers::Dense.new(8, :LeakyReLU),
  Layers::Dense.new(2, :Softmax)
]

net = NeuralNetwork.new optimizer, layers, :CategoricalCrossEntropy
net.randomize

def progress_bar(current, total, length = 20)
  percent = (current.to_f / total) * 100
  progress = ((current.to_f / total) * length).to_i

  bar = "=" * progress + " " * (length - progress)
  percent_str = "#{percent.round(1)}%"
  start_index = [(length - percent_str.length) / 2, 0].max
  bar[start_index, percent_str.length] = percent_str

  bar
end

epochs = 3000
#Parallel.each(0..epochs, progress: true) do |i|
(0..epochs).each do |i|
  print "\r[", (progress_bar i, epochs), "]"

  net.fit input[0], output[0]
  net.fit input[1], output[1]
  net.fit input[2], output[2]
  net.fit input[3], output[3]
end

print "\n"

p1 = net.predict [1,0]
p2 = net.predict [0,0]

print p1, " ", p1.sum, "\n"
print p2, " ", p2.sum, "\n"

net.save 'xor.model'
