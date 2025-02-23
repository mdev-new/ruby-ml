require_relative 'neural_net'
require_relative 'layers'
require_relative 'optimizers'

input = [[0,0],[0,1],[1,0],[1,1]]
output = [[0],[1],[1],[0]]

optimizer = Optimizers::SGD.new 0.1
layers = [
  Layers::Input.new([2]),
  Layers::Dense.new(2, :Tanh),
  Layers::Dense.new(1, :Sigmoid)
]

net = NeuralNetwork.new optimizer, layers, :BinaryCrossEntropy
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

epochs = 100000
for i in 0..epochs
  bar = progress_bar(i, epochs)

  print "\r[#{bar}]"

  net.fit input[0], output[0]
  net.fit input[1], output[1]
  net.fit input[2], output[2]
  net.fit input[3], output[3]
end

print "\n"

p (net.predict [1,0])[0].round 8
p (net.predict [0,0])[0].round 8

io = File.open('xor.model', 'wb')

Marshal.dump(net, io)

io.close()
