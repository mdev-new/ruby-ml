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

# 1000 epochs
for i in 0..5000
  #        A  B  A^B !A^B
  net.fit input[0], output[0]
  net.fit input[1], output[1]
  net.fit input[2], output[2]
  net.fit input[3], output[3]

  # rnd = rand(0..3)
  # net.validate input[rnd], output[rnd]

end

p (net.predict [1,0])[0].round 8
p (net.predict [0,0])[0].round 8

io = File.open('xor_model', 'wb')

Marshal.dump(net, io)

io.close()
