require_relative 'neural_net'

net = NeuralNetwork.load 'xor.model'

p net.predict [1,0]
p net.predict [0,0]
