require_relative 'neural_net'
require_relative 'layers'
require_relative 'optimizers'

io = File.open('xor.model', 'rb')
net = Marshal.load(io)
io.close()

p (net.predict [1,0])[0].round 8
p (net.predict [0,0])[0].round 8
