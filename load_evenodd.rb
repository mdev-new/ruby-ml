require_relative 'lib/neural_net'

net = NeuralNetwork.load 'evenodd.model'

def decompose_to_binary_array(num, size)
  binary_array = Array.new(size, 0)
  
  16.times do |i|
    binary_array[i] = (num >> i) & 1
  end
  
  binary_array
end

test_count = 20
(0..test_count).each_with_index do |inp, idx|

  number = rand(2000..64000)
  inp = decompose_to_binary_array(number, 16)
  outp = [(number % 2 == 0) ? 1 : 0]

  pred = (net.predict inp).map {|e| e.round(1) }

  puts "#{outp} : #{pred} #{pred == outp ? '✓' : '✗'}"
end
