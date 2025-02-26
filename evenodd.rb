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
  "[#{bar}] #{eta_str}"
end

def decompose_to_binary_array(num, size)
  binary_array = Array.new(size, 0)
  
  16.times do |i|
    binary_array[i] = (num >> i) & 1
  end
  
  binary_array
end

input = []
output = []

max_digits = 16
numbers = 1000

(0..numbers).each do |k|

  input.append(decompose_to_binary_array(k, max_digits))
  output.append([(k % 2 == 0) ? 1 : 0])
end

epochs = 10

optimizer = Optimizers::SGD.new 0.1
layers = [
  Layers::Input.new([max_digits]),
  Layers::Dense.new(16, :ELU),
  Layers::Dense.new(1, :Sigmoid)
]

net = NeuralNetwork.new optimizer, layers, :MSE
net.randomize

start = Time.now

# Parallel.each(0..epochs, progress: true) do |i|
(0..epochs).each do |i|

  print "\r\e[2KTotal: ", (progress_bar i, epochs, 35, start), "\n"

  start_local = Time.now
  (0..(input.length - 1)).each do |j|
    print "\r\e[2KEpoch: ", (progress_bar j, input.length - 1, 35, start_local)

    net.fit input[j], output[j]
  end

  number = rand(2000..64000)
  loss = net.validate decompose_to_binary_array(number, 16), [(number % 2 == 0) ? 1 : 0]  

  print "\n\e[#{i}B"
  puts "\rValidation loss: #{loss}\n"
  print "\e[#{[i, 1].max + 3}A"

end

print "\e[#{epochs}B\n"

# input.each_with_index do |inp, idx|
#   p = (net.predict inp).map { |i| i.round 2 }

#   if layers[-1].activation == :Softmax
#     puts "#{inp} -> #{output[idx]} : #{p} ∑=#{p.sum} #{(p == output[idx] && p.sum == 1) ? '✓' : '✗'}"
#   else
#     puts "#{inp} -> #{output[idx]} : #{p} #{p == output[idx] ? '✓' : '✗'}"
#   end

#   # puts net.validate [1, 1], [1, 0]

# end

# p (net.predict [1, 1]).map { |i| i.round 2 }

net.save 'evenodd.model'
