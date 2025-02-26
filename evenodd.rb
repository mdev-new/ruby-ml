require_relative 'lib/neural_net'

def progress_bar(current, total, length = 20, start_time = nil, addition_text = "")
  percent = (current.to_f / total) * 100
  progress = ((current.to_f / total) * length).to_i

  bar = "=" * progress + " " * (length - progress)
  percent_str = "#{(current != total) ? percent.round(2) : percent.round(1)}%"
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
  "[#{bar}] #{eta_str} #{addition_text}"
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

epochs = 1000

optimizer = Optimizers::SGD.new 0.1
layers = [
  Layers::Input.new([max_digits]),
  Layers::Dense.new(16, :ELU),
  Layers::Dense.new(1, :Sigmoid)
]

net = NeuralNetwork.new optimizer, layers, :MSE
net.randomize

start = Time.now

max_val_loss_lines = 10
val_loss_line = 0

(0..epochs).each do |i|

  (0..(input.length - 1)).each do |j|
    print "\r\e[2K", (progress_bar (i * epochs + j), (epochs * input.length), 35, start, "Epoch: #{i}")

    net.fit input[j], output[j]
  end

  number = rand(2000..64000)
  loss = net.validate decompose_to_binary_array(number, 16), [(number % 2 == 0) ? 1 : 0]  

  val_loss_line = (val_loss_line + 1) % max_val_loss_lines

  print "\e[#{val_loss_line}B\rValidation loss: #{loss}\e[#{val_loss_line}A"

end

print "\e[#{max_val_loss_lines}B\n"
net.save 'evenodd.model'

test_count = 20

(0..test_count).each_with_index do |inp, idx|

  number = rand(2000..64000)
  inp = decompose_to_binary_array(number, 16)
  outp = [(number % 2 == 0) ? 1 : 0]

  pred = net.predict inp

  if layers[-1].activation == :Softmax
    puts "#{outp} : #{pred} ∑=#{p.sum} #{(p == output[idx] && p.sum == 1) ? '✓' : '✗'}"
  else
    puts "#{outp} : #{pred} #{pred == outp ? '✓' : '✗'}"
  end

end

# p (net.predict [1, 1]).map { |i| i.round 2 }

