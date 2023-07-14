hidden_size = 2048
l = 32
numel = (hidden_size ** 2 * 4 + hidden_size * hidden_size * 4 * 2 + hidden_size * 2) * l + 50000 * hidden_size + hidden_size
print(numel)