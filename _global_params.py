from tiktoken import get_encoding

batch_size = 12 # B
lenght = 256 # T
d_model = 768 # C
n_layers = 8
heads = 8
vocab_size = get_encoding('gpt2').n_vocab

save_interval = 2000  # сохранять каждые N шагов
save_path = "E:/checkpoints/model_step_{step}.pt"
train_iters = 10000