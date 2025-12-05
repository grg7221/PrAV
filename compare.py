import torch
from TT import ToyTransformer
from tiktoken import get_encoding
from _global_params import heads, vocab_size, d_model, n_layers, lenght, save_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = get_encoding('gpt2') # токенайзер gpt2, vocab_size ~50k 

ckpt_path = save_path

steps = [12000, 14000, 16000, 18000, 20000] # какие чекпоинты проверять
prompt = "Once upon a time"
gen_tokens = 200

@torch.no_grad()
def generate(model, idx, max_new_tokens, max_seq_len=128):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -max_seq_len:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits / 0.8, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_tok], dim=1)
    return idx

for step in steps:
    path = ckpt_path.format(step=step)
    print("=" * 70)
    print(f"Loading checkpoint {path}")

    # Загружаем чекпоинт
    ckpt = torch.load(path, map_location=device)

    # Создаём модель
    model = ToyTransformer(vocab_size, d_model, heads, n_layers).to(device)
    model.load_state_dict(ckpt["model"])

    # Токенизируем вход
    inp = torch.tensor([vocab.encode(prompt)], dtype=torch.long).to(device)

    # Генерация
    out = generate(model, inp, gen_tokens, lenght)

    # Декодируем вывод
    text = vocab.decode(out[0].tolist())

    print(f"\n--- Generation at step {step} ---\n")
    print(text)
    print("\n")
