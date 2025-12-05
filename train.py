import TT
import torch
import torch.nn.functional as F
import time
from _global_params import batch_size, lenght, d_model, n_layers, heads, train_iters, vocab_size, save_interval, save_path

# Вынесем всё на VRAM для максимальной скорости (размер датасета и весов относительно мал)
device = torch.device('cuda')
# Загружаем предзагатовленный тензор датасета
ds =  torch.load('E:/dataset/tensoried/ds_tokens.pt', device)
ds = ds.long()
ds_len = len(ds)

# Генерация батча
def get_batch(batch_size, lenght):
    # Берем случайные срезы
    starts = torch.randint(ds_len-lenght-1, (batch_size,)) # batch_size-количество разных вхождений
    X = torch.stack([ds[start:start+lenght] for start in starts]).cuda() # (B, T), сразу в VRAM
    Y = torch.stack([ds[start+1:start+lenght+1] for start in starts]).cuda()
    return X, Y

class Trainer():
    def __init__(self, model, optimizer, scaler, device=torch.device('cuda')):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler
        
    def _train_step(self, batch):
        self.model.train() # режим обучения

        # Получаем входные токены и целевые токены
        x, y = batch

        self.optimizer.zero_grad() # очищаем градиенты

        # Прогоняем через модель
        with torch.amp.autocast(device_type='cuda'):
            logits = self.model(x)
            B, T, C = logits.shape
            logits = logits.reshape(B*T, C)
            y = y.reshape(B*T)
            loss = F.cross_entropy(logits, y)

        self.scaler.scale(loss).backward() # Обратный проход
        self.scaler.step(self.optimizer) # Применение градиентов
        self.scaler.update()
        
        time.sleep(0.02)

        return loss.item()

    def train(self, train_iters, starting_step=0):
        for i in range(train_iters+1):
            loss = self._train_step(get_batch(batch_size, lenght))

            if i % 100 == 0:
                print(f'step {i+starting_step}: loss={loss}')

            if i % save_interval == 0 and i > 0:
                torch.save({
                    "model": m.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "step": i
                }, save_path.format(step=i))

                print(f"Сохранено: {save_path.format(step=i+starting_step)}")

mode = int(input("Select train mode (0 - new, 1 - resume): "))

m = TT.ToyTransformer(vocab_size, d_model, heads, n_layers).to(device) # создаем образец трансформера
opt = torch.optim.AdamW(m.parameters(), lr=1e-4) # образец оптимизатора
scaler = torch.amp.GradScaler() # FP16

if mode == 1:
    checkpoint = torch.load("E:/checkpoints/model_step_8000.pt")
    m.load_state_dict(checkpoint['model'])

    missing, unexpected = m.load_state_dict(checkpoint['model'], strict=False)
    print("Missing:", missing)
    print("Unexpected:", unexpected)

    #opt.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    step = checkpoint['step']
else:
    step = 0

total_params = sum(p.numel() for p in m.parameters())
print(f"Total parameters: {total_params:,}")
trainer = Trainer(m, opt, scaler)

trainer.train(train_iters, step)