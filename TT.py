import torch
import torch.nn as nn
import torch.nn.functional as F

# Внимание
class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        assert d_model % n_heads == 0 # На каждую голову целое число d_heads

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Линейные проекции для Q, K, V
        self.q_proj = nn.Linear(d_model, d_model) # (B, T, C) -> (B, T, C)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Финальная линейная проекция
        self.out_proj = nn.Linear(d_model, d_model) # W_o

        self.register_buffer("mask", None)

    # Генерация маски для скрытия последующих символов при предсказании
    def _causal_mask(self, T, device):
        mask = torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)
        '''
        (1 0 0 0
         1 1 0 0
         1 1 1 0
         1 1 1 1) - размером TxT, под каждую последовательность в батче
        '''
        return mask

    # Проход
    def forward(self, x):
        B, T, C = x.shape  # batch, time, channels

        # Получаем проекции Q, K, V
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Разбиваем линейные проекции на головы
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, T, H, Dh) -> (B, H, T, Dh)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Attention weights
        att = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)  # (B, H, T, T)

        # Накладываем маску
        mask = self._causal_mask(T, x.device) # размерность T, на GPU 
        att = att.masked_fill(mask == 0, float('-inf'))
        '''
        att(1, 1) = 
        (x  -inf -inf -inf
         x    x  -inf -inf
         x    x    x  -inf        
         x    x    x    x)
        '''

        att = F.softmax(att, dim=-1) # нормализуем значения

        # Взвешенная сумма
        out = att @ v  # (B, H, T, Dh)

        # Обратно объединяем головы
        out = out.transpose(1, 2).contiguous().view(B, T, C) # (B, H, T, Dh) -> (B, T, H, Dh) -> (B, T, C)

        # Финальный линейный слой после вниамния
        out = self.out_proj(out)

        return out

# По сути - отдельный слой Att + MLP
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()

        # Определение слоев
        self.attn = SelfAttention(d_model, n_heads) # Внимания
        self.ln1 = nn.LayerNorm(d_model) # Нормализация входа для внимания
        self.ln2 = nn.LayerNorm(d_model) # Нормализация входа для MLP

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout)
        ) # MLP

        self.dropout = nn.Dropout(dropout) # Межслойный dropout

    def forward(self, x):
        # Внимание + residual
        x = x + self.dropout(self.attn(self.ln1(x)))
        # MLP + residual
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x

# Класс модели
class ToyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model) # Слой эмбеддинга (d_model параметров на каждый элемент словаря)

        # Формирование слоев Att + MLP из TransformerBlock
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model) # Предвыходная нормализация

        self.head = nn.Linear(d_model, vocab_size) # предсказание следующего токена
        self.pos_emb = nn.Embedding(1024, d_model) # Позиционный эмбеддинг - учет позиции токена (заменить на RoPE!)

    # Проход по  модели
    def forward(self, idx):
        B, T = idx.shape

        positions = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(positions) # (B, T, C)

        # Проход по блокам Att + MLP
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x) # Предвыходная нормализация значений
        logits = self.head(x) # Еще не вероятности! -> softmax()

        return logits