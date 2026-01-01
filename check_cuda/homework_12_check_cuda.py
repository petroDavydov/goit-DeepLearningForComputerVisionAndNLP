# ------------1-----------------
from datasets import DatasetDict
import nltk
import matplotlib
from tqdm import tqdm
import warnings
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import evaluate
import torch.optim as optim
import torch.nn as nn
import torch
from datasets import load_dataset
import os
import random
from dataclasses import dataclass
from collections import Counter

import pandas as pd
import numpy as np

import spacy

# Завантажуємо моделі для англійської та польської
en_nlp = spacy.load("en_core_web_sm")
pl_nlp = spacy.load("pl_core_news_sm")

warnings.filterwarnings("ignore")

matplotlib.use("Agg")
# nltk.download("wordnet")
# nltk.download("omw-1.4")
# nltk.download("punkt")

def log_gpu_usage():
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[GPU] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")
log_gpu_usage()
# ---------2-----------
seed = 42

# Фіксуємо seed для відтворюваності
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Додатковий вивід у консоль
    print("✅ CUDA доступна")
    print("Версія CUDA (PyTorch):", torch.version.cuda)
    print("Назва GPU:", torch.cuda.get_device_name(0))
    print("Використовується пристрій:", torch.cuda.current_device())
    print("Зайнята пам'ять:", torch.cuda.memory_allocated(0) / 1024**2, "MB")
    print("Зарезервована пам'ять:", torch.cuda.memory_reserved(0) / 1024**2, "MB")
else:
    print("❌ CUDA недоступна, використовується CPU")


# створюємо випадковий тензор і відправляємо на GPU
x = torch.rand(1000, 1000).to("cuda")
print("Тензор на GPU:", x.device)
print("Зайнята пам'ять:", torch.cuda.memory_allocated(0) / 1024**2, "MB")
print("Зарезервована пам'ять:", torch.cuda.memory_reserved(0) / 1024**2, "MB")

log_gpu_usage()
# ----------3----------------

# Завантажуємо корпус Europarl для англійської та польської
dataset = load_dataset("Helsinki-NLP/europarl", "en-pl")

# Є лише train, тому робимо власний поділ

# Встановимо пропорції: 80% train, 10% valid, 10% test
split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
valid_test = split_dataset["test"].train_test_split(test_size=0.5, seed=42)

train_data = split_dataset["train"]
valid_data = valid_test["train"]
test_data = valid_test["test"]


print("Train size:", len(train_data))
print("Valid size:", len(valid_data))
print("Test size:", len(test_data))

# Перевіримо приклад
print(train_data[0]["translation"])

log_gpu_usage()
# --------4-----------------

def tokenize_example(example, en_nlp, pl_nlp, max_length, lower, sos_token, eos_token):
    # Беремо англійський та польський текст
    en_text = example["translation"]["en"]
    pl_text = example["translation"]["pl"]

    # Токенізація через spaCy
    en_tokens = [tok.text for tok in en_nlp(en_text)][:max_length]
    pl_tokens = [tok.text for tok in pl_nlp(pl_text)][:max_length]

    # Приведення до нижнього регістру (якщо потрібно)
    if lower:
        en_tokens = [t.lower() for t in en_tokens]
        pl_tokens = [t.lower() for t in pl_tokens]

    # Додаємо спеціальні токени
    en_tokens = [sos_token] + en_tokens + [eos_token]
    pl_tokens = [sos_token] + pl_tokens + [eos_token]

    return {"en_tokens": en_tokens, "pl_tokens": pl_tokens}

log_gpu_usage()
# -----------5-----------------
max_length = 100
lower = True
sos_token = "<sos>"
eos_token = "<eos>"

fn_kwargs = {
    "en_nlp": en_nlp,
    "pl_nlp": pl_nlp,
    "max_length": max_length,
    "lower": lower,
    "sos_token": sos_token,
    "eos_token": eos_token,
}

# Беремо повний корпус прикладів для тренування не змінюючи назву змінної


small_train = train_data.map(
    tokenize_example, fn_kwargs=fn_kwargs)

# Беремо 2000 прикладів для валідації
small_valid = valid_data.map(
    tokenize_example, fn_kwargs=fn_kwargs)

# Беремо 2000 прикладів для тесту
small_test = test_data.map(
    tokenize_example, fn_kwargs=fn_kwargs)

print("Train size:", len(small_train))
print("Valid size:", len(small_valid))
print("Test size:", len(small_test))


# Перевіримо приклад
print(small_train[0]["translation"])
print("EN tokens:", small_train[0]["en_tokens"])
print("PL tokens:", small_train[0]["pl_tokens"])


log_gpu_usage()
# -----------6-----------------
min_freq = 2
unk_token = "<unk>"
pad_token = "<pad>"
special_tokens = [unk_token, pad_token, sos_token, eos_token]

# Англійський словник
en_counter = Counter(
    token for example in small_train["en_tokens"] for token in example
)
en_vocab_tokens = special_tokens + [
    tok for tok, freq in en_counter.items()
    if freq >= min_freq and tok not in special_tokens
]
en_vocab = {tok: idx for idx, tok in enumerate(en_vocab_tokens)}

# Польський словник
pl_counter = Counter(
    token for example in small_train["pl_tokens"] for token in example
)
pl_vocab_tokens = special_tokens + [
    tok for tok, freq in pl_counter.items()
    if freq >= min_freq and tok not in special_tokens
]
pl_vocab = {tok: idx for idx, tok in enumerate(pl_vocab_tokens)}

# Перевірки: індекси спецтокенів мають збігатися
assert en_vocab[unk_token] == pl_vocab[unk_token]
assert en_vocab[pad_token] == pl_vocab[pad_token]

unk_index = en_vocab[unk_token]
pad_index = en_vocab[pad_token]

print("UNK index:", unk_index)
print("PAD index:", pad_index)
print("Index of 'the' in EN vocab:", en_vocab.get("the", unk_index))
print("EN vocab size:", len(en_vocab))
print("PL vocab size:", len(pl_vocab))

log_gpu_usage()


# -----------7-----------------

def numericalize_example(example, en_vocab, pl_vocab):
    en_ids = [en_vocab.get(tok, en_vocab["<unk>"])
              for tok in example["en_tokens"]]
    pl_ids = [pl_vocab.get(tok, pl_vocab["<unk>"])
              for tok in example["pl_tokens"]]
    return {"en_ids": en_ids, "pl_ids": pl_ids}


fn_kwargs = {"en_vocab": en_vocab, "pl_vocab": pl_vocab}

small_train = small_train.map(numericalize_example, fn_kwargs=fn_kwargs)
small_valid = small_valid.map(numericalize_example, fn_kwargs=fn_kwargs)
small_test = small_test.map(numericalize_example, fn_kwargs=fn_kwargs)

print(small_train[0]["translation"])
print("EN tokens:", small_train[0]["en_tokens"])
print("EN ids:", small_train[0]["en_ids"])
print("PL tokens:", small_train[0]["pl_tokens"])
print("PL ids:", small_train[0]["pl_ids"])

log_gpu_usage()
# -----------8---------------
# Перевіримо перший приклад
print("Translation:", small_train[0]["translation"])
print("EN tokens:", small_train[0]["en_tokens"])
print("EN ids:", small_train[0]["en_ids"])
print("PL tokens:", small_train[0]["pl_tokens"])
print("PL ids:", small_train[0]["pl_ids"])

log_gpu_usage()
# ------------9--------------------
data_type = "torch"
format_columns = ["en_ids", "pl_ids"]

# Переводимо датасети у torch-формат
small_train = small_train.with_format(
    type=data_type, columns=format_columns, output_all_columns=True
)
small_valid = small_valid.with_format(
    type=data_type, columns=format_columns, output_all_columns=True
)
small_test = small_test.with_format(
    type=data_type, columns=format_columns, output_all_columns=True
)

# Перевіримо структуру
print(small_train)
print("Приклад:", small_train[0])
print("EN ids:", small_train[0]["en_ids"])
print("PL ids:", small_train[0]["pl_ids"])

log_gpu_usage()
# ------------10-------------------


def get_collate_fn(pad_index):
    def collate_fn(batch):
        # Витягуємо списки індексів для EN та PL
        batch_en_ids = [example["en_ids"] for example in batch]
        batch_pl_ids = [example["pl_ids"] for example in batch]

        # Паддінг послідовностей до однакової довжини
        batch_en_ids = nn.utils.rnn.pad_sequence(
            batch_en_ids, padding_value=pad_index, batch_first=False
        )
        batch_pl_ids = nn.utils.rnn.pad_sequence(
            batch_pl_ids, padding_value=pad_index, batch_first=False
        )

        return {"en_ids": batch_en_ids, "pl_ids": batch_pl_ids}
    return collate_fn


log_gpu_usage()
# ---------------11-----------------------


def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )

log_gpu_usage()
# ---------------12-----------------------------
batch_size = 8

# Створюємо DataLoader-и
train_data_loader = get_data_loader(
    small_train, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(small_valid, batch_size, pad_index)
test_data_loader = get_data_loader(small_test, batch_size, pad_index)

# Перевіримо перший батч
batch = next(iter(train_data_loader))

print("EN ids shape:", batch["en_ids"].shape)  # [src_len, batch_size]
print("PL ids shape:", batch["pl_ids"].shape)  # [trg_len, batch_size]

# Ще один батч для перевірки
batch2 = next(iter(train_data_loader))
print("Другий батч EN ids shape:", batch2["en_ids"].shape)
print("Другий батч PL ids shape:", batch2["pl_ids"].shape)


log_gpu_usage()
# ---------------13-------------------------


# --- Encoder ---
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, pad_idx, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(
            input_dim, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(embedding_dim, encoder_hidden_dim,
                          bidirectional=True)
        self.fc = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):  # src: [src_len, batch_size]
        # [src_len, batch_size, emb_dim]
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        # outputs: [src_len, batch_size, enc_hidden_dim * 2]
        # hidden:  [2, batch_size, enc_hidden_dim]

        # concat last forward/backward hidden states → project to dec_hidden_dim
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        # hidden: [batch_size, dec_hidden_dim]
        return outputs, hidden


# --- Attention ---
class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.attn_fc = nn.Linear(
            (encoder_hidden_dim * 2) + decoder_hidden_dim, decoder_hidden_dim)
        self.v_fc = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat hidden across src_len
        # [batch_size, src_len, dec_hidden_dim]
        hidden_rep = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # [batch_size, src_len, enc_hidden_dim*2]
        enc_outs = encoder_outputs.permute(1, 0, 2)

        # [batch_size, src_len, dec_hidden_dim]
        energy = torch.tanh(self.attn_fc(
            torch.cat((hidden_rep, enc_outs), dim=2)))
        # [batch_size, src_len]
        attention = self.v_fc(energy).squeeze(2)
        return torch.softmax(attention, dim=1)


# --- Decoder ---
class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, attention, pad_idx, dropout=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(
            output_dim, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU((encoder_hidden_dim * 2) +
                          embedding_dim, decoder_hidden_dim)
        self.fc_out = nn.Linear(
            (encoder_hidden_dim * 2) + decoder_hidden_dim + embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)                           # [1, batch_size]
        # [1, batch_size, emb_dim]
        embedded = self.dropout(self.embedding(input))

        # [batch_size, src_len]
        a = self.attention(hidden, encoder_outputs)
        # [batch_size, 1, src_len]
        a = a.unsqueeze(1)

        # [batch_size, src_len, enc_hidden_dim*2]
        enc_outs = encoder_outputs.permute(1, 0, 2)
        # [batch_size, 1, enc_hidden_dim*2]
        weighted = torch.bmm(a, enc_outs)
        # [1, batch_size, enc_hidden_dim*2]
        weighted = weighted.permute(1, 0, 2)

        # [1, batch_size, (enc_hidden_dim*2)+emb_dim]
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output: [1, batch_size, dec_hidden_dim]
        # hidden: [1, batch_size, dec_hidden_dim]

        # [batch_size, emb_dim]
        embedded = embedded.squeeze(0)
        # [batch_size, dec_hidden_dim]
        output = output.squeeze(0)
        # [batch_size, enc_hidden_dim*2]
        weighted = weighted.squeeze(0)

        # [batch_size, output_dim]
        prediction = self.fc_out(
            torch.cat((output, weighted, embedded), dim=1))
        return prediction, hidden.squeeze(0), a.squeeze(1)


# --- Seq2Seq ---
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.8):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size,
                              trg_vocab_size, device=self.device)

        encoder_outputs, hidden = self.encoder(src)
        input = trg[0, :]  # перший токен завжди <sos>

        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(dim=1)  # [batch_size]
            input = trg[t] if teacher_force else top1

        return outputs

log_gpu_usage()
# ----------------14----------------------


# --- Параметри моделі ---
input_dim = len(en_vocab)   # англійський словник
output_dim = len(pl_vocab)  # польський словник
encoder_embedding_dim = 128
decoder_embedding_dim = 128
encoder_hidden_dim = 256
decoder_hidden_dim = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = './saved_models'
os.makedirs(model_dir, exist_ok=True)

# --- Ініціалізація моделі ---
attention = Attention(encoder_hidden_dim, decoder_hidden_dim)

encoder = Encoder(
    input_dim,
    encoder_embedding_dim,
    encoder_hidden_dim,
    decoder_hidden_dim,
    pad_index
)

decoder = Decoder(
    output_dim,
    decoder_embedding_dim,
    encoder_hidden_dim,
    decoder_hidden_dim,
    attention,
    pad_index
)

model = Seq2Seq(encoder, decoder, device).to(device)
print(model)

# --- Ініціалізація ваг ---


def init_weights(m):
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)

















# --- Оптимізатор та функція втрат ---
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

# --- Функція тренування ---


def train_fn(model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device):
    model.train()
    epoch_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        src = batch["en_ids"].to(device)  # англійська
        trg = batch["pl_ids"].to(device)  # польська

        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

# --- Функція валідації ---


def evaluate_fn(model, data_loader, criterion, device, teacher_forcing_ratio=0.0):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            src = batch["en_ids"].to(device)
            trg = batch["pl_ids"].to(device)

            output = model(src, trg, teacher_forcing_ratio)

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

# def log_gpu_usage():
#     allocated = torch.cuda.memory_allocated() / 1024**2
#     reserved = torch.cuda.memory_reserved() / 1024**2
#     print(f"[GPU] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")


# --- Навчання ---
n_epochs = 5
clip = 1.0
teacher_forcing_ratio = 0.5
best_valid_loss = float("inf")

train_losses = []
valid_losses = []

for epoch in range(1, n_epochs + 1):
    # тренування та валідація
    train_loss = train_fn(
        model, train_data_loader, optimizer,
        criterion, clip, teacher_forcing_ratio, device
    )
    valid_loss = evaluate_fn(model, valid_data_loader, criterion, device)

    # збереження втрат у списки
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    # збереження найкращої моделі
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), os.path.join(model_dir, "en_pl.pt"))

    # логування
    print(f"Epoch {epoch}/{n_epochs}")
    print(f"\tTrain Loss: {train_loss:.3f}")
    print(f"\tValid Loss: {valid_loss:.3f}")
    print(f"\tBest Valid Loss: {best_valid_loss:.3f}")

    # Викликати після кожної епохи
    log_gpu_usage()
# ----------------15-----------------------------

# --- Завантаження найкращої моделі ---
model.load_state_dict(torch.load(
    os.path.join(model_dir, "en_pl.pt"),
    map_location=device
))

# --- Оцінка на тестових даних ---
test_loss = evaluate_fn(model, test_data_loader, criterion, device)
print(f"| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f} |")

log_gpu_usage()
# ---------------16-------------------------

def translate_sentence(
    sentence,
    model,
    en_nlp,
    pl_nlp,
    en_vocab,
    pl_vocab,
    lower,
    sos_token,
    eos_token,
    device,
    max_output_length=25,
):
    # --- Захист від None ---
    if sentence is None or not isinstance(sentence, (str, list)):
        return [sos_token, eos_token], [sos_token, eos_token], torch.zeros(1, 1, 1)

    model.eval()
    with torch.no_grad():
        # --- Токенізація англійського речення ---
        if isinstance(sentence, str):
            en_tokens = [token.text for token in en_nlp(sentence)]
        else:
            en_tokens = [token for token in sentence]

        if lower:
            en_tokens = [token.lower() for token in en_tokens]

        en_tokens = [sos_token] + en_tokens + [eos_token]

        # --- Перетворення токенів у індекси ---
        ids = [en_vocab.get(tok, en_vocab.get("<unk>", 0))
               for tok in en_tokens]
        tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)  # [src_len, 1]

        # --- Пропуск через енкодер ---
        encoder_outputs, hidden = model.encoder(tensor)

        # --- Початковий вхід у декодер = <sos> ---
        sos_idx = int(pl_vocab.get(sos_token, pl_vocab.get("<unk>", 0)))
        eos_idx = int(pl_vocab.get(eos_token, pl_vocab.get("<unk>", 0)))

        inputs = [sos_idx]
        attentions = torch.zeros(max_output_length, 1, len(ids)).to(device)

        for i in range(max_output_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden, attention = model.decoder(
                inputs_tensor, hidden, encoder_outputs)
            attentions[i] = attention

            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)

            if predicted_token == eos_idx:
                break

        # --- Перетворення індексів назад у токени ---
        inv_pl_vocab = {idx: tok for tok, idx in pl_vocab.items()}
        pl_tokens = [inv_pl_vocab.get(idx, "<unk>") for idx in inputs]

        # --- Захист від None / порожнього списку ---
        if not pl_tokens or all(tok is None for tok in pl_tokens):
            pl_tokens = [sos_token, eos_token]

    return pl_tokens, en_tokens, attentions[:len(pl_tokens)-1]

log_gpu_usage()
# --------------17------------------------

def plot_attention(sentence_tokens, translation_tokens, attention, max_len=50):
    sentence_tokens = sentence_tokens[:max_len]
    translation_tokens = translation_tokens[:max_len]

    attn_matrix = attention[:min(len(translation_tokens), attention.shape[0]),
                            :min(len(sentence_tokens), attention.shape[1])]

    attn_matrix = attn_matrix.squeeze(1).cpu().numpy()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attn_matrix, cmap="bone")

    ax.set_xticks(range(len(sentence_tokens)))
    ax.set_yticks(range(len(translation_tokens)))
    ax.set_xticklabels(sentence_tokens, rotation=90)
    ax.set_yticklabels(translation_tokens)

    plt.xlabel("Source (EN)")
    plt.ylabel("Target (PL)")
    plt.tight_layout()

    os.makedirs("graphs", exist_ok=True)
    plt.savefig("graphs/attention_example.png")
    plt.close()

log_gpu_usage()
# ---------------18------------------------
# Беремо перший приклад із тестових даних
tr = test_data[0].get("translation")

if not tr or not isinstance(tr, dict):
    print("Bad sample at index 0")
else:
    sentence = tr.get("en", "")
    expected_translation = tr.get("pl", "")

    print("English:", sentence)
    print("Expected Polish:", expected_translation)

    # Проганяємо через модель
    translation, sentence_tokens, attention = translate_sentence(
        sentence,
        model,
        en_nlp,
        pl_nlp,
        en_vocab,
        pl_vocab,
        lower,
        sos_token,
        eos_token,
        device,
    )

    print("Predicted translation:", " ".join(translation))
    print("Source tokens:", sentence_tokens)

    # Візуалізація уваги
    plot_attention(sentence_tokens, translation, attention)

log_gpu_usage()
# -----------19-----------------

# Перекладемо перші 10 прикладів із тестового набору
for i in range(10):
    tr = test_data[i].get("translation")

    # Перевірка, чи є поле translation і чи воно словник
    if not tr or not isinstance(tr, dict):
        print(f"Bad sample at index {i}: translation missing or invalid")
        continue

    sentence = tr.get("en")
    expected_translation = tr.get("pl")

    # Перевірка, чи en/pl є рядками
    if not isinstance(sentence, str) or not isinstance(expected_translation, str):
        print(f"Bad sample at index {i}: en/pl not strings")
        continue

    translation, sentence_tokens, attention = translate_sentence(
        sentence,
        model,
        en_nlp,
        pl_nlp,
        en_vocab,
        pl_vocab,
        lower,
        sos_token,
        eos_token,
        device,
    )

    print(f"Example {i+1}")
    print(f"English: {sentence}")
    print(f"Expected Polish: {expected_translation}")
    print(f"Predicted Polish: {' '.join(translation)}")
    print("-" * 60)

    # Візуалізація уваги для кожного прикладу
    if attention is not None and attention.numel() > 0:
        plot_attention(sentence_tokens, translation, attention)
    else:
        print(f"Attention is None/empty for example {i+1}, skipping plot.")

log_gpu_usage()
# ----------------------------20----------------------------

train_losses = []
valid_losses = []

for epoch in range(n_epochs):
    train_loss = train_fn(model, train_data_loader, optimizer,
                          criterion, clip, teacher_forcing_ratio, device)
    valid_loss = evaluate_fn(model, valid_data_loader, criterion, device)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

# --- Побудова графіка втрат ---
plt.figure(figsize=(10, 8))
plt.plot(train_losses, label="Train")
plt.plot(valid_losses, label="Valid")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Зміна функції втрат (EN→PL)")
plt.legend()
# plt.tight_layout()
# plt.pause(1.0)  # збільшена пауза
# plt.close()

# --- Перекладемо кілька прикладів ---
for i in range(len(test_data)):
    tr = test_data[i].get("translation")

    if not tr or not isinstance(tr, dict):
        print(f"Bad sample at index {i}: translation missing or invalid")
        continue

    sentence = tr.get("en")
    expected_translation = tr.get("pl")

    if not isinstance(sentence, str) or not isinstance(expected_translation, str):
        print(f"Bad sample at index {i}: en/pl not strings")
        continue

    translation, sentence_tokens, attention = translate_sentence(
        sentence,
        model,
        en_nlp,
        pl_nlp,
        en_vocab,
        pl_vocab,
        lower,
        sos_token,
        eos_token,
        device,
    )

    print(f"Example {i+1}")
    print(f"English: {sentence}")
    print(f"Expected Polish: {expected_translation}")
    print(f"Predicted Polish: {' '.join(translation)}")

    if attention is not None and attention.numel() > 0:
        plot_attention(sentence_tokens, translation, attention)
    else:
        print(f"Attention is None/empty for example {i+1}, skipping plot.")

# --- METEOR + ChrF оцінка ---
meteor = evaluate.load("meteor")
chrf = evaluate.load("chrf")

preds, refs = [], []
bad_indices = []
limit = len(test_data)  # можна обмежити 500 для швидкої перевірки

for i in range(limit):
    tr = test_data[i].get("translation")

    if not tr or not isinstance(tr, dict):
        bad_indices.append(("no_translation", i, test_data[i]))
        continue

    src = tr.get("en")
    ref_pl = tr.get("pl")

    if not isinstance(src, str) or not isinstance(ref_pl, str):
        bad_indices.append(("bad_types", i, type(src), type(ref_pl)))
        continue

    pred_tokens, _, _ = translate_sentence(
        src, model, en_nlp, pl_nlp, en_vocab, pl_vocab, lower, sos_token, eos_token, device
    )

    if pred_tokens is None or not isinstance(pred_tokens, list):
        bad_indices.append(("pred_none", i, pred_tokens))
        pred_tokens = [sos_token, eos_token]

    pred_tokens = [t for t in pred_tokens if isinstance(t, str)]
    pred_tokens = [t for t in pred_tokens if t not in {
        sos_token, eos_token, "<pad>"}]

    pred_text = " ".join(pred_tokens) if pred_tokens else ""

    if not isinstance(ref_pl, str) or len(ref_pl.strip()) == 0:
        bad_indices.append(("ref_empty", i, ref_pl))
        continue

    preds.append(pred_text)
    refs.append([ref_pl])

if bad_indices:
    print("Diagnostics (first 5):", bad_indices[:5])

# ✅ ДОДАНО: перевірка на порожні preds/refs
if len(preds) == 0 or len(refs) == 0:
    print("METEOR/ChrF cannot be computed: empty preds/refs.")
else:
    meteor_score = meteor.compute(predictions=preds, references=refs)
    chrf_score = chrf.compute(predictions=preds, references=refs)

    # ✅ ДОДАНО: безпечний доступ до ключів
    if meteor_score is not None and "meteor" in meteor_score:
        print(f"METEOR: {meteor_score['meteor']:.4f}")
    else:
        print("METEOR could not be computed")

    if chrf_score is not None and "score" in chrf_score:
        print(f"ChrF: {chrf_score['score']:.4f}")
    else:
        print("ChrF could not be computed")

# ✅ ДОДАНО: діагностика для перевірки
print("preds_len:", len(preds), "refs_len:", len(refs))
print("preds_sample:", preds[:3])
print("refs_sample:", refs[:3])


# --- Статистика ---
print("preds_len:", len(preds), "refs_len:", len(refs))
print("preds_sample:", preds[:3])
print("refs_sample:", refs[:3])

log_gpu_usage()
# ----------------------------21----------------------------
os.makedirs("graphs", exist_ok=True)

# --- Train vs Valid Loss ---
plt.figure(figsize=(10, 8))
plt.plot(train_losses, label="Train")
plt.plot(valid_losses, label="Valid")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Valid Loss (EN→PL)")
plt.legend()
plt.tight_layout()
plt.savefig("graphs/train_valid_loss.png")
plt.close()

# --- Train Loss ---
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label="Train Loss", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train Loss")
plt.legend()
plt.tight_layout()
plt.savefig("graphs/train_loss.png")
plt.close()

# --- Valid Loss ---
plt.figure(figsize=(8, 6))
plt.plot(valid_losses, label="Valid Loss", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("graphs/valid_loss.png")
plt.close()

# --- Test Loss (баром) ---
plt.figure(figsize=(8, 6))
plt.bar(["Test"], [test_loss], color="green")
plt.ylabel("Loss")
plt.title("Test Loss")
plt.tight_layout()
plt.savefig("graphs/test_loss.png")
plt.close()

# --- Збереження результатів у файл TXT ---
with open("results.txt", "w", encoding="utf-8") as f:
    f.write("=== Результати навчання Seq2Seq (PL→EN) ===\n\n")

    # поепохові втрати
    for i, (tr, vl) in enumerate(zip(train_losses, valid_losses), start=1):
        f.write(f"Epoch {i}/{n_epochs}\n")
        f.write(f"\tTrain Loss: {tr:.3f}\n")
        f.write(f"\tValid Loss: {vl:.3f}\n")
        f.write(f"\tBest Valid Loss (so far): {min(valid_losses[:i]):.3f}\n\n")

    # узагальнена статистика
    f.write("Train Losses:\n")
    f.write(", ".join([f"{loss:.4f}" for loss in train_losses]) + "\n\n")
    f.write("Valid Losses:\n")
    f.write(", ".join([f"{loss:.4f}" for loss in valid_losses]) + "\n\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Perplexity: {np.exp(test_loss):.4f}\n\n")
    f.write("Графіки збережені у папку 'graphs/'\n")


log_gpu_usage()
print("Done. Результати збережені у results.txt")


