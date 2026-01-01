# 3. Імпорти та налаштування
from tqdm import tqdm
from collections import Counter
import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from datasets import load_dataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# --- Seed та пристрій ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Токенізатори ---
# Використовуй 'en_core_web_sm', якщо 'en_core_web_md' не встановлений
en_nlp = spacy.load("en_core_web_sm")
pl_nlp = spacy.load("pl_core_news_sm")


# --- Завантаження датасету ---

# беремо невелику частину для тесту (3%)
dataset = load_dataset("Helsinki-NLP/europarl", "en-pl", split="train[:3%]")

# розділяємо на train/valid
split_dataset = dataset.train_test_split(test_size=0.1, seed=seed)
train_dataset, valid_dataset = split_dataset["train"], split_dataset["test"]

print("Train size:", len(train_dataset))
print("Valid size:", len(valid_dataset))

# приклад першого рядка
print(train_dataset[0])


# --- Розділення датасету на train/valid ---
split_dataset = dataset.train_test_split(test_size=0.1, seed=seed)

train_dataset = split_dataset["train"]
valid_dataset = split_dataset["test"]

print("Train size:", len(train_dataset))
print("Valid size:", len(valid_dataset))


print(train_dataset[0])
print("EN:", train_dataset[0]["translation"]["en"])
print("PL:", train_dataset[0]["translation"]["pl"])


# --- Перегляд перших прикладів ---

with pd.option_context("display.max_colwidth", None):
    example_df = pd.DataFrame(train_dataset[:5])
    example_df.to_csv("train_examples.csv", index=False, encoding="utf-8")


# --- Токенізація прикладів ---
def tokenize_example(
    example,
    en_nlp,       # токенізатор для англійської
    pl_nlp,       # токенізатор для польської
    max_length,   # максимальна довжина послідовності
    lower,        # якщо True → переводимо в нижній регістр
    sos_token,    # токен початку послідовності
    eos_token     # токен кінця послідовності
):
    # дістаємо текст із поля "translation"
    en_text = example["translation"]["en"]
    pl_text = example["translation"]["pl"]

    # токенізація англійського та польського речення
    en_tokens = [tok.text for tok in en_nlp.tokenizer(en_text)][:max_length]
    pl_tokens = [tok.text for tok in pl_nlp.tokenizer(pl_text)][:max_length]

    # переведення в нижній регістр (якщо потрібно)
    if lower:
        en_tokens = [t.lower() for t in en_tokens]
        pl_tokens = [t.lower() for t in pl_tokens]

    # додаємо <sos> і <eos>
    en_tokens = [sos_token] + en_tokens + [eos_token]
    pl_tokens = [sos_token] + pl_tokens + [eos_token]

    return {"en_tokens": en_tokens, "pl_tokens": pl_tokens}


# --- Параметри токенізації ---
max_length = 200
lower = True   # якщо True → усі токени переводяться у нижній регістр
sos_token = "<sos>"  # токен початку послідовності
eos_token = "<eos>"  # токен кінця послідовності

# аргументи для функції tokenize_example
fn_kwargs = {
    "en_nlp": en_nlp,
    "pl_nlp": pl_nlp,
    "max_length": max_length,
    "lower": lower,
    "sos_token": sos_token,
    "eos_token": eos_token,
}

# застосовуємо токенізацію до train/valid
train_dataset = train_dataset.map(tokenize_example, fn_kwargs=fn_kwargs)
valid_dataset = valid_dataset.map(tokenize_example, fn_kwargs=fn_kwargs)

# перевірка результату
print(train_dataset[0])        # перший приклад
print(train_dataset[1])        # другий приклад
print(train_dataset[:5])       # перші 5 прикладів


class SimpleVocab:
    def __init__(self, tokens_iterable, specials=None, min_freq=1):
        specials = specials or []
        counter = Counter()
        for seq in tokens_iterable:
            counter.update(seq)

        # спочатку specials, потім частотні токени
        itos = list(specials)
        for tok, freq in counter.most_common():
            if freq >= min_freq and tok not in itos:
                itos.append(tok)

        self.itos = itos
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}
        self._default_index = self.stoi.get("<unk>", 0)

    def set_default_index(self, idx):
        self._default_index = int(idx)

    def __getitem__(self, token):
        return self.stoi.get(token, self._default_index)

    def lookup_indices(self, tokens):
        return [self.stoi.get(t, self._default_index) for t in tokens]

    def lookup_tokens(self, indices):
        return [self.itos[i] if 0 <= i < len(self.itos) else "<unk>" for i in indices]

    # ✅ нові методи
    def __len__(self):
        return len(self.itos)

    def __contains__(self, token):
        return token in self.stoi

    def __iter__(self):
        return iter(self.itos)


# --- створення словників ---
min_freq = 2
unk_token = "<unk>"
pad_token = "<pad>"
special_tokens = [unk_token, pad_token, sos_token, eos_token]

en_vocab = SimpleVocab(
    train_dataset["en_tokens"], specials=special_tokens, min_freq=min_freq)
pl_vocab = SimpleVocab(
    train_dataset["pl_tokens"], specials=special_tokens, min_freq=min_freq)

unk_index = en_vocab[unk_token]
pad_index = en_vocab[pad_token]

en_vocab.set_default_index(unk_index)
pl_vocab.set_default_index(unk_index)


assert en_vocab[unk_token] == pl_vocab[unk_token]
assert en_vocab[pad_token] == pl_vocab[pad_token]

unk_index = en_vocab[unk_token]
pad_index = en_vocab[pad_token]

print(unk_index)
print(pad_index)

print(en_vocab['the'])


# створення словників
en_vocab = SimpleVocab(
    train_dataset["en_tokens"], specials=special_tokens, min_freq=min_freq)
pl_vocab = SimpleVocab(
    train_dataset["pl_tokens"], specials=special_tokens, min_freq=min_freq)

# індекси special-токенів
unk_index = en_vocab[unk_token]
pad_index = en_vocab[pad_token]

# задаємо default_index
en_vocab.set_default_index(unk_index)
pl_vocab.set_default_index(unk_index)


# --- Numericalize ---
def numericalize_example(example, en_vocab, pl_vocab):
    en_ids = en_vocab.lookup_indices(example["en_tokens"])
    pl_ids = pl_vocab.lookup_indices(example["pl_tokens"])
    return {"en_ids": en_ids, "pl_ids": pl_ids}


# аргументи для функції
fn_kwargs = {"en_vocab": en_vocab, "pl_vocab": pl_vocab}

# застосовуємо до train/valid
train_dataset = train_dataset.map(numericalize_example, fn_kwargs=fn_kwargs)
valid_dataset = valid_dataset.map(numericalize_example, fn_kwargs=fn_kwargs)

# тепер можна працювати з torch DataLoader
train_dataset = train_dataset.with_format(
    type="torch",
    columns=["en_ids", "pl_ids"],
    output_all_columns=True
)
valid_dataset = valid_dataset.with_format(
    type="torch",
    columns=["en_ids", "pl_ids"],
    output_all_columns=True
)

# перевірка
print(train_dataset[0])


fn_kwargs = {
    "en_vocab": en_vocab,
    "pl_vocab": pl_vocab
}

train_dataset = train_dataset.map(numericalize_example, fn_kwargs=fn_kwargs)
valid_dataset = valid_dataset.map(numericalize_example, fn_kwargs=fn_kwargs)


print(train_dataset[0])


data_type = "torch"
format_columns = ["en_ids", "pl_ids"]

train_dataset = train_dataset.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True
)

valid_dataset = valid_dataset.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True,
)


print(train_dataset[0])


def get_collate_fn(pad_index):
    def collate_fn(batch):
        # беремо списки індексів для EN та PL
        batch_en_ids = [example["en_ids"] for example in batch]
        batch_pl_ids = [example["pl_ids"] for example in batch]

        # падимо до однакової довжини
        batch_en_ids = nn.utils.rnn.pad_sequence(
            batch_en_ids, batch_first=False, padding_value=pad_index)
        batch_pl_ids = nn.utils.rnn.pad_sequence(
            batch_pl_ids, batch_first=False, padding_value=pad_index)

        return {
            "en_ids": batch_en_ids,
            "pl_ids": batch_pl_ids,
        }
    return collate_fn


def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader


batch_size = 32

train_data_loader = get_data_loader(
    train_dataset, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_dataset, batch_size, pad_index)


next(iter(train_data_loader))


batch = next(iter(train_data_loader))
print("EN IDs", batch["en_ids"].max())
print("PL IDs :", batch["pl_ids"].max())


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, encoder_hidden_dim,
                          bidirectional=True)
        self.fc = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, src):  # (src_length, batch size)
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.attn_fc = nn.Linear(
            (encoder_hidden_dim * 2) + decoder_hidden_dim,
            decoder_hidden_dim
        )
        self.v_fc = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_length = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_length, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn_fc(
            torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v_fc(energy).squeeze(2)

        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        embedding_dim,
        encoder_hidden_dim,
        decoder_hidden_dim,
        attention,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.GRU((encoder_hidden_dim * 2) +
                          embedding_dim, decoder_hidden_dim)
        self.fc_out = nn.Linear(
            (encoder_hidden_dim * 2) + decoder_hidden_dim + embedding_dim,
            output_dim
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(
            torch.cat((output, weighted, embedded), dim=1))

        return prediction, hidden.squeeze(0), a.squeeze(1)


model_dir = os.getcwd()   # поточна робоча директорія


input_dim = len(pl_vocab)
output_dim = len(en_vocab)
encoder_embedding_dim = 128
decoder_embedding_dim = 128
encoder_hidden_dim = 256
decoder_hidden_dim = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = './working_1'


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio):
        batch_size = src.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_length, batch_size,
                              trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        input = trg[0, :]
        for t in range(1, trg_length):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs


# словники
input_dim = len(pl_vocab)       # польський словник
output_dim = len(en_vocab)      # англійський словник

# гіперпараметри
encoder_embedding_dim = 128
decoder_embedding_dim = 128
encoder_hidden_dim = 256
decoder_hidden_dim = 256

# пристрій
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# директорія для збереження моделі
model_dir = "./models_1"
os.makedirs(model_dir, exist_ok=True)

# створення компонентів
attention = Attention(encoder_hidden_dim, decoder_hidden_dim)

encoder = Encoder(
    input_dim,
    encoder_embedding_dim,
    encoder_hidden_dim,
    decoder_hidden_dim
)

decoder = Decoder(
    output_dim,
    decoder_embedding_dim,
    encoder_hidden_dim,
    decoder_hidden_dim,
    attention,
)

# Seq2Seq модель
model = Seq2Seq(encoder, decoder, device).to(device)

print(model)  # перевірка архітектури


def init_weights(m):
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)


optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)


torch.cuda.empty_cache()


def train_fn(model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device):
    model.train()
    epoch_loss = 0
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Training"):
        src = batch["pl_ids"].to(device)
        trg = batch["en_ids"].to(device)

        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)

        # reshape для CrossEntropyLoss
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

    torch.cuda.empty_cache()  # очищення кешу після епохи
    return epoch_loss / len(data_loader)


def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluating"):
            src = batch["pl_ids"].to(device)
            trg = batch["en_ids"].to(device)

            output = model(src, trg, 0)  # без teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(data_loader)


n_epochs = 5
clip = 1.0
teacher_forcing_ratio = 0.5

train_losses = []
val_losses = []
best_valid_loss = float("inf")

for epoch in tqdm(range(n_epochs), desc="Epochs"):
    train_loss = train_fn(
        model,
        train_data_loader,
        optimizer,
        criterion,
        clip,
        teacher_forcing_ratio,
        device,
    )
    train_losses.append(train_loss)

    valid_loss = evaluate_fn(
        model,
        valid_data_loader,
        criterion,
        device,
    )
    val_losses.append(valid_loss)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), os.path.join(model_dir, "en_pl.pt"))

    print(f"Epoch {epoch+1}/{n_epochs}")
    print(f"\tTrain Loss: {train_loss:.3f}")
    print(f"\tValid Loss: {valid_loss:.3f}")


# директорія для збереження графіків
plot_dir = "./plots_1"
os.makedirs(plot_dir, exist_ok=True)

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# збереження графіка у файл
plt.savefig(os.path.join(plot_dir, "loss_curve.png"))

# показати графік у VSCode/Jupyter
plt.tight_layout()
plt.pause(0.01)
plt.close()


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
    model.eval()
    with torch.no_grad():
        if isinstance(sentence, str):
            pl_tokens = [token.text for token in pl_nlp.tokenizer(sentence)]
        else:
            pl_tokens = [token for token in sentence]
        if lower:
            pl_tokens = [token.lower() for token in pl_tokens]
        pl_tokens = [sos_token] + pl_tokens + [eos_token]
        ids = pl_vocab.lookup_indices(pl_tokens)
        tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)
        encoder_outputs, hidden = model.encoder(tensor)
        inputs = en_vocab.lookup_indices([sos_token])
        attentions = torch.zeros(max_output_length, 1, len(ids))
        for i in range(max_output_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden, attention = model.decoder(
                inputs_tensor, hidden, encoder_outputs)
            attentions[i] = attention
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == en_vocab[eos_token]:
                break
        en_tokens = en_vocab.lookup_tokens(inputs)
    return en_tokens[:max_output_length], pl_tokens, attentions[:len(en_tokens)-1]


def plot_attention(sentence, translation, attention, save_path=None):
    fig, ax = plt.subplots(figsize=(16, 14))
    attention = attention.squeeze(1).cpu().numpy()

    translation = translation[1:]  # прибираємо <sos>
    attention = attention[:len(translation), :len(sentence)]

    cax = ax.matshow(attention, cmap="viridis")
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(sentence)))
    ax.set_yticks(np.arange(len(translation)))
    ax.set_xticklabels(sentence, rotation=60, ha='right', fontsize=11)
    ax.set_yticklabels(translation, fontsize=11)

    ax.set_xlabel('Вхідне речення (польською)', fontsize=14)
    ax.set_ylabel('Переклад (англійською)', fontsize=14)
    ax.set_title('Візуалізація механізму уваги', fontsize=18)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.tight_layout()
        plt.pause(0.01)
        plt.close()


# Inspect the dataset and sample item first
print(type(valid_dataset))
print("Length:", len(valid_dataset))

item = valid_dataset[0]
print("Item type:", type(item))

# Try to extract (pl, en) robustly
item = valid_dataset[0]

if isinstance(item, dict):
    tr = item.get("translation")
    if tr and isinstance(tr, dict):
        pl = tr.get("pl")
        en = tr.get("en")
    else:
        raise KeyError(f"Translation field missing. Available keys: {list(item.keys())}")
else:
    raise TypeError("valid_dataset[0] is not a dict")

sentence = pl
expected_translation = en
print("PL sentence:", sentence)
print("EN expected:", expected_translation)


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

print(translation)
print(sentence_tokens)


translation, sentence_tokens, attention = translate_sentence(
    sentence,
    model,
    en_nlp,
    pl_nlp,
    en_vocab,
    pl_vocab,
    lower=True,
    sos_token="<sos>",
    eos_token="<eos>",
    device=device,
)

# якщо translation містить індекси
if isinstance(translation[0], int):
    translation = en_vocab.lookup_tokens(translation)

plot_attention(sentence_tokens, translation, attention)


item = valid_dataset[5]

# якщо це просто рядок
sentence = item
expected_translation = None  # переклад відсутній у структурі

print("Sentence:", sentence)
print("Expected:", expected_translation)


translation, sentence_tokens, attention = translate_sentence(
    sentence,
    model,
    en_nlp,
    pl_nlp,
    en_vocab,
    pl_vocab,
    lower=True,
    sos_token="<sos>",
    eos_token="<eos>",
    device=device,
)

# якщо translation містить індекси
if isinstance(translation[0], int):
    translation = en_vocab.lookup_tokens(translation)

print("Translation:", " ".join(translation))
print("Source tokens:", sentence_tokens)
print("Attention shape:", attention.shape)


translation, sentence_tokens, attention = translate_sentence(
    sentence,
    model,
    en_nlp,
    pl_nlp,
    en_vocab,
    pl_vocab,
    lower=True,
    sos_token="<sos>",
    eos_token="<eos>",
    device=device,
)

# якщо translation містить індекси
if isinstance(translation[0], int):
    translation = en_vocab.lookup_tokens(translation)

plot_attention(sentence_tokens, translation, attention)


print(train_dataset)
print(train_dataset.column_names)
print(train_dataset[0])
