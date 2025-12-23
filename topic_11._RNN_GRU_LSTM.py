# This is double with change for python script from Topic_11_RNN_GRU_LSTM.ipynb from Keggle and Colab
# Load all necessary libraries
import os
import gc
import math
import random
from collections import defaultdict

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.optim import Adam
from tqdm import tqdm  # change for python script

from sklearn.metrics import classification_report, f1_score

import warnings
warnings.filterwarnings('ignore')

print("Визначимо шлях до даних і пристрій, на якому будемо проводити розрахунки.\n")

data_path = './conll003-englishversion/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('###\n')
print(data_path)
print(device)
print("###\n")

print(
    "Визначимо функцію для читання даних у форматі CoNLL2003 і зчитаємо їх .Оскільки нас цікавлять лише іменовані сутності, будемо зчитувати тільки елементи на позиції 0 — слова — і 3 — мітки іменованих сутностей. У коді це буде відображатися так: sentences.append((l[0],l[3].strip('\\n'))).\n")


def load_sentences(filepath):
    final = []
    sentences = []
    with open(filepath, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            if line.strip() == "" or line.startswith("-DOCSTART-"):
                if len(sentences) > 0:
                    final.append(sentences)
                    sentences = []
            else:
                l = line.split(' ')
                if len(l) >= 4:
                    sentences.append((l[0], l[3].strip('\n')))
    return final


train_sents = load_sentences(data_path + 'train.txt')
test_sents = load_sentences(data_path + 'test.txt')
val_sents = load_sentences(data_path + 'valid.txt')

train_sents = train_sents
print("###\n")
print(train_sents[:3])
print("###\n")


print("Визначимо список міток класів і закодуємо їх для чисельного представлення.\n")

ner_labels = ['O', 'B-PER', 'I-PER', 'B-ORG',
              'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
# NUM_CLASSES = len(ner_labels)  # додано для python
id2label = {str(i): label for i, label in enumerate(ner_labels)}
label2id = {value: int(key) for key, value in id2label.items()}

print("Представимо наші завантажені речення як словник, де під ключем text будуть зберігатися наші речення, а під ключем label — відповідні мітки іменованих сутностей.\n")


def get_df(samples):
    df, label = [], []
    for lines in samples:
        cur_line, cur_label = list(zip(*lines))
        df.append(list(cur_line))
        label.append([label2id[i] for i in cur_label])
    return {'text': df, 'label': label}


train_df = get_df(train_sents)
test_df = get_df(test_sents)
val_df = get_df(val_sents)

print("###\n")
print("Train first 2 samples:\n", train_df['text'][:2])
print("Test first 2 samples:\n", test_df['text'][:2])
print("Val first 2 samples:\n", val_df['text'][:2])
print("###\n")

print("Для подальшої роботи з даними нам потрібно представити їх у чисельній формі. Спочатку побудуємо словник. Для цього спершу підрахуємо кількість появи кожного слова в корпусі.\n")

word_dict = defaultdict(int)

for line in train_df['text']:
    for word in line:
        word_dict[word] += 1

print("Ми не будемо використовувати слова, які дуже рідко з’являються, для тренування. Таким чином, ми зменшимо кількість неінформативних ознак у наборі даних.\n")

lower_freq_word = []
for k, v in word_dict.items():
    if v < 2:
        lower_freq_word.append(k)

for word in lower_freq_word:
    del word_dict[word]


print("""Додамо до словника два спеціальні токени.
# Перший токен <UNK> позначатиме всі слова, які не присутні у словнику, так звані Out Of Vocabulary words, OOV words.
# Другий токен <PAD> позначає падинг (padding)\n""")

word_dict['<UNK>'] = -1
word_dict['<PAD>'] = -2


print("Створюємо словник, який буде містити слово та його індекс. Ми будемо використовувати цей словник, щоб представити наші речення в числовому вигляді для подальшої обробки нейронною мережею.\n")

word2id = {}

for idx, word in enumerate(word_dict.keys()):
    word2id[word] = idx


print("Dataset і DataLoader.\n")

print("##############################################\n")


def prepare_sequence(seq, to_ix):
    idxs = []
    for w in seq:
        if w in to_ix.keys():
            idxs.append(to_ix[w])
        else:
            idxs.append(to_ix['<UNK>'])
    return idxs


print("Опишемо клас Dataset")


class CoNLLDataset(Dataset):
    def __init__(self, df):
        self.texts = df['text']
        self.labels = df['label']

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_sequence(self.texts[item], word2id)
        label = self.labels[item]
        return {
            'input_ids': inputs,
            'labels': label
        }


print("Для тренування поточної моделі нам необхідно визначити collate-функцію.")


class Collate:
    def __init__(self, train):
        self.train = train

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        if self.train:
            output["labels"] = [sample["labels"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding

        output["input_ids"] = [
            s + (batch_max - len(s)) * [word2id['<PAD>']] for s in output["input_ids"]]
        if self.train:
            output['labels'] = [s + (batch_max - len(s)) * [-100]
                                for s in output["labels"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(
            output["input_ids"], dtype=torch.long)
        if self.train:
            output["labels"] = torch.tensor(output["labels"], dtype=torch.long)

        return output


collate_fn = Collate(True)


print("##############################################\n")

print("Клас моделі.\n")
print("##############################################\n")


class BiLSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_size, embeddings=None):
        super(BiLSTMTagger, self).__init__()

        # 1. Embedding Layer
        if embeddings is None:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embeddings = nn.Embedding.from_pretrained(embeddings)

        # 2. LSTM Layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            bidirectional=True, num_layers=3, batch_first=True)

        # 3. Dense Layer
        self.fc = nn.Linear(2*hidden_dim, output_size)

    def forward(self, batch_text):

        embeddings = self.embeddings(batch_text)

        lstm_output, _ = self.lstm(embeddings)

        logits = self.fc(lstm_output)
        return logits


print("##############################################\n")


print("Допоміжні функції для тренування\n")
print("Оскільки нас цікавить тільки якість передбачення для міток іменованих сутностей, будемо прибирати з результатів токени зі значенням, меншим за 0.\n")


def remove_predictions_for_masked_items(predicted_labels, correct_labels):

    predicted_labels_without_mask = []
    correct_labels_without_mask = []

    for p, c in zip(predicted_labels, correct_labels):
        if c > 0:
            predicted_labels_without_mask.append(p)
            correct_labels_without_mask.append(c)

    return predicted_labels_without_mask, correct_labels_without_mask


print("Тепер визначимо функцію, відповідальну за навчання й валідацію. Як валідаційну метрику використаємо macro F1.\n")


def train(model, train_loader, val_loader, batch_size, max_epochs, num_batches, patience, output_path):
    criterion = nn.CrossEntropyLoss(
        ignore_index=-100)  # we mask the <pad> labels
    optimizer = Adam(model.parameters())

    train_f_score_history = []
    dev_f_score_history = []
    no_improvement = 0
    for epoch in range(max_epochs):

        total_loss = 0
        predictions, correct = [], []
        model.train()
        for batch in tqdm(train_loader, total=num_batches, desc=f"Epoch {epoch}"):

            cur_batch_size, text_length = batch['input_ids'].shape

            pred = model(batch['input_ids'].to(device)).view(
                cur_batch_size*text_length, NUM_CLASSES)
            gold = batch['labels'].to(device).view(cur_batch_size*text_length)

            loss = criterion(pred, gold)

            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred_indices = torch.max(pred, 1)

            predicted_labels = list(pred_indices.cpu().numpy())
            correct_labels = list(batch['labels'].view(
                cur_batch_size*text_length).numpy())

            predicted_labels, correct_labels = remove_predictions_for_masked_items(predicted_labels,
                                                                                   correct_labels)

            predictions += predicted_labels
            correct += correct_labels

        train_score = f1_score(correct, predictions, average="macro")
        train_f_score_history.append(train_score)

        print("Total training loss:", total_loss)
        print("Training Macro F1:", train_score)

        total_loss = 0
        predictions, correct = [], []

        model.eval()
        with torch.no_grad():
            for batch in val_loader:

                cur_batch_size, text_length = batch['input_ids'].shape

                pred = model(batch['input_ids'].to(device)).view(
                    cur_batch_size*text_length, NUM_CLASSES)
                gold = batch['labels'].to(device).view(
                    cur_batch_size*text_length)

                loss = criterion(pred, gold)
                total_loss += loss.item()

                _, pred_indices = torch.max(pred, 1)
                predicted_labels = list(pred_indices.cpu().numpy())
                correct_labels = list(batch['labels'].view(
                    cur_batch_size*text_length).numpy())

                predicted_labels, correct_labels = remove_predictions_for_masked_items(predicted_labels,
                                                                                       correct_labels)

                predictions += predicted_labels
                correct += correct_labels

        dev_score = f1_score(correct, predictions, average="macro")

        print("Total validation loss:", total_loss)
        print("Validation Macro F1:", dev_score)

        dev_f = dev_score
        if len(dev_f_score_history) > patience and dev_f < max(dev_f_score_history):
            no_improvement += 1

        elif len(dev_f_score_history) == 0 or dev_f > max(dev_f_score_history):
            print("Saving model.")
            torch.save(model.state_dict(), OUTPUT_PATH)
            no_improvement = 0

        if no_improvement > patience:
            print("Validation F-score does not improve anymore. Stop training.")
            dev_f_score_history.append(dev_f)
            break

        dev_f_score_history.append(dev_f)

    return train_f_score_history, dev_f_score_history


print("☝ Зверніть увагу, тут ми додаємо механізм ранньої зупинки тренування.")

print("Early stopping (рання зупинка) — це стратегія в машинному навчанні, яка використовується для запобігання перенавчанню моделі\n")

print("Визначимо функцію для тестування.\n")


def test(model, test_iter, batch_size, labels, target_names):
    total_loss = 0
    predictions, correct = [], []

    model.eval()
    with torch.no_grad():

        for batch in test_iter:

            cur_batch_size, text_length = batch['input_ids'].shape

            pred = model(batch['input_ids'].to(device)).view(
                cur_batch_size*text_length, NUM_CLASSES)
            gold = batch['labels'].to(device).view(cur_batch_size*text_length)

            _, pred_indices = torch.max(pred, 1)
            predicted_labels = list(pred_indices.cpu().numpy())
            correct_labels = list(batch['labels'].view(
                cur_batch_size*text_length).numpy())

            predicted_labels, correct_labels = remove_predictions_for_masked_items(predicted_labels,
                                                                                   correct_labels)

            predictions += predicted_labels
            correct += correct_labels

    print(classification_report(correct, predictions,
          labels=labels, target_names=target_names))


print("##############################################\n")

print("Тренування моделі\n")

print("Спершу визначимо гіперпараметри моделі.\n")


EMBEDDING_DIM = 200  # was 100
HIDDEN_DIM = 128  # was 64
NUM_CLASSES = len(id2label)
MAX_EPOCHS = 50
PATIENCE = 3
BATCH_SIZE = 8  # was 32
VOCAB_SIZE = len(word2id)
OUTPUT_PATH = "./tmp/bilstmtagger"
num_batches = math.ceil(len(train_df) / BATCH_SIZE)


print("Створимо об’єкти Dataset і DataLoader.\n")

train_dataset = CoNLLDataset(train_df)
val_dataset = CoNLLDataset(val_df)
test_dataset = CoNLLDataset(test_df)

train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,  # chahe to True
                          collate_fn=collate_fn,
                          num_workers=0,  # change to 0 for python
                          pin_memory=True,
                          drop_last=False)

val_loader = DataLoader(val_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        collate_fn=collate_fn,
                        num_workers=0,  # change to 0 for python
                        pin_memory=True,
                        drop_last=False)

test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,  # change to True
                         collate_fn=collate_fn,
                         num_workers=0,  # change to 0 for python
                         pin_memory=True,
                         drop_last=False)


print("Створимо об’єкт моделі.\n")

tagger = BiLSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE+2, NUM_CLASSES)
print(f"THIS IS TAGGER:🎵 \n {tagger}")


print("Нарешті, переходимо до тренування!\n")


if __name__ == '__main__':
    train_f, dev_f = train(tagger.to(device), train_loader, val_loader,
                           BATCH_SIZE, MAX_EPOCHS, num_batches, PATIENCE, OUTPUT_PATH)

    print("Візуалізуємо навчальну й валідаційну метрики.\n")

    df = pd.DataFrame({'epochs': range(0, len(train_f)),
                       'train_f1': train_f,
                       'dev_f1': dev_f})

    plt.plot('epochs', 'train_f1', data=df, color='blue', linewidth=2)
    plt.plot('epochs', 'dev_f1', data=df, color='green', linewidth=2)
    plt.legend()
    plt.tight_layout()
    plt.pause(100)
    plt.close()

    print("Перевіримо якість моделі на тестовому наборі даних. Завантажимо кращу зі збережених моделей.\n")

    tagger = BiLSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE+2, NUM_CLASSES)
    tagger.load_state_dict(torch.load(OUTPUT_PATH))
    tagger.to(device)

    print("Виконаємо перевірку.\n")

    labels = list(label2id.keys())[1:]
    label_idxs = list(label2id.values())[1:]

    test(tagger, test_loader, BATCH_SIZE,
         labels=label_idxs, target_names=labels)


print("END OF SCRIPT\n ")
