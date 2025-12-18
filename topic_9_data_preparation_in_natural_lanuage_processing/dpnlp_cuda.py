import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------
# 1. CUDA + FP16
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = torch.amp.GradScaler("cuda")  # FP16
print("Device:", device)

# -----------------------------
# 2. MODEL + TOKENIZER
# -----------------------------
model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
).to(device)

# -----------------------------
# 3. LOAD DATA
# -----------------------------
df = pd.read_csv("./Reviews.csv", index_col="Id")
df = df[df["Score"] != 3]
df["sentiment"] = df["Score"].apply(lambda x: 1 if x >= 4 else 0)

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# -----------------------------
# 4. DATASET + DATALOADER
# -----------------------------


class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.texts = df["Text"].tolist()
        self.labels = df["sentiment"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


train_dataset = ReviewDataset(train_df, tokenizer)
test_dataset = ReviewDataset(test_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# -----------------------------
# 5. OPTIMIZER
# -----------------------------
optimizer = AdamW(model.parameters(), lr=2e-5)

# -----------------------------
# 6. TRAINING LOOP + VALIDATION
# -----------------------------
epochs = 1
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)
    total_loss = 0

    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch = batch["labels"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):  # FP16
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_batch
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # -----------------------------
    # VALIDATION
    # -----------------------------
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_batch
            )
            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(test_loader)
    val_losses.append(avg_val_loss)

    print(f"Validation Loss: {avg_val_loss}")

# -----------------------------
# 7. ROC-AUC ON TEST SET
# -----------------------------
model.eval()
all_probs = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch = batch["labels"].cpu().numpy()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()

        all_probs.extend(probs)
        all_labels.extend(labels_batch)

auc = roc_auc_score(all_labels, all_probs)
print("ROC-AUC:", auc)

# -----------------------------
# 8. SAVE MODEL
# -----------------------------
model.save_pretrained("./distilbert_finetuned_amazon")
tokenizer.save_pretrained("./distilbert_finetuned_amazon")

# -----------------------------
# 9. PLOT LOSS
# -----------------------------
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
