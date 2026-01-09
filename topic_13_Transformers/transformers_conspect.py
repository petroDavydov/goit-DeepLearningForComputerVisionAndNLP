# -1- Імпорти
from torch.optim import AdamW
from transformers import (
    pipeline,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    DataCollatorForSeq2Seq,
    default_data_collator
)
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset  # fixed
import os
import argparse
import torch
from tqdm.auto import tqdm
import evaluate
import nltk
import numpy as np
import collections
import random
import gc
import warnings
warnings.filterwarnings('ignore')

# Runtime / debug flags (CLI)
parser = argparse.ArgumentParser()
parser.add_argument('--pin-memory', dest='pin_memory',
                    action='store_true', help='Enable pin_memory (default)')
parser.add_argument('--no-pin-memory', dest='pin_memory',
                    action='store_false', help='Disable pin_memory')
parser.set_defaults(pin_memory=True)
parser.add_argument('--disable-cudnn', action='store_true',
                    help='Disable cuDNN for debugging')
parser.add_argument('--cuda-launch-blocking', dest='cuda_launch_blocking',
                    action='store_true', help='Set CUDA_LAUNCH_BLOCKING=1 for debugging')
parser.add_argument('--debug-small', action='store_true',
                    help='Use smaller batch size for debugging')
args = parser.parse_args()
if args.cuda_launch_blocking:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
if args.disable_cudnn:
    torch.backends.cudnn.enabled = False


# -2- Пристрій
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
    print("Memory Allocated:", torch.cuda.memory_allocated()/1024**3, "GB")


# -3- Допоміжна функція
def sample_n_examples(dataset, p):
    n = int(len(dataset) * p)
    indices = random.sample(range(len(dataset)), n)
    return dataset.select(indices)


# -4- Завантаження датасету
raw_datasets = load_dataset("squad")
print(raw_datasets)


# -5- Перевірка прикладу
print("Context: ", raw_datasets["train"][0]["context"])
print("Question: ", raw_datasets["train"][0]["question"])
print("Answer: ", raw_datasets["train"][0]["answers"])


# -6- Фільтрація (діагностика)
raw_datasets["train"].filter(lambda x: len(x["answers"]["text"]) != 1)


# -7- Перевірка validation
print(raw_datasets["validation"][0]["answers"])
print(raw_datasets["validation"][2]["answers"])


# -8- Модель і токенізатор
model_checkpoint = "bert-base-cased"
model = AutoModelForQuestionAnswering.from_pretrained(
    "./working").to(device)
print(next(model.parameters()).device)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
torch.cuda.empty_cache()

# -9- Базова токенізація
context = raw_datasets["train"][0]["context"]
question = raw_datasets["train"][0]["question"]

inputs = tokenizer(question, context)
tokenizer.decode(inputs["input_ids"])
torch.cuda.empty_cache()

# -10- Токенізація зі stride
inputs = tokenizer(
    question,
    context,
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
)
for ids in inputs["input_ids"]:
    print(tokenizer.decode(ids))
    print()


# -11- Токенізація з offsets
inputs = tokenizer(
    question,
    context,
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)
inputs.keys()
torch.cuda.empty_cache()

# -12- Mapping
inputs["overflow_to_sample_mapping"]


# -13- Multi-example токенізація
inputs = tokenizer(
    raw_datasets["train"][2:6]["question"],
    raw_datasets["train"][2:6]["context"],
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)
print(f"The 4 examples gave {len(inputs['input_ids'])} features.")
print(
    f"Here is where each comes from: {inputs['overflow_to_sample_mapping']}.")
torch.cuda.empty_cache()

# -14- Обчислення позицій
answers = raw_datasets["train"][2:6]["answers"]
start_positions = []
end_positions = []
for i, offset in enumerate(inputs["offset_mapping"]):
    sample_idx = inputs["overflow_to_sample_mapping"][i]
    answer = answers[sample_idx]
    start_char = answer["answer_start"][0]
    end_char = answer["answer_start"][0] + len(answer["text"][0])
    sequence_ids = inputs.sequence_ids(i)
    idx = 0
    while sequence_ids[idx] != 1:
        idx += 1
    context_start = idx
    while sequence_ids[idx] == 1:
        idx += 1
    context_end = idx - 1
    if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
        start_positions.append(0)
        end_positions.append(0)
    else:
        idx = context_start
        while idx <= context_end and offset[idx][0] <= start_char:
            idx += 1
        start_positions.append(idx - 1)
        idx = context_end
        while idx >= context_start and offset[idx][1] >= end_char:
            idx -= 1
        end_positions.append(idx + 1)
print(start_positions, end_positions)
torch.cuda.empty_cache()

# -15- Перевірка відповіді
idx = 0
sample_idx = inputs["overflow_to_sample_mapping"][idx]
answer = answers[sample_idx]["text"][0]
start = start_positions[idx]
end = end_positions[idx]
labeled_answer = tokenizer.decode(inputs["input_ids"][idx][start: end + 1])
print(f"Theoretical answer: {answer}, labels give: {labeled_answer}")


# -16- Функція препроцесингу train

max_length = 384
stride = 128


def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


torch.cuda.empty_cache()

# -18- Функція препроцесингу validation


def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


torch.cuda.empty_cache()

# -25- Функція compute_metrics


def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(
                start_logit)[-1: -n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0]: offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [
        {"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)


torch.cuda.empty_cache()

if __name__ == "__main__":

    # -17- Map train
    train_dataset = raw_datasets["train"].map(
        preprocess_training_examples,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        num_proc=1,
        keep_in_memory=True
    )
    print(len(raw_datasets["train"]), len(train_dataset))
    torch.cuda.empty_cache()
    # -19- Map validation

    validation_dataset = raw_datasets["validation"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
        num_proc=1,
        keep_in_memory=True
    )
    print(len(raw_datasets["validation"]), len(validation_dataset))

    # -20- Метрика
    metric = evaluate.load("squad")

    # -21- Форматування датасетів і DataLoader
    from typing import cast  # !!!!
    from torch.utils.data import Dataset as TorchDataset  # !!!!
    torch.cuda.empty_cache()
    train_dataset.set_format("torch")
    validation_set = validation_dataset.remove_columns(
        ["example_id", "offset_mapping"])
    validation_set.set_format("torch")

    BATCH_SIZE = 2 if args.debug_small else 8
    print(
        f"Config: pin_memory={args.pin_memory}, cudnn_enabled={torch.backends.cudnn.enabled}, batch_size={BATCH_SIZE}")
    train_dataloader = DataLoader(
        cast(TorchDataset, train_dataset),
        # train_dataset, # HuggingFace Dataset
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=args.pin_memory
    )
    eval_dataloader = DataLoader(
        cast(TorchDataset, validation_set),
        # validation_set,
        collate_fn=default_data_collator,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=args.pin_memory
    )
    torch.cuda.empty_cache()
    # -22- Оптимізатор
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # -23- Гіперпараметри
    n_best = 20
    max_answer_length = 30
    num_train_epochs = 3
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    # -24- Шлях збереження
    output_dir = './working'
    torch.cuda.empty_cache()
    # -26- Навчання
    print("ПОЧАЛОСЯ НАВЧАННЯ")
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        model.eval()
        start_logits, end_logits = [], []
        for batch in tqdm(eval_dataloader):
            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
            start_logits.append(outputs.start_logits.cpu().numpy())
            end_logits.append(outputs.end_logits.cpu().numpy())
        start_logits = np.concatenate(start_logits)[:len(validation_dataset)]
        end_logits = np.concatenate(end_logits)[:len(validation_dataset)]
        metrics = compute_metrics(start_logits, end_logits,
                                  validation_dataset, raw_datasets["validation"])
        print(f"epoch {epoch}:", metrics)
        model.save_pretrained(output_dir)
        torch.cuda.empty_cache()
