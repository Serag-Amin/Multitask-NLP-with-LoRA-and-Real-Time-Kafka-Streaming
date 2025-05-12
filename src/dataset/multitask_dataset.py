from typing import Dict, List, Tuple
from datasets import concatenate_datasets, DatasetDict, Features, Value
from transformers import AutoTokenizer, RobertaTokenizer
from dataset.prepare_dataset import load_task_dataset
from collections import Counter

task_name_to_id = {"sentiment": 0, "hate": 1, "emotion": 2}

def tokenize_example(example, tokenizer, max_length: int):
    encoding = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length
    )
    encoding["label"] = example["label"]
    encoding["task_id"] = task_name_to_id[example["task_name"]]  # Use int
    return encoding


def prepare_multitask_datasets(task_names: List[str], tokenizer_name: str = "roberta-base", max_length: int = 256) -> Tuple[DatasetDict, Dict[str, Dict[str, int]]]:
    """
    Prepares multitask datasets by:
    - Loading and tagging each task dataset with 'task_name'
    - Tokenizing all text
    - Concatenating across tasks for train/val/test

    Returns:
        merged_datasets: DatasetDict with tokenized data
        task_label_maps: Dict of {task_name: {label2id, id2label}}
    """
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    all_splits = {"train": [], "validation": [], "test": []}
    task_label_maps = {}

    for task in task_names:
        dataset, label2id, id2label = load_task_dataset(task)
        task_label_maps[task] = {"label2id": label2id, "id2label": id2label}

        # Add task_name to each example
        dataset = dataset.map(lambda x: {**x, "task_name": task})

        # Tokenize each example
        dataset = dataset.map(lambda x: tokenize_example(x, tokenizer, max_length), batched=False)

        # Changing the labels to 0,1,2,3,... instead of 0:joy for consistency (different for each dataset)
        for split in dataset:
            new_features = dataset[split].features.copy()
            new_features["label"] = Value('int64')
            dataset[split] = dataset[split].cast(new_features)
        # Mainly for hate dataset as it contains only train
        if "validation" not in dataset:
            if "train" in dataset:
                split = dataset["train"].train_test_split(test_size=0.2, seed=42)
                dataset["train"] = split["train"]
                dataset["validation"] = split["test"]

        if "test" not in dataset:
            if "validation" in dataset:
                split = dataset["validation"].train_test_split(test_size=0.6, seed=42)
                dataset["validation"] = split["train"]
                dataset["test"] = split["test"]

        for split in all_splits:
            if split in dataset:
                all_splits[split].append(dataset[split])
    # Concatenation
    concated_data = {
        split: concatenate_datasets(all_splits[split]) for split in all_splits if all_splits[split]
    }

    concated_data['train'] = concated_data['train'].shuffle(seed=42)

    # Return merged datasets and task label mappings
    return DatasetDict(concated_data), task_label_maps


def print_label_distribution(merged_datasets: DatasetDict, task_names: List[str]):
    for task in task_names:
        print(f"\nLabel distribution for task: {task}")
        task_data = merged_datasets["train"].filter(lambda x: x["task_name"] == task)
        label_counts = Counter(task_data["label"])
        print(f"Label distribution: {dict(label_counts)}")


if __name__ == "__main__":
    task_names = ["emotion", "sentiment", 'hate']
    tokenizer_name = "roberta-base"
    max_length = 128

    # Prepare multitask datasets and label mappings
    merged_datasets, task_label_maps = prepare_multitask_datasets(task_names, tokenizer_name, max_length)

    # Print merged datasets information
    print("Merged Datasets Information:")
    for split in merged_datasets:
        print(f"Split: {split}, Number of examples: {len(merged_datasets[split])}")

    # Print task label mappings
    print("\nTask Label Mappings:")
    for task, label_map in task_label_maps.items():
        print(f"Task: {task}")
        print(f"Label2ID: {label_map['label2id']}")
        print(f"ID2Label: {label_map['id2label']}")
        print("-" * 30)

    # Print sample examples from the train split
    print("\nSample examples from the train split:")
    sample = merged_datasets["train"]['task_name'][0:20]
    print(sample)

    # Print label distribution for each task
    print_label_distribution(merged_datasets, task_names)
