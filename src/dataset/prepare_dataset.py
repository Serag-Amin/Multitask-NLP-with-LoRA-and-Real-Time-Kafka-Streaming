from datasets import load_dataset, DatasetDict
from typing import Tuple, Dict, List
from collections import Counter

# Registry of available tasks
TASK_REGISTRY = {
    "sentiment": ("tweet_eval", "sentiment"),
    "emotion": ("tweet_eval", "emotion"),
    "hate": ("Intuit-GenSRF/hate-speech-offensive", None),
}

label2id = {"hate": 0, "offensive": 1, "neither": 2}
id2label = {v: k for k, v in label2id.items()}

def resolve_label(label_list: List[str]) -> int:
    """
    Resolve a list of labels to a single integer class.
    """
    if not label_list:
        return label2id["neither"]
    if "hate" in label_list:
        return label2id["hate"]
    if "offensive" in label_list:
        return label2id["offensive"]
    return label2id["neither"]


def load_task_dataset(task_name: str) -> Tuple[DatasetDict, Dict[str, int], Dict[int, str]]:
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Task '{task_name}' not found. Available: {list(TASK_REGISTRY.keys())}")

    dataset_name, config_name = TASK_REGISTRY[task_name]
    dataset = load_dataset(dataset_name, config_name) if config_name else load_dataset(dataset_name)

    if task_name == "hate":
        def convert(example):
            return {"label": resolve_label(example["labels"])}  # Keep the field name "label"

        dataset = dataset.map(convert)

        # Keep only 'text' and 'label'
        dataset = dataset.remove_columns([col for col in dataset["train"].column_names if col not in ["text", "label"]])

    else:
        # Handle tweet_eval tasks normally
        label_names = dataset["train"].features["label"].names
        label2id_eval = {label: i for i, label in enumerate(label_names)}
        id2label_eval = {i: label for label, i in label2id_eval.items()}
        dataset = dataset.map(lambda x: {"text": x["text"], "label": x["label"]})
        return dataset, label2id_eval, id2label_eval

    return dataset, label2id, id2label


if __name__ == "__main__":
    task_name = "hate"
    dataset, label2id, id2label = load_task_dataset(task_name)

    print("\nProcessed hate dataset loaded.")
    print(f"Splits: {list(dataset.keys())}")
    print(f"Label2ID: {label2id}")
    print(f"ID2Label: {id2label}")

    # Print sample examples
    print("\nSample examples:")
    for i in range(5):
        ex = dataset["train"][i]
        print(f"[{id2label[ex['label']]}] {ex['text']}")

    # Print label distribution
    print("\nLabel distribution:")
    counts = Counter(dataset["train"]["label"])
    for label_id, count in counts.items():
        print(f"{id2label[label_id]} ({label_id}): {count} examples")
