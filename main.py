import torch
from transformers import AutoTokenizer
from models.multitask_model import MultiTaskModel, task_name_to_id
from safetensors.torch import load_file
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from collections import Counter
from dataset.multitask_dataset import prepare_multitask_datasets 


MODEL_PATH = "results_shuffled_deep128_heads_r16_seq128_3e-5_roberta_newData/checkpoint-21470"

TASK_REGISTRY = {
    "sentiment": ("tweet_eval", "sentiment"),
    "emotion": ("tweet_eval", "emotion"),
    "hate": ("Intuit-GenSRF/hate-speech-offensive", None),
}
task_id_to_name = {v: k for k, v in task_name_to_id.items()}
task_name_to_id = {"sentiment": 0, "hate": 1, "emotion": 2}
task_labels = {
    "sentiment": ["negative", "neutral", "positive"],
    "hate": ["hate", "offensive", 'neither'],
    "emotion": ["joy", "sadness", "anger", "fear"]
}

model = MultiTaskModel()
state_dict = load_file(f"{MODEL_PATH}/model.safetensors")
model.load_state_dict(state_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def predict(text, task_name, debug=False):
    if task_name not in task_name_to_id:
        raise ValueError(f"Invalid task: {task_name}")
        

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    task_id = torch.tensor([task_name_to_id[task_name]], device=device)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            task_id=task_id
        )
    task_logits = outputs["logits"][task_name]
    predicted_class = torch.argmax(task_logits, dim=1).item()

    if debug:
        print(f"Logits: {task_logits}")
        print(f"Predicted index: {predicted_class}")

    return predicted_class

task_names = ["sentiment", "emotion", "hate"]
merged_datasets, task_label_maps = prepare_multitask_datasets(task_names, tokenizer_name="roberta-base", max_length=128)
test_dataset = merged_datasets["test"]


# Evaluation function
def evaluate_task(task_name, test_dataset):
    task_id = task_name_to_id[task_name]
    task_data = test_dataset.filter(lambda x: x["task_name"] == task_name)

    all_preds = []
    all_labels = []
    error_samples = []

    for sample in tqdm(task_data, desc=f"Evaluating {task_name}"):
        text = sample["text"]
        true_label = sample["label"]
        pred_index = predict(text, task_name)

        all_preds.append(pred_index)
        all_labels.append(true_label)

        if pred_index != true_label:
            error_samples.append({
                "text": text,
                "true": true_label,
                "pred": pred_index
            })

    acc = accuracy_score(all_labels, all_preds)
    print(f"{task_name.capitalize()} Accuracy: {acc:.4f}")

    pred_counts = Counter(all_preds)
    for cls_idx, count in pred_counts.items():
        label_str = task_labels[task_name][cls_idx] if task_labels[task_name] else str(cls_idx)
        print(f"{label_str}: {count} times")

    print(f"\nSample Errors (up to 5):")
    for err in error_samples[:5]:
        print(f"Text: {err['text']}\nTrue: {err['true']} | Pred: {err['pred']}\n")

    return acc

if __name__ == "__main__":
    total_acc = []
    for task in ["sentiment", "emotion", 'hate']:
        acc = evaluate_task(task, test_dataset)
        total_acc.append(acc)

    overall = sum(total_acc) / len(total_acc)
    print(f"\nOverall Average Accuracy: {overall:.4f}")
