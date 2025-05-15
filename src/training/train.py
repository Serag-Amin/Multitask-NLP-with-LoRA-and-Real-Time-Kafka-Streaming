import dataclasses
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import Trainer, TrainingArguments, AutoTokenizer, get_scheduler, BertModel, AutoConfig, AutoModel, DataCollatorWithPadding, EarlyStoppingCallback, RobertaTokenizer
from torch.optim import AdamW
from dataset.multitask_dataset import prepare_multitask_datasets  
from models.multitask_model import MultiTaskModel
from training.trainer import MultitaskTrainer
from torch.utils.data import DataLoader
import torch
import json
import numpy as np

#from src.utils import NLPDataCollator, MultitaskTrainer  # Importing the custom classes

# Training arguments setup
training_args = TrainingArguments(
    output_dir='./results_shuffled_deep128_heads_r16_seq128_3e-5_roberta_newData',               
    eval_strategy="epoch",                
    learning_rate=3e-5,                   
    per_device_train_batch_size=32,      
    per_device_eval_batch_size=128,        
    num_train_epochs=5,                   
    weight_decay=0.01,                    
    logging_dir='./logs',                 
    save_strategy="epoch",                
    logging_steps=10,                     
    load_best_model_at_end=True,          
    metric_for_best_model="eval_loss",
    fp16=True,
    gradient_accumulation_steps=1,    
)

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

task_names = ["sentiment", "hate", "emotion"] 

dataset, task_label_maps = prepare_multitask_datasets(task_names)

train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

# Initialize the base model (BERT)
# base_model = BertModel.from_pretrained("bert-base-uncased")
# config = AutoConfig.from_pretrained("bert-base-uncased")
model = MultiTaskModel()

optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

num_train_steps = len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs
lr_scheduler = get_scheduler(
    name="linear",  
    optimizer=optimizer,
    num_warmup_steps=0,  
    num_training_steps=num_train_steps,
)

# Use custom data collator
#data_collator = NLPDataCollator()

MAX_SEQUENCE_LENGTH = 128  

def data_collator(batch, tokenizer):
    padding_token_id = tokenizer.pad_token_id
    input_ids = [example["input_ids"] for example in batch]
    attention_mask = [example["attention_mask"] for example in batch]
    labels = [example["label"] for example in batch]
    task_ids = [example["task_id"] for example in batch]

    encoded = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        padding=True,
        return_tensors="pt"
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long),
        "task_id": torch.tensor(task_ids, dtype=torch.long),
    }

collate_fn = lambda batch: data_collator(batch, tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  
    eval_dataset=eval_dataset,
    #tokenizer=tokenizer,
    data_collator=collate_fn,
    #compute_metrics=compute_metrics,
    optimizers=(optimizer, lr_scheduler),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  #
)

import numpy as np

# # Filter the data for the 'hate' task only
# hate_data = train_dataset.filter(lambda x: x['task_name'] == 'hate')

# # 1. Check the label distribution for the 'hate' task
# label_counts = np.unique(hate_data['label'], return_counts=True)
# print(f"Label distribution for 'hate' task: {dict(zip(label_counts[0], label_counts[1]))}")

# # 2. Display some sample examples for inspection
# print("\nSample examples from 'hate' task:")
# for i in range(5):  # Display first 5 examples
#     example = hate_data[i]
#     print(f"Text: {example['text']}, Label: {example['label']}")

# # 3. Check if the labels are consistent with your expectations
# # Assuming that 0 represents non-hate and 1 represents hate
# labels = np.array(hate_data["label"])
# non_hate_count = np.sum(labels == 0)
# hate_count = np.sum(labels == 1)
# print(f"\nNumber of non-hate samples: {non_hate_count}")
# print(f"Number of hate samples: {hate_count}")

# # 4. Check for possible data imbalance
# total_samples = len(hate_data)
# print(f"Total number of samples in 'hate' task: {total_samples}")

trainer.train(resume_from_checkpoint=True)

trainer.evaluate(dataset["test"])

# eval_results = trainer.evaluate(dataset["test"])

# with open('evaluation_results.json', 'w') as f:
#     json.dump(eval_results, f, indent=4)

#trainer.save_model('./saved_model')
#model.save_pretrained("./saved_model")
# tokenizer.save_pretrained("./saved_model")