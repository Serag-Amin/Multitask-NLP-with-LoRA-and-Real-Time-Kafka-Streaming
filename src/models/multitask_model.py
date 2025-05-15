import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, RobertaTokenizer, RobertaModel, RobertaConfig, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import LoraConfig, LoraModel

task_name_to_id = {"sentiment": 0, "hate": 1, "emotion": 2}

# Number of classes for each task
task_num_labels = {
    "sentiment": 3,
    "hate": 3,
    "emotion": 4
}



class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        model_name="roberta-base"
        config = RobertaConfig.from_pretrained(model_name)
        base_model  = RobertaModel.from_pretrained(model_name, config=config)

        # self.task_weights = {
        #     "sentiment": 1, 
        #     "hate": 2,      
        #     "emotion": 1       
        # }

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=['query', 'value'],
            lora_dropout=0.05,
            bias='none',
            task_type='SEQ_CLS'
        )

        self.model = LoraModel(base_model, lora_config, adapter_name="shared")

        # self.model.add_adapter(lora_config, adapter_name= 'hate')    # For hate
        # self.model.set_adapter("shared")  # Set default


        hidden_size = config.hidden_size
        dropout_prob = 0.1
        intermediate_size = 128 

        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(intermediate_size, task_num_labels["sentiment"])
        )

        self.hate_head = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(intermediate_size, task_num_labels["hate"])
        )

        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(intermediate_size, task_num_labels["emotion"])
        )
        #self.bert.print_trainable_parameters()r
        print(f"Trainable parameters (LoRA): {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        print(f"Total parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, task_id=None, labels=None):

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # Use first token (CLS)

        sentiment_mask = task_id == task_name_to_id["sentiment"]
        hate_mask = task_id == task_name_to_id["hate"]
        emotion_mask = task_id == task_name_to_id["emotion"]

        logits = {}
        loss = 0

    # Sentiment task
        if sentiment_mask.any():
            sentiment_pooled = pooled[sentiment_mask]
            sentiment_logits = self.sentiment_head(sentiment_pooled)
            logits["sentiment"] = sentiment_logits
            if labels is not None:
                sentiment_labels = labels[sentiment_mask]
                loss += self.loss_fct(sentiment_logits, sentiment_labels)
        else:
            logits["sentiment"] = torch.empty(0, task_num_labels["sentiment"], device=input_ids.device)

        # Hate task
        if hate_mask.any():
            hate_pooled = pooled[hate_mask]
            hate_logits = self.hate_head(hate_pooled)
            logits["hate"] = hate_logits
            if labels is not None:
                hate_labels = labels[hate_mask]
                loss += self.loss_fct(hate_logits, hate_labels)
        else:
            logits["hate"] = torch.empty(0, task_num_labels["hate"], device=input_ids.device)

        # Emotion task
        if emotion_mask.any():
            emotion_pooled = pooled[emotion_mask]
            emotion_logits = self.emotion_head(emotion_pooled)
            logits["emotion"] = emotion_logits
            if labels is not None:
                emotion_labels = labels[emotion_mask]
                loss += self.loss_fct(emotion_logits, emotion_labels)
        else:
            logits["emotion"] = torch.empty(0, task_num_labels["emotion"], device=input_ids.device)

        return {"loss": loss, "logits": logits} if labels is not None else {"logits": logits}

    