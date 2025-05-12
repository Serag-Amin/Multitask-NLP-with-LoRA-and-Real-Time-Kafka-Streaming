import torch
from transformers import RobertaTokenizer
from safetensors.torch import load_file
import os

# Import your model
from models.multitask_model import MultiTaskModel, task_name_to_id

class MultitaskModelWrapper:
    def __init__(self, model_path="results_shuffled_deep128_heads_r16_seq128_3e-5_roberta_newData/checkpoint-21470"):
        self.model_path = model_path
        self.task_name_to_id = task_name_to_id  
        self.task_labels = {
            "sentiment": ["negative", "neutral", "positive"],
            "hate": ["hate", "offensive", 'neither'],
            "emotion": ["anger", "joy", "optimism", "sadness"]
        }
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.max_length = 128  
    
    def _load_model(self):
        """Load the multitask model from saved state_dict"""
        model = MultiTaskModel()  
        
        if not os.path.exists(f"{self.model_path}/model.safetensors"):
            raise FileNotFoundError(f"Model file not found at {self.model_path}/model.safetensors")
        
        state_dict = load_file(f"{self.model_path}/model.safetensors")
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model
        
    def predict(self, text, task_name, return_label=True, debug=False):
        """
        Predict the class for a given text and task
        
        Args:
            text (str): Input text to classify
            task_name (str): One of "sentiment", "emotion", "hate"
            return_label (bool): Return text label instead of index
            debug (bool): Print debug info
            
        Returns:
            dict: Results containing prediction index, label, and confidence
        """
        if task_name not in self.task_name_to_id:
            raise ValueError(f"Invalid task: {task_name}")
            
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        task_id = torch.tensor([self.task_name_to_id[task_name]], device=self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                task_id=task_id
            )
            
        task_logits = outputs["logits"][task_name]
        
        probabilities = torch.nn.functional.softmax(task_logits, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
        
        result = {
            "task": task_name,
            "predicted_index": predicted_class,
            "confidence": confidence
        }
        
        if return_label:
            result["label"] = self.task_labels[task_name][predicted_class]
            
        if debug:
            print(f"Task: {task_name}")
            print(f"Logits: {task_logits}")
            print(f"Probabilities: {probabilities}")
            print(f"Predicted index: {predicted_class}")
            print(f"Predicted label: {self.task_labels[task_name][predicted_class]}")
            print(f"Confidence: {confidence:.4f}")

        return result
    
    def predict_all(self, text, debug=False):
        """Run predictions for all tasks on the input text"""
        results = {}
        for task in self.task_name_to_id.keys():
            results[task] = self.predict(text, task, debug=debug)
        return results


def main():
    
    text = "I just saw a terrible accident on the street. It's really scary."

    model_wrapper = MultitaskModelWrapper(model_path="results_shuffled_deep128_heads_r16_seq128_3e-5_roberta_newData/checkpoint-21470")

    print("Predicting individual tasks:")
    sentiment_result = model_wrapper.predict(text, task_name="sentiment", debug=True)
    print("Sentiment Prediction:", sentiment_result)

    emotion_result = model_wrapper.predict(text, task_name="emotion", debug=True)
    print("Emotion Prediction:", emotion_result)

    hate_result = model_wrapper.predict(text, task_name="hate", debug=True)
    print("Hate Prediction:", hate_result)

    print("\nPredicting for all tasks:")
    all_results = model_wrapper.predict_all(text, debug=True)
    print("All Task Predictions:", all_results)


if __name__ == "__main__":
    main()
