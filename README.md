# MultiTaskNLP-Studio

## Overview

**this_studio** is a multitask NLP system for classifying text into sentiment, emotion, and hate/offensive categories. It leverages a RoBERTa-based model with LoRA adapters for efficient multitask learning and provides a real-time Kafka-based pipeline for streaming predictions, along with Streamlit apps for interactive data production and consumption.

## Features
- **Multitask Model:** Single model for sentiment, emotion, and hate/offensive classification.
- **LoRA Adapters:** Efficient parameterization for multitask learning.
- **Dataset Preparation:** Scripts for preparing and tokenizing datasets from HuggingFace.
- **Training & Evaluation:** Scripts for training and evaluating the multitask model.
- **Real-Time Streaming:** Kafka-based producer/consumer pipeline for real-time predictions.
- **Streamlit Apps:**
  - Producer: Stream messages (from CSV or manual input) to Kafka with predictions.
  - Consumer: Visualize predictions in real time with interactive charts.

## Project Structure
```
main.py                  # Evaluate multitask model on test data
src/
  models/
    multitask_model.py   # Model architecture
  dataset/
    prepare_dataset.py   # Prepare individual datasets
    multitask_dataset.py # Prepare multitask datasets
  training/
    train.py             # Training script
app/
  model_wrapper.py       # Model loading and prediction wrapper
  producer_app.py        # Streamlit Kafka producer app
  consumer_app.py        # Streamlit Kafka consumer app
  kafka_setup.py         # Kafka/Zookeeper setup utilities
requirements.txt         # Project dependencies
exploration.ipynb        # Data exploration and analysis
LICENSE                  # MIT License
```

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the pre-trained model:
   - Visit [RoBERTa_Multitask](https://huggingface.co/SeragAmin/RoBERTa_Multitask/tree/main)
   - Download `model.safetensors` and place it in the `results_shuffled_deep128_heads_r16_seq128_3e-5_roberta_newData/checkpoint-21470/` directory
   - Note: The model is not currently compatible with Hugging Face's model loader. Use the provided `model_wrapper.py` for loading and inference.
4. (Optional) Set up Kafka and Zookeeper (see `app/kafka_setup.py`).

## Usage
### Model Training
Train the multitask model:
```bash
python src/training/train.py
```

### Model Evaluation
Evaluate the model on test data:
```bash
python main.py
```

### Real-Time Streaming
- **Start Kafka/Zookeeper:**
  - Use `app/kafka_setup.py` to start services and create topics.
- **Producer App:**
  - Run `app/producer_app.py` with Streamlit to send messages (CSV/manual) to Kafka:
    ```bash
    streamlit run app/producer_app.py
    ```
- **Consumer App:**
  - Run `app/consumer_app.py` with Streamlit to visualize predictions:
    ```bash
    streamlit run app/consumer_app.py
    ```

## License
MIT License (see LICENSE)

## Acknowledgements
- [HuggingFace Transformers & Datasets](https://huggingface.co/)
- [LoRA/PEFT](https://github.com/huggingface/peft)
- [Streamlit](https://streamlit.io/)
- [Kafka](https://kafka.apache.org/)
- [Lightning AI](https://lightning.ai/)