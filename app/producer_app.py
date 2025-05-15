import streamlit as st
import pandas as pd
import time
import json
from kafka import KafkaProducer
import os
from model_wrapper import MultitaskModelWrapper

def get_producer():
    return KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

def process_csv_file(producer, csv_file, topic_name, model, delay=1.0):
    try:
        df = pd.read_csv(csv_file)
        row_count = len(df)
        
        for i, row in df.iterrows():
            text = row.get("text", "")
            if pd.isna(text) or text.strip() == "":
                continue

            predictions = model.predict_all(text)
            prediction_data = {
                task: {
                    "label": result["label"],
                    "confidence": result["confidence"]
                }
                for task, result in predictions.items()
            }

            message = {
                "text": text,
                "source": "csv",
                "predictions": prediction_data,
                "timestamp": time.time()
            }

            producer.send(topic_name, message)
            producer.flush()

            # Progress + show current message
            st.session_state.progress = (i + 1) / row_count
            st.session_state.current_message = f"Processing: {text[:50]}"
            st.session_state.messages.append(message)
            if len(st.session_state.messages) > 10:
                st.session_state.messages.pop(0)

            # Visual update
            st.progress(st.session_state.progress)
            st.info(st.session_state.current_message)

            time.sleep(delay)

        st.success("CSV streaming completed!")
    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")

def initialize_session_state():
    if "progress" not in st.session_state:
        st.session_state.progress = 0.0
    if "current_message" not in st.session_state:
        st.session_state.current_message = "Ready"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model" not in st.session_state:
        st.session_state.model = MultitaskModelWrapper()

def main():
    st.set_page_config(page_title="Twitter Stream Producer", layout="wide")

    initialize_session_state()
    st.title("🐦 Twitter Stream Producer")

    with st.sidebar:
        st.header("Settings")
        topic_name = st.text_input("Kafka Topic", value="Twitternewtopic")
        csv_delay = st.slider("CSV Streaming Delay (seconds)", 0.1, 5.0, 1.0)
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file and st.button("Start CSV Stream", type="primary"):
            st.session_state.progress = 0.0
            st.session_state.current_message = "Starting CSV stream..."
            producer = get_producer()
            with open("temp_upload.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            process_csv_file(producer, "temp_upload.csv", topic_name, st.session_state.model, csv_delay)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Send Manual Message")
        with st.form("manual_message"):
            message_text = st.text_area("Enter message text", height=100)
            submit = st.form_submit_button("Send Message")
            if submit and message_text.strip():
                try:
                    producer = get_producer()
                    predictions = st.session_state.model.predict_all(message_text)
                    prediction_data = {
                        task: {
                            "label": result["label"],
                            "confidence": result["confidence"]
                        }
                        for task, result in predictions.items()
                    }

                    message = {
                        "text": message_text,
                        "source": "manual",
                        "predictions": prediction_data,
                        "timestamp": time.time()
                    }

                    producer.send(topic_name, message)
                    producer.flush()
                    st.session_state.messages.append(message)
                    if len(st.session_state.messages) > 10:
                        st.session_state.messages.pop(0)

                    st.success("Message sent!")
                except Exception as e:
                    st.error(f"Error sending message: {str(e)}")

    with col2:
        st.subheader("Recent Messages")
        for i, msg in enumerate(reversed(st.session_state.messages)):
            st.markdown(f"**Message {i+1}**")
            st.markdown(f"**Source:** {msg.get('source', '')}")
            st.markdown(f"**Text:** {msg.get('text', '')}")
            st.markdown("**Predictions:**")
            for task, pred in msg.get("predictions", {}).items():
                st.markdown(f"- {task}: {pred['label']} ({pred['confidence']:.2%})")
            st.markdown(f"**Time:** {pd.to_datetime(msg['timestamp'], unit='s').strftime('%H:%M:%S')}")
            st.divider()

if __name__ == "__main__":
    main()
