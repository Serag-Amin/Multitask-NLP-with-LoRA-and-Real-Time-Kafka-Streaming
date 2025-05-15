# consumer_app.py
import streamlit as st
import json
import time
import pandas as pd
import plotly.graph_objects as go


def get_consumer(topic_name):
    return KafkaConsumer(
        topic_name,
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='latest',
        enable_auto_commit=True,
        group_id='twitter_analysis_group',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )


def initialize_state():
    if 'consuming' not in st.session_state:
        st.session_state.consuming = False
    if 'label_counts' not in st.session_state:
        st.session_state.label_counts = {
            "sentiment": {},
            "emotion": {},
            "hate": {}
        }
    if 'message_history' not in st.session_state:
        st.session_state.message_history = []
    if 'consumer' not in st.session_state:
        st.session_state.consumer = None


def update_label_counts(predictions):
    for task, pred in predictions.items():
        label = pred.get("label", "unknown")
        if label in st.session_state.label_counts[task]:
            st.session_state.label_counts[task][label] += 1
        else:
            st.session_state.label_counts[task][label] = 1


def create_pie_chart(task):
    data = st.session_state.label_counts[task]
    if not data:
        return None
    fig = go.Figure(data=[go.Pie(
        labels=list(data.keys()), 
        values=list(data.values()), 
        hole=0.3
    )])
    fig.update_layout(title=f"{task.capitalize()}", height=350)
    return fig


def main():
    st.set_page_config(page_title="Kafka Consumer", layout="wide")
    initialize_state()

    st.title("📡 Real-Time Kafka Consumer")
    st.markdown("Streaming tweets with sentiment, emotion, and hate predictions.")

    with st.sidebar:
        topic_name = st.text_input("Kafka Topic", "Twitternewtopic")
        max_messages = st.slider("Maximum messages to display", 5, 50, 20)

        col_start, col_stop = st.columns(2)

        with col_start:
            start_clicked = st.button("▶️ Start", key="start_btn")
        with col_stop:
            stop_clicked = st.button("⏹️ Stop", key="stop_btn")

        if start_clicked and not st.session_state.consuming:
            st.session_state.consumer = get_consumer(topic_name)
            st.session_state.consuming = True

        if stop_clicked and st.session_state.consuming:
            st.session_state.consuming = False

        if st.button("Clear History"):
            st.session_state.label_counts = {
                "sentiment": {},
                "emotion": {},
                "hate": {}
            }
            st.session_state.message_history = []

        if st.session_state.consuming:
            st.success("✅ Consumer is running...")
        else:
            st.warning("⏸️ Consumer is stopped.")


    # Main layout
    col1, col2 = st.columns([3, 2])
     
    with col1:
        st.subheader("📊 Label Distributions")
        chart_cols = st.columns(3)
        chart_placeholders = [col.empty() for col in chart_cols]
                    
    with col2:
        st.subheader("📝 Latest Message")
        latest_message = st.empty()
        
    st.subheader("📋 Message History")
    message_table = st.empty()

    # Consumer Logic
    while st.session_state.consuming:
        try:
            msg_pack = st.session_state.consumer.poll(timeout_ms=1000)
            new_messages = []
            for tp, messages in msg_pack.items():
                for msg in messages:
                    new_messages.append(msg.value)

            if new_messages:
                for msg in new_messages:

                    msg['received_time'] = pd.to_datetime(msg.get('timestamp', time.time()), unit='s').strftime('%H:%M:%S')
    
                    st.session_state.message_history.append(msg)
                     
                    if len(st.session_state.message_history) > max_messages:
                        st.session_state.message_history = st.session_state.message_history[-max_messages:]
                     
                    if "predictions" in msg:
                        update_label_counts(msg["predictions"])
                
                  

                for i, task in enumerate(["sentiment", "emotion", "hate"]):
                    fig = create_pie_chart(task)
                    if fig:
                        chart_placeholders[i].plotly_chart(fig, use_container_width=True)
                    else:
                        chart_placeholders[i].info(f"No data for {task}")

            # latest message
            if st.session_state.message_history:
                latest_msg = st.session_state.message_history[-1]
                with latest_message.container():
                    st.markdown(f"**Text:** {latest_msg.get('text', '')}")
                    st.markdown("**Predictions:**")
                    for task in ["sentiment", "emotion", "hate"]:
                        prediction = latest_msg.get("predictions", {}).get(task, {})
                        label = prediction.get("label", "N/A")
                        confidence = prediction.get("confidence", 0.0)
                        st.markdown(f"- **{task.capitalize()}:** {label} ({confidence*100:.2f}%)")
            
            # Display message history as table
            if st.session_state.message_history:
                table_data = []
                for msg in st.session_state.message_history:
                    row = {
                        "Time": msg.get('received_time', 'N/A'),
                        "Source": msg.get('source', 'N/A'),
                        "Text": msg.get('text', '')[:50] + ('...' if len(msg.get('text', '')) > 50 else ''),
                    }
                    
                    predictions = msg.get('predictions', {})
                    for task in ["sentiment", "emotion", "hate"]:
                        pred = predictions.get(task, {})
                        label = pred.get('label', 'N/A')
                        confidence = pred.get('confidence', 0)
                        row[f"{task.capitalize()}"] = f"{label} ({confidence*100:.0f}%)"
                    
                    table_data.append(row)
                
                df = pd.DataFrame(table_data)
                message_table.dataframe(
                    df, 
                    use_container_width=True,
                    hide_index=True
                )
            else:
                message_table.info("No messages received yet.")

        except Exception as e:
            st.error(f"Error polling Kafka: {e}")
            break

        time.sleep(1)  # Wait a bit before next poll
 
    if not st.session_state.consuming:
        st.info("Consumer is idle. Press 'Start Consuming' to begin.")


if __name__ == "__main__":
    main()