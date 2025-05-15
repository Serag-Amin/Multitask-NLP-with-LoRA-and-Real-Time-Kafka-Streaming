import os
import time
import subprocess
from kafka.admin import KafkaAdminClient, NewTopic
from kafka import KafkaProducer, KafkaConsumer
import json

def start_zookeeper():
    """Start Zookeeper server"""
    print("Starting Zookeeper server...")

    kafka_path = r"C:\kafka"
    cmd = [
        os.path.join(kafka_path, "bin", "windows", "zookeeper-server-start.bat"),
        os.path.join(kafka_path, "config", "zookeeper.properties")
    ]

    return subprocess.Popen(cmd)


def start_kafka_server():
    """Start Kafka server"""
    print("Starting Kafka server...")

    kafka_path = r"C:\kafka"
    cmd = [
        os.path.join(kafka_path, "bin", "windows", "kafka-server-start.bat"),
        os.path.join(kafka_path, "config", "server.properties")
    ]

    return subprocess.Popen(cmd)

def create_topic(topic_name="Twitternewtopic", num_partitions=1, replication_factor=1):
    """Create Kafka topic"""
    print(f"Creating topic: {topic_name}")
    try:
        admin_client = KafkaAdminClient(bootstrap_servers=['localhost:9092'])
        
        # Check if topic exists
        consumer = KafkaConsumer(bootstrap_servers=['localhost:9092'])
        existing_topics = consumer.topics()
        if topic_name in existing_topics:
            print(f"Topic {topic_name} already exists")
            return
        
        topic_list = [
            NewTopic(
                name=topic_name,
                num_partitions=num_partitions,
                replication_factor=replication_factor
            )
        ]
        admin_client.create_topics(new_topics=topic_list, validate_only=False)
        print(f"Topic {topic_name} created successfully")
    except Exception as e:
        print(f"Error creating topic: {e}")

def setup_kafka():
    """Setup Kafka environment"""
    zk_process = start_zookeeper()
    time.sleep(10)
    
    kafka_process = start_kafka_server()
    time.sleep(10)
    
    create_topic()
    
    return zk_process, kafka_process

if __name__ == "__main__":
    zk_process, kafka_process = setup_kafka()
    print("Kafka environment is running. Press Ctrl+C to stop.")
    
    try:
        # Keep processes running
        zk_process.wait()
        kafka_process.wait()
    except KeyboardInterrupt:
        print("Shutting down Kafka environment...")
        kafka_process.kill()
        zk_process.kill()
        print("Shutdown complete.")