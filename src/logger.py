import tensorflow as tf
import os

LOG_DIR = "./logs/agentic_ai"
os.makedirs(LOG_DIR, exist_ok=True)
summary_writer = tf.summary.create_file_writer(LOG_DIR)

def log_training_performance(start_time, end_time, model_size):
    training_time = end_time - start_time
    tf.summary.scalar(f"training_time_{model_size}", training_time, step=1)
    print(f"Training Time for {model_size}: {training_time} seconds")

def log_memory_efficiency(memory_used, model_size):
    tf.summary.scalar(f"memory_used_{model_size}", memory_used, step=1)
    print(f"Memory Usage for {model_size}: {memory_used} GB")

def log_error_rates(error_rate, model_size):
    tf.summary.scalar(f"error_rate_{model_size}", error_rate, step=1)
    print(f"Error Rate for {model_size}: {error_rate}")

def log_cost_efficiency(training_cost, model_size):
    tf.summary.scalar(f"training_cost_{model_size}", training_cost, step=1)
    print(f"Training Cost for {model_size}: ${training_cost}")

def log_infrastructure_requirements(gpu_count, model_size):
    tf.summary.scalar(f"gpu_count_{model_size}", gpu_count, step=1)
    print(f"GPU Count for {model_size}: {gpu_count}")

def log_tensorflow_stats_full(start_time, end_time, memory_used, error_rate, training_cost, gpu_count, model_size):
    with summary_writer.as_default():
        log_training_performance(start_time, end_time, model_size)
        log_memory_efficiency(memory_used, model_size)
        log_error_rates(error_rate, model_size)
        log_cost_efficiency(training_cost, model_size)
        log_infrastructure_requirements(gpu_count, model_size)
        summary_writer.flush()
