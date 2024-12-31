import tensorflow as tf
from ..decryption import encrypt_file, generate_key
import os
import random

# Constants
LEARNING_RATE = 0.01
EPOCHS = 100
TRAINING_DIR = "./payloads/training/"
TEST_DIR = "./payloads/test/"
DECRYPTED_DIR = "./payloads/decrypted/"
LOG_DIR = "./logs/agentic_ai/"

# Create necessary directories
os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(DECRYPTED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# TensorBoard writer
writer = tf.summary.create_file_writer(LOG_DIR)

def generate_training_data():
    key = generate_key()
    plaintext_path = f"{TRAINING_DIR}payload-{random.randint(1000, 9999)}.bin"
    encrypted_path = f"{TEST_DIR}encrypted-{random.randint(1000, 9999)}.bin"
    with open(plaintext_path, 'wb') as f:
        f.write(os.urandom(1024))  # Simulate a 1KB ransomware payload
    encrypt_file(key, plaintext_path, encrypted_path)
    return encrypted_path, key

class DecryptionModel(tf.keras.Model):
    def __init__(self):
        super(DecryptionModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation="relu")
        self.dense2 = tf.keras.layers.Dense(128, activation="relu")
        self.dense3 = tf.keras.layers.Dense(16)  # Predict 16-byte key

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

def train_decryption_model():
    model = DecryptionModel()
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    for epoch in range(EPOCHS):
        encrypted_path, key = generate_training_data()
        with open(encrypted_path, 'rb') as f:
            encrypted_data = f.read()

        with tf.GradientTape() as tape:
            predicted_key = model(tf.convert_to_tensor([list(encrypted_data[:256])], dtype=tf.float32))
            loss = tf.reduce_mean(tf.square(predicted_key - tf.convert_to_tensor(list(key), dtype=tf.float32)))
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Log metrics to TensorBoard
        with writer.as_default():
            tf.summary.scalar("Loss", loss, step=epoch)

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.numpy()}")

    model.save("decryption_model.h5")
