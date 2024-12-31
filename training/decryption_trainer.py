import tensorflow as tf
from decryption import encrypt_file, generate_key
import os
import random
from tensorflow.keras.callbacks import EarlyStopping  # Import EarlyStopping

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

    # Create the EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor='loss',  # Monitors training loss
        patience=10,  # Stops after 10 epochs of no improvement
        restore_best_weights=True  # Restores weights from the best epoch
    )

    # Prepare data for model
    encrypted_paths = []
    keys = []
    for _ in range(EPOCHS):
        encrypted_path, key = generate_training_data()
        encrypted_paths.append(encrypted_path)
        keys.append(key)

    # Convert to TensorFlow datasets
    def load_data():
        for encrypted_path, key in zip(encrypted_paths, keys):
            with open(encrypted_path, 'rb') as f:
                encrypted_data = f.read()
            yield (tf.convert_to_tensor([list(encrypted_data[:256])], dtype=tf.float32), 
                   tf.convert_to_tensor(list(key), dtype=tf.float32))

    dataset = tf.data.Dataset.from_generator(load_data, 
                                             output_signature=(tf.TensorSpec(shape=(256,), dtype=tf.float32), 
                                                               tf.TensorSpec(shape=(16,), dtype=tf.float32)))

    # Train the model using fit() instead of manual epoch loop
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())

    # Fit the model with early stopping
    model.fit(dataset, epochs=EPOCHS, callbacks=[early_stopping], verbose=1)

    # Save the model
    model.save("decryption_model.h5")

