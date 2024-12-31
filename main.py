import tensorflow as tf
from number_theory import find_special_numbers
from decryption import generate_key, encrypt_file, decrypt_file
import json
import os

log_dir = "./logs/agentic_ai"
summary_writer = tf.summary.create_file_writer(log_dir)

episode_dir = "./episodes"
os.makedirs(episode_dir, exist_ok=True)

def run_episode(episode_number):
    special_numbers = find_special_numbers(limit=1000)
    key = generate_key()

    with summary_writer.as_default():
        tf.summary.scalar("special_numbers_count", len(special_numbers), step=episode_number)

    message = 42
    encrypted_payload_path = f"./payloads/encrypted/encrypted_{episode_number}.bin"
    decrypted_payload_path = f"./payloads/decrypted/decrypted_{episode_number}.bin"

    encrypt_file(key, "./payloads/sample_payload.bin", encrypted_payload_path)
    decrypt_file(key, encrypted_payload_path, decrypted_payload_path)

    episode_data = {
        "episode": episode_number,
        "special_numbers": special_numbers,
        "message": message,
        "encrypted_payload": encrypted_payload_path,
        "decrypted_payload": decrypted_payload_path,
    }

    with open(f"{episode_dir}/episode_{episode_number}.json", "w") as f:
        json.dump(episode_data, f)

    print(f"Episode {episode_number} saved!")

if __name__ == "__main__":
    for episode in range(1, 11):
        print(f"Running episode {episode}...")
        run_episode(episode)
