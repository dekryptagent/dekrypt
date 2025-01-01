import os

def xor_data(data: bytes, key: bytes) -> bytes:
    return bytes([data[i] ^ key[i % len(key)] for i in range(len(data))])

def encrypt_file(key: bytes, filepath: str, output_path: str):
    with open(filepath, 'rb') as f:
        plaintext = f.read()
    ciphertext = xor_data(plaintext, key)
    with open(output_path, 'wb') as f:
        f.write(ciphertext)

def decrypt_file(key: bytes, filepath: str, output_path: str):
    with open(filepath, 'rb') as f:
        ciphertext = f.read()
    plaintext = xor_data(ciphertext, key)
    with open(output_path, 'wb') as f:
        f.write(plaintext)

def generate_key() -> bytes:
    return os.urandom(16)
