from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import os

BLOCK_SIZE = 16

def encrypt_file(key: bytes, filepath: str, output_path: str):
    cipher = AES.new(key, AES.MODE_CBC)
    with open(filepath, 'rb') as f:
        plaintext = f.read()
    ciphertext = cipher.encrypt(pad(plaintext, BLOCK_SIZE))
    with open(output_path, 'wb') as f:
        f.write(cipher.iv + ciphertext)

def decrypt_file(key: bytes, filepath: str, output_path: str):
    with open(filepath, 'rb') as f:
        data = f.read()
    iv = data[:BLOCK_SIZE]
    ciphertext = data[BLOCK_SIZE:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = unpad(cipher.decrypt(ciphertext), BLOCK_SIZE)
    with open(output_path, 'wb') as f:
        f.write(plaintext)

def generate_key() -> bytes:
    return os.urandom(16)
