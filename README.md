---

## Features

1. **Encryption & Decryption**
   - Utilizes XOR encryption with a 16-byte key.
   - Generates encrypted-decrypted data pairs for model training.

2. **Decryption Model**
   - A TensorFlow-based neural network for predicting XOR keys from encrypted data.
   - Model architecture consists of three dense layers with ReLU activations.

3. **Number Theory**
   - Explores mathematical properties such as prime numbers and Fibonacci relationships.
   - Identifies "special numbers" meeting predefined criteria.

4. **Episode Management**
   - Runs simulation episodes that track encryption, decryption, and number-theory findings.
   - Saves results in structured JSON files.

5. **TensorFlow Logging**
   - Logs metrics such as loss and accuracy for model training and episode statistics.
   - Supports visualization with TensorBoard.

---

## Prerequisites

- Python 3.8 or higher
- TensorFlow 2.x
- PyCryptodome
- SymPy

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dekryptagent/dekrypt.git
   cd dekrypt
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add a sample payload:
   ```bash
   dd if=/dev/urandom of=sample_payload.bin bs=1k count=1
   ```

---

## Usage

### Running Episodes
Run the main script to execute simulation episodes:
```bash
python main.py
```

### Training the Decryption Model
Train the decryption model using the training script:
```bash
python training/decryption_trainer.py
```

### Monitoring with TensorBoard
Launch TensorBoard to view metrics:
```bash
tensorboard --logdir=logs/agentic_ai
```

### Exploring Special Numbers
Use `number_theory.py` to find special numbers:
```python
from number_theory import find_special_numbers
print(find_special_numbers(limit=1000))
```

---

## Model Details

### Decryption Model
- **Layer 1**: Dense (256 units, ReLU activation)
- **Layer 2**: Dense (128 units, ReLU activation)
- **Output Layer**: Dense (16 units, linear activation)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (learning rate: 0.01)

### Training Data
- Payloads are generated dynamically and encrypted with random 16-byte keys using XOR.
- The model predicts the encryption key from the first 256 bytes of encrypted data.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with a detailed description of your changes.

## Support

If you find Dekrypt useful or interesting, please consider giving the repository a star â­!

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=dekryptagent/dekrypt&type=Date)](https://star-history.com/#dekryptagent/dekrypt&Date)

## Socials

https://x.com/dekryptagent
