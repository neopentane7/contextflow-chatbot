# ContextFlow Hybrid Chatbot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-000000.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

A sophisticated hybrid chatbot system combining **LSTM** and **Transformer** architectures with advanced inference strategies and a modern web interface.

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [API](#-api-documentation) • [Training](#-training)

</div>

---

## ✨ Features

- 🧠 **Hybrid Architecture**: Combines LSTM for sequential processing and Transformer for attention mechanisms
- 🎯 **Advanced Inference**: Multiple decoding strategies including:
  - Beam Search for high-quality responses
  - Temperature Sampling for creative outputs
  - Constrained Decoding for safe responses
- 💬 **Context Awareness**: Maintains conversation history and contextual understanding
- 🌐 **Web Interface**: Clean, responsive UI built with Flask
- 🐳 **Docker Support**: Easy deployment with Docker and Docker Compose
- 📊 **Pattern-Based Responses**: Intelligent fallback for common queries
- 🔄 **GPT-2 Integration**: Hybrid approach with custom model + GPT-2 fallback

---

## 🏗️ Architecture

```
Input Text
    ↓
Tokenization
    ↓
Embedding Layer (256-dim)
    ↓
LSTM Layers (2 layers, 512 hidden)
    ↓
Transformer Encoder (4 layers, 8 heads)
    ↓
Output Layer (vocab_size)
    ↓
Advanced Inference (Beam Search / Temperature Sampling)
    ↓
Generated Response
```

### Model Configuration
- **Embedding Dimension**: 256
- **Hidden Dimension**: 512
- **LSTM Layers**: 2 (bidirectional)
- **Transformer Layers**: 4
- **Attention Heads**: 8
- **Dropout**: 0.1

---

## 📁 Project Structure

```
contextflow-chatbot/
├── backend/
│   ├── app_advanced.py      # Main Flask application
│   ├── app_hybrid.py        # Hybrid model integration
│   ├── inference.py         # Advanced inference strategies
│   ├── context_manager.py   # Conversation context management
│   ├── templates/           # HTML templates
│   │   └── index.html       # Web interface
│   ├── static/              # CSS/JS/Images
│   └── requirements.txt     # Python dependencies
├── models/
│   ├── chatbot_model.py     # PyTorch model definition
│   ├── tokenizer.pkl        # Trained tokenizer
│   └── checkpoints/         # Model weights (not in repo)
├── training/
│   ├── training.py          # Training pipeline
│   └── dataset.py           # Dataset utilities
├── utils/
│   └── tokenizer.py         # Custom tokenizer implementation
├── data/                    # Training data (not in repo)
├── ContextFlow_Project.ipynb # Jupyter notebook demo
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose setup
└── README.md               # This file
```

---

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster inference

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/neopentane7/contextflow-chatbot.git
   cd contextflow-chatbot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r backend/requirements.txt
   ```

---

## 💻 Usage

### Running Locally

1. **Start the Flask server**
   ```bash
   python backend/app_advanced.py
   ```

2. **Open your browser**
   Navigate to: `http://localhost:5000`

3. **Start chatting!**
   The web interface provides an intuitive chat experience with real-time responses.

### Running with Docker

1. **Build and run the container**
   ```bash
   docker-compose up --build
   ```

2. **Access the application**
   Open `http://localhost:5000` in your browser

3. **Stop the container**
   ```bash
   docker-compose down
   ```

---

## 🎓 Training

### Training Data

Due to file size limitations, training data is **not included** in this repository.

To train the model, you'll need conversation datasets such as:
- [Cornell Movie Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
- [Ubuntu Dialogue Corpus](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)
- Custom conversation datasets

### Data Preparation

1. **Download datasets** and place them in the `data/` directory:
   ```
   data/
   ├── cornell/
   │   └── movie_lines.tsv
   └── ubuntu/
       └── dialogueText.csv
   ```

2. **Merge and preprocess data**
   ```bash
   python mergedata.py
   ```

3. **Train the model**
   ```bash
   python training/training.py
   ```

### Training Configuration

Edit `training/training.py` to customize:
- Number of epochs
- Batch size
- Learning rate
- Model architecture parameters

### Monitoring Training

Training progress is logged to console and saved checkpoints are stored in `models/checkpoints/`.

---

## 📡 API Documentation

### Endpoint: `POST /api/chat`

Send a message to the chatbot and receive a response.

#### Request

```json
{
  "message": "What is machine learning?",
  "session_id": "user_123",
  "method": "beam_search"
}
```

**Parameters:**
- `message` (string, required): User's input message
- `session_id` (string, optional): Session identifier for context tracking
- `method` (string, optional): Inference method (`beam_search`, `temperature_sampling`, `greedy`)

#### Response

```json
{
  "bot_response": "Machine learning is a field of AI where computers learn from data without explicit programming.",
  "session_id": "user_123"
}
```

#### Example using cURL

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello!",
    "method": "beam_search"
  }'
```

#### Example using Python

```python
import requests

response = requests.post(
    'http://localhost:5000/api/chat',
    json={
        'message': 'What is deep learning?',
        'method': 'beam_search'
    }
)

print(response.json()['bot_response'])
```

---

## 🔬 Inference Methods

### 1. Beam Search
- **Best for**: High-quality, coherent responses
- **Speed**: Slower but more accurate
- **Parameters**: `beam_width=5`, `temperature=0.7`

### 2. Temperature Sampling
- **Best for**: Creative, diverse responses
- **Speed**: Fast
- **Parameters**: `temperature=0.8`, `top_k=50`, `top_p=0.9`

### 3. Greedy Decoding
- **Best for**: Fast, deterministic responses
- **Speed**: Fastest
- **Use case**: Production environments with latency constraints

---

## 🧪 Testing

### Interactive Testing

Run the command-line interface:
```bash
python modeling.py
```

### Jupyter Notebook

Explore the full project in the included notebook:
```bash
jupyter notebook ContextFlow_Project.ipynb
```

### Unit Tests

Run the test suite:
```bash
python backend/test_advanced.py
```

---

## 🛠️ Technologies Used

- **Deep Learning**: PyTorch
- **NLP**: Transformers, Custom Tokenizer
- **Backend**: Flask
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Docker, Docker Compose
- **Database**: SQLite (for conversation storage)

---

## 📊 Performance

- **Inference Speed**: ~100-500ms per response (CPU)
- **Inference Speed**: ~50-150ms per response (GPU)
- **Model Size**: ~50MB (compressed)
- **Context Window**: Up to 15 previous conversation turns

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Cornell Movie Dialogs Corpus for training data
- Ubuntu Dialogue Corpus for conversational data
- PyTorch team for the excellent deep learning framework
- Hugging Face for Transformers library

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

<div align="center">

**Made with ❤️ using PyTorch and Flask**

⭐ Star this repo if you find it helpful!

</div>
