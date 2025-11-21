# ContextFlow Hybrid Chatbot

A sophisticated hybrid chatbot system combining **LSTM** and **Transformer** architectures with a **Flask** backend and a modern **web interface**.

## Features

- **Hybrid Model**: Combines LSTM for sequential processing and Transformer for attention mechanisms.
- **Advanced Inference**: Supports Beam Search, Temperature Sampling, and Constrained Decoding.
- **Context Awareness**: Maintains conversation history and context.
- **Web Interface**: Clean, responsive UI for interacting with the bot.
- **Docker Support**: Easy deployment with Docker and Docker Compose.

## Project Structure

```
.
├── backend/
│   ├── app_advanced.py    # Main Flask application
│   ├── inference.py       # Advanced inference logic
│   ├── templates/         # HTML templates
│   └── static/            # CSS/JS assets
├── models/
│   ├── chatbot_model.py   # PyTorch model definition
│   └── checkpoints/       # Saved model weights
├── training/
│   └── training.py        # Training pipeline
├── utils/
│   └── tokenizer.py       # Custom tokenizer
├── Dockerfile
└── docker-compose.yml
```

## Installation

1.  **Clone the repository**
2.  **Create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```
3.  **Install dependencies**
    ```bash
    pip install -r backend/requirements.txt
    ```

## Usage

### Running Locally

1.  Start the server:
    ```bash
    python backend/app_advanced.py
    ```
2.  Open your browser and navigate to:
    `http://localhost:5000`

### Running with Docker

1.  Build and run:
    ```bash
    docker-compose up --build
    ```
2.  Access the application at `http://localhost:5000`.

## Training

To retrain the model:

```bash
python training/training.py
```

Ensure your data is in `data/merged_training_data.csv`.

## API Documentation

### `POST /api/chat`

Send a message to the bot.

**Payload:**
```json
{
    "message": "Hello",
    "session_id": "user_123",
    "method": "beam_search"
}
```

**Response:**
```json
{
    "bot_response": "Hello! How can I help you?",
    "session_id": "user_123"
}
```
