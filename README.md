# Sophie AI Chatbot

Sophie is an advanced, interactive chatbot that combines natural language understanding, memory management, and speech capabilities. The system supports text-based interactions, voice communication, and integration with Discord for a seamless user experience.

## Features

1. **Interactive Chat:**
   - Handles conversations with a focus on natural, human-like responses.
   - Supports different models, such as Groq and Character.AI.

2. **Memory Management:**
   - Uses short-term and long-term memory to retain context and improve interactions.
   - Clusters long-term memory using embeddings for contextually relevant recall.

3. **Speech Integration:**
   - Generates audio responses for text inputs.
   - Records user audio triggered by a wake word ("Hey Sophie").

4. **Discord Integration:**
   - Responds to messages in a Discord channel.
   - Supports commands like `/quit` to gracefully shut down the chatbot.

## Requirements

- Python 3.12.8
- Virtual environment with required dependencies installed.

### Dependencies
Install dependencies with:
```bash
pip install -r requirements.txt
```

Key libraries include:
- `PyCharacterAI` for Character.AI integration
- `Groq` for AI API code base
- `Cartesia` for TTS
- `SpeechRecognition` for voice input
- `pyaudio` for audio recording
- `playsound` for audio playback
- `discord.py` for Discord bot integration
- `sentence-transformers` for embedding generation
- `scikit-learn` for clustering


## Setup and Usage

1. **Clone the Repository:**
```bash
git clone <repository_url>
cd <repository_name>
```

2. **Configure Settings:**
   - Update API keys in the `ChatModel` class for Character.AI and Groq.
   - Set the desired Discord channel name and token in `discord_wrapper.py`.

3. **Run the Application:**
   - For Discord bot integration, set `using_discord = True` in `main.py`
   - To turn on TTS, set `voice_activated = True` in `main.py`
   - ```bash
     python main.py
     ```

## File Structure

- **`chat.py`**
  Contains the `ChatModel` class, responsible for managing conversations, memory integration, and audio interactions.

- **`memory_manager.py`**
  Implements short-term and long-term memory functionality, including clustering with KMeans.

- **`discord_wrapper.py`**
  Manages Discord bot functionality, including message handling and channel integration.

- **`main.py`**
  Entry point for running the chatbot either in the terminal or through Discord.

## Key Functionalities

### TTS and Microphone Input
- Sophie listens for the wake word ("Hey Sophie") using the `record_audio` method.
- Records audio until silence is detected, then responds and records audio again.
- Sophie will go back to listening for the wake word if told "Bye Sophie"
- Close the program by saying "quit" or "exit"

### Memory Management
- Uses embeddings to organize and recall memories.
- Dynamically reclusters long-term memory as it grows.

### Discord Commands
- `/quit`: Shuts down the bot and saves its state.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with detailed explanations.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or support, open an issue on GitHub or contact the repository owner.

