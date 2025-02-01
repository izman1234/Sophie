import json
import os

GLOBAL_FILE = "globals.json"

def global_init():
    """Initialize global variables from a file or set default values if the file does not exist."""
    global model_name # Options are: groq or c.ai
    global using_discord
    global voice_activated
    global quality_voice

    # Default values
    defaults = {
        "model_name": "groq",
        "using_discord": False,
        "voice_activated": True,
        "quality_voice": False
    }

    # Load values from file if it exists
    if os.path.exists(GLOBAL_FILE):
        try:
            with open(GLOBAL_FILE, "r") as f:
                data = json.load(f)
                model_name = data.get("model_name", defaults["model_name"])
                using_discord = data.get("using_discord", defaults["using_discord"])
                voice_activated = data.get("voice_activated", defaults["voice_activated"])
                quality_voice = data.get("quality_voice", defaults["quality_voice"])
        except Exception as e:
            print(f"Error loading globals: {e}. Using default values.")
            model_name, using_discord, voice_activated, quality_voice = defaults.values()
    else:
        # Use default values if no file exists
        model_name, using_discord, voice_activated, quality_voice = defaults.values()

def global_save():
    """Save the global variables to a file."""
    try:
        data = {
            "model_name": model_name,
            "using_discord": using_discord,
            "voice_activated": voice_activated,
            "quality_voice": quality_voice
        }
        with open(GLOBAL_FILE, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Globals saved successfully to {GLOBAL_FILE}")
    except Exception as e:
        print(f"Error saving globals: {e}")