from PyCharacterAI.exceptions import SessionClosedError
from memory_manager import MemoryManager
from PyCharacterAI import get_client
from playsound import playsound
import speech_recognition as sr
from cartesia import Cartesia
from groq import Groq
import numpy as np
import datetime
import asyncio
import time
import wave
import os

class QuitException(Exception): pass

class ChatModel():

    def __init__(self, model_name, AI_name, using_discord):
        self.model_name = model_name
        self.memory_manager = None
        self.last_boot_timestamp = None
        self.chat_model = None
        self.character_id = None
        self.long_term_memory = None
        self.chat = None
        self.turn_id = None
        self.candidate_id = None
        self.using_discord = using_discord
        self.greeting_message = None
        self.audio_state = "LISTEN_FOR_KEYWORD"
        self.audio_frames = []
        self.speech_client = None

    @classmethod
    async def create(cls, model_name, AI_name, using_discord):
        self = cls(model_name, AI_name, using_discord)
        self.memory_manager = MemoryManager(model_name)
        self.speech_client = Cartesia(api_key=os.environ.get("CARTESIA_KEY"))
        self.last_boot_timestamp = self.memory_manager.get_last_boot_timestamp()
        if(self.model_name == "groq"):
            self.chat_model = Groq(api_key=os.environ['GROQ_KEY'])
        elif(self.model_name == 'c.ai'):
            self.chat_model = await get_client(token=os.environ['CHARACTER_KEY'])
            self.character_id = "1j0SUtg4YHXM1HkKYFq9gX69ObZtrS5cEOt-wTOnvWU"
            if os.path.exists("c.ai_chat_id.txt"):
                with open("c.ai_chat_id.txt", "r") as f:
                    fetched_chat = f.readline().strip()
                    self.last_message = f.read().strip()
                if not self.using_discord:
                    print(f"[Sophie]: {self.last_message}")
                self.chat = await self.chat_model.chat.fetch_chat(fetched_chat)
            else:
                self.chat, greeting_message = await self.chat_model.chat.create_chat(self.character_id)
                self.greeting_message = greeting_message
                if not self.using_discord:
                    print(f"[Sophie]: {greeting_message.get_primary_candidate().text}")
        elif(self.model_name == 'chatgpt'):
            pass

        # Check if last_boot_timestamp exists and generate a message for Sophie
        if(self.model_name == "groq"):
            if self.last_boot_timestamp:
                time_diff = datetime.datetime.now() - self.last_boot_timestamp
                minutes_since_last_boot = int(time_diff.total_seconds() // 60)  # Convert seconds to minutes
                time_message = [
                    {"role": "system", "content": f"Sophie, you were shutdown for {minutes_since_last_boot} minutes since you were last booted up. Please mention something about it briefly"}
                ]

                # Generate a response
                time_diff_reply = self.chat_model.chat.completions.create(
                    messages= time_message,
                    model="llama-3.3-70b-versatile",
                    max_tokens=75,
                    stream=False,
                )

                # Get and display the assistant's response
                assistant_reply = time_diff_reply.choices[0].message.content
                self.greeting_message = assistant_reply
                if not self.using_discord:
                    print(f"[{AI_name}]:\n", assistant_reply)
        return self
    
    async def send_greeting_message(self):
        return self.greeting_message

    async def close(self):
        if self.model_name == 'c.ai' and hasattr(self.chat_model, "session"):
            await self.chat_model.close_session()
    
    async def have_conversation(self, user_input, username="User"):
        if user_input.lower() in ['exit', 'quit'] and self.using_discord == False:
            try:
                if(self.model_name == "c.ai"):
                    with open("c.ai_chat_id.txt", "w") as f:
                        f.write(str(self.chat.chat_id))
                        f.write('\n')
                        f.write(self.last_message)
                elif(self.model_name == "groq"):
                    # Save short memory to a file
                    self.memory_manager.force_save_ltm()
                    self.memory_manager.force_save_stm()
                    self.memory_manager.force_save_embeddings()
            except Exception as e:
                print(f"Error saving memory: {e}")
            
            # Close aiohttp session
            try:
                await self.close()
            except Exception as e:
                print(f"Error during cleanup: {e}")
            
            print("Exiting the chat.")
            raise QuitException
        else:
            try:
                if(self.model_name == "groq"):
                    # Generate a response
                    chat_completion = self.chat_model.chat.completions.create(
                        messages= self.memory_manager.get_relevant_memories(user_input),
                        model="llama-3.3-70b-versatile",
                        max_tokens=75,
                        stream=False,
                    )

                    # Get and display the assistant's response
                    assistant_reply = chat_completion.choices[0].message.content
                    self.memory_manager.save_reply(user_input, assistant_reply, username)
                
                elif(self.model_name == "c.ai"):
                    answer = await self.chat_model.chat.send_message(self.character_id, self.chat.chat_id, user_input, streaming=True)

                    assistant_reply = ""
                    printed_length = 0
                    async for message in answer:
                        if printed_length == 0:
                            self.turn_id = message.turn_id
                            self.candidate_id = message.get_primary_candidate().candidate_id

                        text = message.get_primary_candidate().text
                        assistant_reply += text[printed_length:]  # Accumulate the text
                        printed_length = len(text)
                    self.last_message = assistant_reply

                return assistant_reply
            except Exception as e:
                print(f"Error during model response generation: {e}")
    
    async def speech(self, text=None):
        sound_file = "voice.mp3"
        
        if(self.model_name == 'c.ai'):
            chat_id = self.chat.chat_id
            voice_id = "9b961615-c541-496e-befd-d20b78c11167"

            speech = await self.chat_model.utils.generate_speech(chat_id, self.turn_id, self.candidate_id, voice_id)

            with open(sound_file, 'wb') as f:
                f.write(speech)

        elif(self.model_name == 'groq' and text != None):
            # This is temporary code until I found out how I want to do TTS. The voice is pretty realistic though
            data = self.speech_client.tts.bytes(
            model_id="sonic-english",
            transcript=text,
            voice_id="f9836c6e-a0bd-460e-9d3c-f7299fa60f94",
            output_format={
                "container": "mp3",
                "encoding": "pcm_f32le",
                "sample_rate": 44100,
            },
            )

            with open(sound_file, "wb") as f:
                f.write(data)
        
        if(text != None):
            try:
                playsound(sound_file)
                os.remove(sound_file)
            except FileNotFoundError:
                print("File not found.")
    
    async def record_audio(self, wake_word="hey sophie", output_file="recorded_audio.wav"):
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()

        text = None

        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            recognizer.pause_threshold = 2.0
            recognizer.non_speaking_duration = 1.5
            try:
                match self.audio_state:
                    case "LISTEN_FOR_KEYWORD":
                        #print("Listening for the wake word...")
                        while True:
                            # Listen for audio input
                            audio = recognizer.listen(source)

                            # Convert audio to text
                            input = recognizer.recognize_google(audio).lower()

                            if input in ['exit', 'quit']:
                                return "quit"
                        
                            # Check for the wake word
                            if wake_word in input:
                                print(f"Wake word detected")
                                self.audio_state = "RECORD"
                                break
                    case "RECORD":
                        while True:
                            print("Starting recording...")
                            audio = recognizer.listen(source)
                            raw_data = audio.get_raw_data()
                            self.audio_frames.append(raw_data)

                            print("Stopping recording...")

                            input = recognizer.recognize_google(audio).lower()
                            print(f"[User]: {input}")

                            if "bye sophie" in input:
                                self.audio_state = "LISTEN_FOR_KEYWORD"
                                return None
                            else:
                                text = input
                                break

            except sr.UnknownValueError:
                    # Ignore unrecognized audio
                    pass
            except sr.RequestError as e:
                print(f"Error with speech recognition service: {e}")

        return text