from chat import ChatModel, QuitException
from discord_wrapper import DiscordClient
from dotenv import load_dotenv
from groq import Groq
import discord
import warnings
import asyncio
import sys
import os

warnings.filterwarnings("ignore", 
                        message=".*Found Intel OpenMP.*", 
                        category=RuntimeWarning, 
                        module="threadpoolctl")

warnings.filterwarnings("ignore", 
                        message=".*Torch was not compiled with flash attention.*", 
                        category=UserWarning, 
                        module="transformers.models.bert.modeling_bert")

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

AI_name = "Sophie"
chat_model = "groq"
discord_token = os.environ['DISCORD_KEY']
using_discord = False
voice_activated = True
quality_voice = False

async def main():
    try:
        chat = await ChatModel.create(chat_model, AI_name, using_discord, quality_voice)
        while True:
            if voice_activated:
                user_input = await chat.record_audio()
            else:
            # Get user input
                user_input = input("[YOU]:\n")
            if user_input != None:
                assistant_reply = await chat.have_conversation(user_input)
                print(f"[{AI_name}]:\n", assistant_reply)
                if voice_activated:
                    await chat.speech(assistant_reply)

    except QuitException: # Exit loop if user wants to end program
        pass

def run_discord_mode():
    intents = discord.Intents.default()
    intents.message_content = True
    discord_client = DiscordClient(intents=intents)
    discord_client.initialize(chat_model, AI_name)
    discord_client.run(discord_token)

if __name__ == "__main__":
    if using_discord:
        run_discord_mode()
    else:
        asyncio.run(main())