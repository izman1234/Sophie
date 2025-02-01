from chat import ChatModel, QuitException
from discord_wrapper import DiscordClient
from dotenv import load_dotenv
from groq import Groq
import settings as s
import warnings
import discord
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

global main_state
AI_name = "Sophie"
discord_token = os.environ['DISCORD_KEY']

s.global_init()

async def main():
    try:
        main_state = "CONVERSATION"
        chat = await ChatModel.create(AI_name)
        while True:
            match main_state:
                case "CONVERSATION":
                    if s.voice_activated:
                        user_input, main_state = await chat.record_audio()
                    else:
                        # Get user input
                        user_input = input("[YOU]:\n")
                    if user_input != None:
                        assistant_reply = await chat.have_conversation(user_input)
                        print(f"[{AI_name}]:\n", assistant_reply)
                        if s.voice_activated:
                            await chat.speech(assistant_reply)
                case "COMMAND":
                    # Temporary for now until there is something to actually do in this state
                    print("Command State")
                    main_state = "CONVERSATION"

    except QuitException: # Exit loop if user wants to end program
        pass

def run_discord_mode():
    intents = discord.Intents.default()
    intents.message_content = True
    discord_client = DiscordClient(intents=intents)
    discord_client.initialize(s.model_name, AI_name)
    discord_client.run(discord_token)

if __name__ == "__main__":
    if s.using_discord:
        run_discord_mode()
    else:
        asyncio.run(main())

    s.global_save()