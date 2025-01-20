from chat import ChatModel, QuitException
import discord

izman1234_id = 221750115130540032
using_discord = True

class DiscordClient(discord.Client):

    def initialize(self, model_name, AI_name):
        """Custom initialization function for additional setup."""
        self.model_name = model_name
        self.AI_name = AI_name
        self.channel_name = "soph13-channel"

    async def on_ready(self):
        self.chat = await ChatModel.create(self.model_name, self.AI_name, using_discord)

        for guild in self.guilds:
            self.target_channel = discord.utils.get(guild.text_channels, name=self.channel_name)
            if self.target_channel:
                greetings_message = await self.chat.send_greeting_message()
                await self.target_channel.send(greetings_message)
                break
        if not self.target_channel:
            print(f"Channel '{self.channel_name}' not found.")
    
    async def on_message(self, message):
        if message.author == self.user:
            return
        
        # Handle custom commands like /quit
        if message.content.lower() == "/quit" and message.author.id == izman1234_id:
            await message.channel.send("I am shutting down now. Have a nice day.")
            if(self.chat.model_name == "c.ai"):
                    with open("c.ai_chat_id.txt", "w") as f:
                        f.write(str(self.chat.chat_id))
                        f.write('\n')
                        f.write(self.last_message)
            elif(self.chat.model_name == "groq"):
                # Save short memory to a file
                self.chat.memory_manager.force_save_ltm()
                self.chat.memory_manager.force_save_embeddings()
                self.chat.memory_manager.force_save_stm()

            await self.close()  # Gracefully shut down the client and close the connection
            exit(0)  # Optionally, call exit(0) to terminate the program completely
        
        if message.channel.name == self.channel_name:
            assistant_reply = await self.chat.have_conversation(message.content, username=message.author.name)
            await message.channel.send(assistant_reply)
