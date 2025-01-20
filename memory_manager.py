from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from memoripy import EmbeddingModel
from sklearn.cluster import KMeans
import numpy as np
import datetime
import json
import os

starting_message = "You are an AI named Sophie. Your responses should be as human-like as possible, but keep the responses under 50 tokens."

class HuggingFaceEmbeddedModel(EmbeddingModel):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()

    def get_embedding(self, text: str):
        """ Generate an embedding for a given text. """

        return self.model.encode(text, normalize_embeddings=True).tolist()

    def initialize_embedding_dimension(self) -> int:
        """ Initialize and return the embedding dimension. """
        return self.embedding_dimension


class MemoryManager():

    def __init__(self, model_name):
        self.model_name = model_name
        self.embedding_model = HuggingFaceEmbeddedModel()
        self.short_term_memory = []
        self.long_term_memory = []
        self.time_message = []
        self.last_boot_timestamp = None
        self.embeddings = None

        self.initialize_memory()

    def initialize_memory(self):
        # Load memory from files
        try:
            if os.path.exists(f"{self.model_name}_stm.json"):
                with open(f"{self.model_name}_stm.json", "r") as f:
                    # Attempt to load the short-term memory data
                    self.short_term_memory = json.load(f)
                if {"role": "system", "content": starting_message} not in self.short_term_memory:
                    self.short_term_memory.insert(0, {"role": "system", "content": starting_message})
            else:
                # Initialize memory
                self.short_term_memory = [
                    {"role": "system", "content": starting_message},
                ]

            self.load_long_term_memory()
            self.load_embeddings()

        except Exception as e:
            print(f"Error loading memory: {e}")
            self.short_term_memory = [
                {"role": "system", "content": starting_message},
            ]

    def load_embeddings(self):
        if os.path.exists(f"{self.model_name}_embeddings.npy"):
            self.embeddings = np.load(f"{self.model_name}_embeddings.npy")
        else:
            # Initialize an empty array with the correct shape for 384-dimensional embeddings
            self.embeddings = np.empty((0, 384))

    def load_long_term_memory(self):
        try:
            # Load memory content
            if os.path.exists(f"{self.model_name}_ltm.json"):
                with open(f"{self.model_name}_ltm.json", "r") as f:
                    self.long_term_memory = json.load(f)

                    # Find the most recent timestamp in long-term memory
                    if self.long_term_memory:
                        last_memory = self.long_term_memory[-1]  # Get the last memory
                        # Find the timestamp embedded in the content
                        timestamp_str = last_memory.get('time', '')
                        if timestamp_str:
                            # Convert the timestamp string back into a datetime object
                            self.last_boot_timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")  # Convert string to datetime
                    else:
                        self.last_boot_timestamp = None
        except Exception as e:
            print(f"Error loading long-term memory: {e}")
            self.last_boot_timestamp = None
    
    def get_last_boot_timestamp(self):
        return self.last_boot_timestamp
    
    # Function to get embedding
    def get_embedding(self, text):
        return self.embedding_model.get_embedding(text)
    
    def add_to_memory(self, user_input, assistant_reply, username="User"):
        """Add a user-assistant interaction to long-term memory."""
        # Compute the embedding for the memory
        embedding = self.get_embedding(f"[{username}]: {user_input} [Assistant]: {assistant_reply}")

         # Determine cluster assignment (if clusters exist)
        cluster_id = -1
        if hasattr(self, 'kmeans') and self.kmeans:  # Ensure clustering model exists
            cluster_id = self.kmeans.predict([embedding])[0]

        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create a memory object without the embedding
        memory = {
            "role": "system",  # Optional, categorize if needed
            "content": f"[{username}]: {user_input} [Assistant]: {assistant_reply}",
            "time": timestamp,  # Save the timestamp in a separate field
            "cluster": cluster_id
        }

        # Add the memory to the long-term memory list
        self.long_term_memory.append(memory)

        # Save the embedding to the embeddings file
        self.save_embeddings(embedding)
    
    def save_embeddings(self, embedding, batch_size=10):
        try:
            # Ensure embedding is converted to a 2D array for stacking
            embedding = np.array(embedding).reshape(1, -1)
            
            if self.embeddings.size == 0:
                self.embeddings = embedding  # Directly assign the first embedding
            else:
                self.embeddings = np.vstack([self.embeddings, embedding])
            
            if len(self.embeddings) % batch_size == 0:
                np.save(f"{self.model_name}_embeddings.npy", self.embeddings)
        except Exception as e:
            print(f"Error saving embedding: {e}")

    # Save function for long-term memory
    def save_long_term_memory(self, batch_size=10):
        """Save the long-term memory to a JSON file."""
        try:
            if len(self.long_term_memory) % batch_size == 0:
                with open(f"{self.model_name}_ltm.json", "w") as f:
                    json.dump(self.long_term_memory, f, indent=4)
        except Exception as e:
            print(f"Error saving long-term memory: {e}")
    
    def force_save_ltm(self):
        try:
            with open(f"{self.model_name}_ltm.json", "w") as f:
                json.dump(self.long_term_memory, f, indent=4)
        except Exception as e:
            print(f"Error saving long-term memory: {e}")
    
    def force_save_stm(self):
        try:
            with open(f"{self.model_name}_stm.json", "w") as f:
                json.dump(self.short_term_memory, f, indent=4)
        except Exception as e:
            print(f"Error saving long-term memory: {e}")
    
    def force_save_embeddings(self):
        try:
            np.save(f"{self.model_name}_embeddings.npy", self.embeddings)
        except Exception as e:
            print(f"Error saving embedding: {e}")

    # Save function for short-term memory
    def save_short_term_memory(self):
        """Save the short-term memory to a JSON file."""
        try:
            with open(f"{self.model_name}_stm.json", "w") as f:
                json.dump(self.short_term_memory, f, indent=4)
        except Exception as e:
            print(f"Error saving short-term memory: {e}")

    # Function to detect context and retrieve relevant long-term memory
    def get_relevant_memories(self, user_input):
        """Retrieve the top 6 most relevant memories based on embedding similarity."""
        user_embedding = self.get_embedding(user_input)
        relevant_memories = []

        # Ensure clusters are available
        if not any("cluster" in memory for memory in self.long_term_memory) or self.embeddings.size == 0:
            print("No clusters found. Consider running `cluster_memories` first.")
            # Add user input to conversation history
            self.short_term_memory.append({"role": "user", "content": user_input})

            # No embeddings exist, return an only short term memory
            return self.short_term_memory[-7:]
        
        # Construct combined memory strings from short-term memory
        short_term_contents = set()
        for i in range(0, len(self.short_term_memory)):  # Iterate in pairs (user, assistant)
            if (self.short_term_memory[i]["role"] != "system"):
                short_term_contents.add(self.short_term_memory[i]["content"].replace("This is a recent memory: ", ""))

        # Iterate over memories and compute similarity
        cluster_scores = {}
        for i, memory in enumerate(self.long_term_memory):
            cluster_id = memory["cluster"]
            if i >= len(self.embeddings):  # Ensure embedding index matches memory index
                print("embeddings length does not match long term memory length")
                break
            memory_embedding = self.embeddings[i]
            similarity = np.dot(user_embedding, memory_embedding) / (
                np.linalg.norm(user_embedding) * np.linalg.norm(memory_embedding)
            )
            if cluster_id not in cluster_scores:
                cluster_scores[cluster_id] = []
            cluster_scores[cluster_id].append(similarity)
        
        # Average similarity for each cluster
        cluster_scores = {k: np.mean(v) for k, v in cluster_scores.items()}
        top_cluster = max(cluster_scores, key=cluster_scores.get)

        # Retrieve memories from the top cluster and exclude memories that are in short-term memory
        for memory in self.long_term_memory:
            if memory["cluster"] == top_cluster and memory["content"] not in short_term_contents:
                relevant_memories.append(memory)
        
        # Sort by similarity within the top cluster
        relevant_memories.sort(
            key=lambda mem: np.dot(
                user_embedding,
                self.embeddings[self.long_term_memory.index(mem)]
            ),
            reverse=True
        )

        # Format the top 6 memories for sending to the model
        formatted_memories = []
        current_time = datetime.datetime.now()
        for memory in relevant_memories[:6]:
            # Calculate the time difference
            timestamp = memory.get("time", "")
            time_diff_minutes = "unknown"
            if timestamp:
                try:
                    memory_time = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    time_diff_minutes = int((current_time - memory_time).total_seconds() // 60)
                except Exception as e:
                    print(f"Error parsing memory timestamp: {e}")

            # Format the memory
            time_label = f"{time_diff_minutes} minutes ago:" if time_diff_minutes != "unknown" else "an unknown time ago"
            formatted_memory = (
                f"This is a memory from {time_label}: {memory['content']}"
            )
            formatted_memories.append({"role": memory["role"], "content": formatted_memory})

        # Add user input to conversation history

        # Combine short-term memory (last 6 messages) + relevant long-term memory + starting message + user message
        combined_history = formatted_memories
        combined_history.extend(self.short_term_memory)
        combined_history.append({"role": "user", "content": user_input})

        return combined_history
    
    def save_reply(self, user_input, assistant_reply, username="User"):

        # Add the interaction to short term memory
        interaction = {"role": "system", "content": f"This is a recent memory: [{username}]: {user_input} [Assistant]: {assistant_reply}"}

        self.short_term_memory.append(interaction)

        # Store information in long-term memory
        self.add_to_memory(user_input, assistant_reply, username)
        self.check_and_recluster()

        # Manage short-term memory size
        if len(self.short_term_memory) > 6:  # Adjust the number to fit the model's token limit
            # Check if the system message is already in memory, if so, preserve it
            
            # Trim other messages, keeping system message intact
            self.short_term_memory = self.short_term_memory[-6:]  # Keep the last 6 user-assistant pairs and the system message

            # Ensure system message is in memory and preserve it
            if {"role": "system", "content": starting_message} not in self.short_term_memory:
                self.short_term_memory.insert(0, {"role": "system", "content": starting_message})  # Add it to the front if missing
    
    def cluster_memories(self, n_clusters):
        """Cluster long-term memory embeddings into categories."""
        if len(self.embeddings) < n_clusters:
            print("Not enough embeddings for the specified number of clusters.")
            return
        
        # Fit KMeans on the embeddings
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.embeddings)
        
        # Attach cluster labels to long-term memory
        for i, memory in enumerate(self.long_term_memory):
            memory["cluster"] = int(cluster_labels[i])
        
        # Save cluster information
        self.save_long_term_memory()
        print(f"Memories clustered into {n_clusters} categories.")

    def get_memory_clusters(self):
        """Retrieve memories grouped by cluster."""
        clusters = {}
        for memory in self.long_term_memory:
            cluster_id = memory.get("cluster", -1)  # Default to -1 if not clustered
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(memory)
        return clusters

    def find_optimal_clusters(self, max_k):
        """Find the optimal number of clusters using silhouette score."""
        if len(self.embeddings) < 2:
            print("Not enough embeddings to cluster.")
            return 0

        best_score = -1
        best_k = 2
        for k in range(2, min(max_k, len(self.embeddings)) + 1):
            kmeans = KMeans(n_clusters=k, random_state=42).fit(self.embeddings)
            score = silhouette_score(self.embeddings, kmeans.labels_)
            if score > best_score:
                best_score = score
                best_k = k
        print(f"Optimal number of clusters: {best_k} (Silhouette Score: {best_score:.2f})")
        return best_k
    
    def check_and_recluster(self):
        """Check if reclustering is needed and update clusters."""
        if len(self.long_term_memory) % max(50, len(self.long_term_memory) // 10) == 0:  # Every 10% growth or 50 entries
            print("Checking optimal clusters and updating...")
            optimal_clusters = self.find_optimal_clusters(25)
            self.cluster_memories(optimal_clusters)