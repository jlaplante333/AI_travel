import pymongo
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

# Placeholder for Factory API
# FACTORY_API_KEY = "your_factory_api_key_here"
# FACTORY_API_URL = "https://api.factory.example.com/v1/reasoning"

# 1. Simulate getting user preferences from chatbot

def get_user_preferences_from_chatbot():
    # Mocked preferences for demonstration
    return {
        "user_id": "user123",
        "budget": 2000,
        "duration": 5,  # days
        "likes": ["museums", "nature", "local food"],
        "dislikes": ["nightclubs"],
        "dietary_restrictions": ["vegetarian"]
    }

# 2. Connect to MongoDB Atlas
MONGO_URI = "mongodb+srv://ananta:1234@cluster0.j111vp7.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGO_URI)
db = client["travel_ai"]
user_pref_col = db["user_preferences"]
reasoning_col = db["reasoning_traces"]

# 3. Store user preferences
def store_user_preferences(preferences):
    user_pref_col.insert_one(preferences)

# 4. Define the RL environment
class TravelItineraryEnv(gym.Env):
    def __init__(self, user_preferences, activity_pool):
        super().__init__()
        self.user_preferences = user_preferences
        self.activity_pool = activity_pool
        self.current_step = 0
        self.itinerary = []
        # Observation: budget, duration, likes count, dislikes count, itinerary length
        self.observation_space = spaces.Box(
            low=np.array([0, 1, 0, 0, 0]),
            high=np.array([10000, 30, 10, 10, 30]),
            dtype=np.float32
        )
        # Actions: 0 = suggest next, 1 = modify, 2 = finalize
        self.action_space = spaces.Discrete(3)
        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.itinerary = []
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        return np.array([
            self.user_preferences["budget"],
            self.user_preferences["duration"],
            len(self.user_preferences["likes"]),
            len(self.user_preferences["dislikes"]),
            len(self.itinerary)
        ], dtype=np.float32)

    def step(self, action):
        done = False
        reward = 0
        info = {}
        if action == 0:  # suggest next
            if self.current_step < len(self.activity_pool):
                self.itinerary.append(self.activity_pool[self.current_step])
                reward = 1  # Simple reward for adding
            self.current_step += 1
        elif action == 1:  # modify
            reward = 0.5  # Less reward for modifying
        elif action == 2:  # finalize
            done = True
            reward = 2 if len(self.itinerary) >= self.user_preferences["duration"] else -1
        if self.current_step >= len(self.activity_pool):
            done = True
        obs = self._get_obs()
        return obs, reward, done, False, info

# 5. Main RL pipeline
if __name__ == "__main__":
    # Get user preferences (mocked)
    user_prefs = get_user_preferences_from_chatbot()
    store_user_preferences(user_prefs)

    # Define a simple activity pool
    activity_pool = [
        "Visit the city museum",
        "Hiking in the national park",
        "Try local vegetarian cuisine",
        "Boat tour on the river",
        "Explore the art district"
    ]

    # Initialize environment
    env = TravelItineraryEnv(user_prefs, activity_pool)

    # 6. Train PPO model
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=5000)

    # 7. Use trained model to generate actions and reasoning traces
    obs, _ = env.reset()
    done = False
    step_idx = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        # activity = env.itinerary[-1] if env.itinerary else None
        # # Call Factory API for reasoning trace (mocked call)
        # if activity:
        #     prompt = f"Explain why this activity ('{activity}') was chosen given user preferences: {user_prefs}"
        #     # Placeholder for API call
        #     try:
        #         response = requests.post(
        #             FACTORY_API_URL,
        #             headers={"Authorization": f"Bearer {FACTORY_API_KEY}"},
        #             json={"prompt": prompt}
        #         )
        #         reasoning = response.json().get("reasoning", "No reasoning returned.")
        #     except Exception as e:
        #         reasoning = f"API call failed: {e}"
        #     # Store in MongoDB
        #     reasoning_col.insert_one({
        #         "user_id": user_prefs["user_id"],
        #         "step": step_idx,
        #         "activity": activity,
        #         "reasoning": reasoning
        #     })
        step_idx += 1

    print("RL pipeline complete.") 