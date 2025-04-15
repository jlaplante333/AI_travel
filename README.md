# 🌍 AI Travel Agent

An interactive AI-powered travel planning assistant that generates personalized itineraries, AI voiceovers, destination images, and cinematic videos — all from a single prompt.

![Screenshot 2025-02-16 161217](https://github.com/user-attachments/assets/8575e242-3413-446d-b74d-74c2ea75ba37)


---

## ✨ Features

- 🗺️ **AI Itinerary Generator** – Custom travel plans using LLaMA 3 via Ollama.
- 🔊 **Voiceover Creation** – Google TTS for immersive travel narration.
- 🖼️ **Image Generation** – Beautiful visuals using DALL·E-style image tools.
- 🎞️ **Video Generator** – Generate short cinematic travel videos using RunwayML.
- 🤖 **Chat Agent Ready** – Easily extendable with LangChain Agents for booking, weather, etc.

---

## 🛠️ Built With

- **Frontend**: HTML, CSS (Bootstrap, custom styles)
- **Backend**: Python + Flask
- **AI Tools**: Ollama, OpenAI GPT-4o, Google TTS
- **Media APIs**: LumaAI API for videos, OpenAI DALL-E API for images, Google Cloud

---

## 📁 Project Structure

```bash
AI-Travel-Agent/
│
├── app.py                 # Flask backend
├── config.txt             # API keys and config (keep private!)
├── load_config.py         # Environment loader
│
├── templates/
│   └── index.html         # Frontend layout
│
├── static/
│   ├── styles.css         # UI styles
│         
│
└── README.md              # You're here!

🚀 Getting Started
✅ Prerequisites
Python 3.9+

Git

A modern browser

📦 Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/Anantaverma20/AI-Travel-Agent.git
cd AI-Travel-Agent
Create a virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # macOS/Linux
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Set up API Keys:

Edit config.txt with your keys:

makefile
Copy
Edit
GOOGLE_TTS_KEY=
OLLAMA_MODEL=llama3
OPENAI_API_KEY=
RUNWAY_API_KEY=
LUMA_API_KEY=
GEMINI_API_KEY=
Run the app:

bash
Copy
Edit
python app.py
The app will launch on http://127.0.0.1:5000/

🖼️ Screenshots
📍 Home & Prompt Input

🗺️ AI Itinerary Output + Voice

📌 Future Improvements
 Integrate chatbot using LangChain or Gemini

 Enable real hotel/flight searches via Amadeus or Skyscanner APIs

 Offline itinerary downloads (PDF + video)

 User login to save past trips

🤝 Contribution Guide
Fork the repo

Create your feature branch (git checkout -b feature/your-feature)

Commit your changes (git commit -m 'Add new feature')

Push to the branch (git push origin feature/your-feature)

Open a Pull Request!
