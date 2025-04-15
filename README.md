# 🌍 AI Travel Agent

An interactive AI-powered travel planning assistant that generates personalized itineraries, AI voiceovers, destination images, and cinematic videos with music — all from a single prompt.

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
- **AI Tools**: Ollama, OpenAI GPT-4o, RunwayML, Gemini API, Google TTS
- **Media APIs**: Pexels (for visuals), RapidAPI (video extension), Google Cloud

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
│   └── screenshots/       # Add screenshots for demo
│
└── README.md              # You're here!
