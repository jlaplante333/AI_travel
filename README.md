# 🌍 AI Travel Agent

An interactive AI-powered travel planning assistant that generates personalized itineraries, AI voiceovers, destination images, and cinematic videos — all from a single prompt.

![Screenshot 2025-02-16 161217](https://github.com/user-attachments/assets/8575e242-3413-446d-b74d-74c2ea75ba37)


---

## ✨ Features

- 🗺️ **AI Itinerary Generator** – Custom travel plans using LLaMA 3 via Ollama.
- 🔊 **Voiceover Creation** – Google TTS for immersive travel narration.
- 🖼️ **Image Generation** – Beautiful visuals using DALL·E-style image tools.
- 🎞️ **Video Generator** – Generate short cinematic travel videos using LumaAI.
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
├── location_agent.py      # Location extraction and processing
├── static/
│   ├── css/               # Stylesheets
│   ├── js/                # JavaScript files
│   └── images/            # Generated images
│
├── templates/
│   └── index.html         # Frontend layout
│
└── README.md              # You're here!
```

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.9+
- Git
- A modern browser

### 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jlaplante333/AI_travel_world_fair.git
   cd AI_travel_world_fair
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API Keys**:
   - Create a `.env` file in the project root with your keys:
     ```
     GOOGLE_MAPS_API_KEY=your_google_maps_api_key
     OPENAI_API_KEY=your_openai_api_key
     LUMA_API_KEY=your_luma_api_key
     ```

5. **Run the app**:
   ```bash
   python app.py
   ```
   The app will launch on `http://127.0.0.1:5000/`


---

## 📌 Future Improvements

- [ ] Integrate chatbot using LangChain or Gemini
- [ ] Enable real hotel/flight searches via Amadeus or Skyscanner APIs
- [ ] Offline itinerary downloads
- [ ] User login to save past trips

---

## 🤝 Contribution Guide

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request!


---

## 👤 Author

- **Jean-Laurent Plante**
- [GitHub](https://github.com/jlaplante333)

---

## 📌 Environment Variables

The following environment variables are required:

- `GOOGLE_MAPS_API_KEY`: Your Google Maps API key for geocoding and maps
- `OPENAI_API_KEY`: Your OpenAI API key for image generation
- `LUMA_API_KEY`: Your Luma AI API key for video generation

Create a `.env` file in the project root and add these variables. Never commit the `.env` file to version control.

## 📌 Security Notes

- Never commit API keys or sensitive credentials to version control
- Keep your `.env` file secure and local
- Use environment variables for all sensitive configuration
- Regularly rotate your API keys
