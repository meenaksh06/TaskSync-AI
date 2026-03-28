# AI Assistant Frontend

A modern, beautiful React + Tailwind CSS frontend for the AI Personal Assistant.

## Features

- 🎨 **Modern UI** - Clean, glassmorphism design with smooth animations
- 🌙 **Dark Mode** - Beautiful dark theme optimized for extended use
- 🎤 **Voice Input** - Record voice messages with real-time visualization
- 💬 **Smart Chat** - Dynamic conversation interface with intent/entity display
- 📱 **Responsive** - Works great on desktop and mobile
- ⚡ **Fast** - Built with Vite for instant hot reload

## Quick Start

1. **Install Node.js** (if not installed):
   ```bash
   # macOS with Homebrew
   brew install node
   
   # Or download from https://nodejs.org
   ```

2. **Install Dependencies**:
   ```bash
   cd frontend
   npm install
   ```

3. **Start Development Server**:
   ```bash
   npm run dev
   ```

4. **Open in Browser**:
   - Frontend: http://localhost:3000
   - Make sure the backend is running on http://localhost:8000

## Build for Production

```bash
npm run build
npm run preview
```

## Tech Stack

- **React 18** - UI library
- **Tailwind CSS** - Utility-first styling
- **Framer Motion** - Smooth animations
- **Lucide React** - Beautiful icons
- **Vite** - Fast build tool

## Project Structure

```
frontend/
├── public/
│   └── vite.svg           # App icon
├── src/
│   ├── components/
│   │   ├── ChatInterface.jsx   # Main chat component
│   │   ├── Message.jsx         # Message bubbles
│   │   ├── Sidebar.jsx         # Navigation sidebar
│   │   ├── TypingIndicator.jsx # Loading animation
│   │   ├── VoiceRecorder.jsx   # Voice recording overlay
│   │   └── WelcomeScreen.jsx   # Initial welcome UI
│   ├── App.jsx            # Root component
│   ├── main.jsx           # Entry point
│   └── index.css          # Global styles
├── index.html             # HTML template
├── package.json           # Dependencies
├── tailwind.config.js     # Tailwind configuration
├── postcss.config.js      # PostCSS config
└── vite.config.js         # Vite configuration
```

## Customization

### Colors
Edit `tailwind.config.js` to change the color scheme:

```javascript
colors: {
  primary: {
    500: '#6366f1', 
  }
}
```

### Animations
Modify animation keyframes in `tailwind.config.js` and `src/index.css`.

## API Integration

The frontend communicates with the FastAPI backend:

- `POST /infer` - Send text messages
- `POST /voice` - Send voice recordings
- `GET /health` - Check API status
- `GET /auth/google/status` - Check Google connection

Configure the API URL in `src/App.jsx`:
```javascript
const API_BASE = 'http://localhost:8000';
```

