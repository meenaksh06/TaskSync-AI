# 🍏 TaskSync-AI: The Future of Personal Productivity

TaskSync-AI is a production-grade, multi-modal AI assistant designed to streamline your daily workflow. By combining state-of-the-art NLP with seamless Google Workspace integration and a premium SaaS-inspired interface, TaskSync-AI transforms how you manage meetings, tasks, and communications.

[![Next.js](https://img.shields.io/badge/Next.js-15-black?logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Prisma](https://img.shields.io/badge/Prisma-ORM-2D3748?logo=prisma)](https://www.prisma.io/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-CSS-38B2AC?logo=tailwind-css)](https://tailwindcss.com/)
[![OpenAI Whisper](https://img.shields.io/badge/STT-Whisper-blue?logo=openai)](https://openai.com/research/whisper)

---

## 🚀 Key Features

- **🧠 Intelligent Intent Detection**: Powered by a fine-tuned DistilBERT model achieved ~95% accuracy on scheduling, mailing, and task management intents.
- **🎙️ Voice-First Architecture**: Real-time speech-to-text processing using OpenAI's Whisper-base for hands-free productivity.
- **📅 Google Workspace Integration**: Secure OAuth-based connection to Google Calendar and Gmail for real-time meeting scheduling and invites.
- **💎 Premium Dashboard**: A high-performance, responsive UI built with Next.js and Framer Motion, inspired by industry leaders like Linear and Notion.
- **🔐 Enterprise Security**: Robust authentication via NextAuth.js, environment-driven secret management, and stateless API architecture.
- **🗄️ Relational Persistence**: Reliable data handling using PostgreSQL (via Prisma ORM) for user sessions and conversation history.

---

## 🛠 Tech Stack

### Frontend
- **Framework**: Next.js 15 (App Router)
- **Styling**: Tailwind CSS + Shadcn UI
- **Animations**: Framer Motion
- **State/Auth**: NextAuth.js

### Backend & AI
- **Core API**: FastAPI (Python 3.12)
- **NLP**: Transformers (DistilBERT), spaCy (NER)
- **STT**: OpenAI Whisper
- **ORM**: Prisma (SQLite for local, PostgreSQL for Production)

---

## 🏗 Architecture

TaskSync-AI utilizes a hybrid architecture to balance high-performance ML inference with modern web capabilities:
1. **Next.js Utility Layer**: Handles user authentication, dashboard rendering, and proxies requests.
2. **FastAPI Engine**: Dedicated high-speed inference server for specialized NLP and STT models.
3. **Integration Layer**: Direct secure hooks into Google APIs for real-world action execution.

---

## 🏁 Quick Start

### 1. Requirements
- Node.js 20+
- Python 3.9+
- Google Cloud Console Credentials (for Calendar/Gmail)

### 2. Backend Setup
```bash
cd "Ai Agent"
pip install -r requirements.txt
uvicorn app_enhanced:app --port 8000
```

### 3. Frontend Setup
```bash
cd "frontend"
npm install
npx prisma db push
npm run dev
```

---

## 📸 Screenshots

### AI Assistant Chat Interface
![Chat Interface](.github/chat_preview.png)

### Dashboard Overview
![Dashboard Preview](.github/dashboard_preview.png)

---

## 📝 Author & Acknowledgments

- **Lead Developer**: Meenaksh Singhania
- **Core Contributor**: Antigravity AI
- **Technologies**: Google DeepMind, Hugging Face, Vercel

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
