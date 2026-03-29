# TaskSync-AI: The Intelligent Executive Assistant

[![Next.js](https://img.shields.io/badge/Next.js-15-black?style=for-the-badge&logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4-38B2AC?style=for-the-badge&logo=tailwind-css)](https://tailwindcss.com/)
[![Prisma](https://img.shields.io/badge/Prisma-ORM-2D3748?style=for-the-badge&logo=prisma)](https://www.prisma.io/)

**TaskSync-AI** is a production-grade, full-stack AI application designed to streamline personal productivity. It combines a premium SaaS dashboard with a custom-built AI inference engine to manage meetings, emails, and reminders through natural language.

---

## Key Features

### Custom AI Inference Engine
- **Intent Classification**: Fine-tuned DistilBERT model for high-accuracy intent detection (Calendar, Email, Reminders, General).
- **Entity Extraction**: Advanced spaCy-based NER (Named Entity Recognition) to extract dates, times, attendees, and task descriptions.
- **Context Awareness**: Multi-turn conversation state management for complex scheduling flows.

### Premium User Experience
- **Modern Dashboard**: High-fidelity UI inspired by Linear and Stripe, built with **Next.js 15** and **Shadcn UI**.
- **Real-time Interactions**: Smooth animations using **Framer Motion** and instant feedback loops.
- **Responsive Design**: Fully optimized for both desktop and mobile power users.

### 🛠 Enterprise-Grade Architecture
- **Bi-Directional Proxy**: Next.js API routes act as a secure proxy to the FastAPI backend.
- **Secure Authentication**: Integrated **NextAuth.js v5** with Google OAuth provider.
- **Relational Data**: Scalable schema management using **Prisma ORM** (PostgreSQL ready).

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Frontend** | Next.js 15, React 19, Tailwind CSS, Framer Motion, Lucide Icons |
| **Backend** | Python 3.12, FastAPI, Uvicorn |
| **AI/ML** | Hugging Face Transformers, PyTorch, SpaCy (en_core_web_sm) |
| **Database** | Prisma ORM, SQLite (Dev) / PostgreSQL (Prod) |
| **Auth** | NextAuth.js (Auth.js v5) |

---

## Getting Started

### 1. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app_enhanced:app --reload --port 8000
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 3. Environment Variables
Create a `.env` in the root and add:
```env
# Database
DATABASE_URL="file:./dev.db"

# NextAuth
NEXTAUTH_URL="http://localhost:3000"

# Backend
FASTAPI_URL="http://localhost:8000"
```

---

## Project Showcase

> [!NOTE]
> Designed for high-performance AI interactions with a focus on clean typography and intuitive user flow.

---

## Author
**Meenaksh Singhania**  
*AI/ML Enthusiast & Full-Stack Developer*

---
*Developed as part of a transformation project to showcase production-grade AI integration.*
