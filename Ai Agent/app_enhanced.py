"""
Enhanced AI Assistant API with Google Integration
==================================================
Production-ready AI assistant with:
- Fine-tuned intent classification
- Advanced entity extraction (spaCy NER)
- Speech-to-text integration
- Google Calendar & Meet integration
- Gmail for sending invites
- Conversation flow for collecting information
"""

import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
import json
import time
import re
from datetime import datetime, timedelta, date
import dateparser
import tempfile

# Import our custom modules
from entity_extractor import extract_entities, extract_entities_simple

# Try to import Google integration
try:
    from google_integration import get_google_manager, GoogleServicesManager
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("Google integration not available. Install google-api-python-client to enable.")

# ============================================
# App Configuration
# ============================================
app = FastAPI(
    title="Enhanced AI Assistant with Google Integration",
    description="AI-powered personal assistant with Google Calendar, Meet, and Gmail integration",
    version="3.0.0"
)

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Loading
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Starting AI Assistant... Device: {DEVICE}")

# Load fine-tuned intent classifier (if available)
INTENT_MODEL = None
INTENT_TOKENIZER = None
ID2LABEL = None
LABEL2ID = None
USE_FINE_TUNED = False

model_path = "models/intent_classifier"
if os.path.exists(model_path):
    try:
        INTENT_TOKENIZER = DistilBertTokenizer.from_pretrained(model_path)
        INTENT_MODEL = DistilBertForSequenceClassification.from_pretrained(model_path)
        INTENT_MODEL.to(DEVICE)
        INTENT_MODEL.eval()
        
        with open(os.path.join(model_path, "label_mappings.json"), 'r') as f:
            mappings = json.load(f)
        LABEL2ID = mappings['label2id']
        ID2LABEL = {int(k): v for k, v in mappings['id2label'].items()}
        
        USE_FINE_TUNED = True
        print("Fine-tuned model loaded")
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        USE_FINE_TUNED = False

ZERO_SHOT_CLASSIFIER = None
if not USE_FINE_TUNED:
    ZERO_SHOT_CLASSIFIER = pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli",
        device=-1
    )
    print("Zero-shot classifier loaded")

ZERO_SHOT_LABELS = [
    "Schedule or book a meeting or event",
    "Send an email to someone",
    "Check or show my calendar or meetings",
    "Cancel or delete a meeting",
    "Add or set a reminder or task",
    "Check or show my reminders or tasks",
    "Notify, inform, or message someone about an update",
    "General question or small talk"
]

ZEROSHOT_TO_INTENT = {
    "Schedule or book a meeting or event": "schedule_meeting",
    "Send an email to someone": "send_email",
    "Check or show my calendar or meetings": "check_calendar",
    "Cancel or delete a meeting": "cancel_meeting",
    "Add or set a reminder or task": "add_reminder",
    "Check or show my reminders or tasks": "check_reminders",
    "Notify, inform, or message someone about an update": "send_message",
    "General question or small talk": "general_query"
}

STT_PIPELINE = None
try:
    STT_PIPELINE = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        device=-1
    )
    print("STT model loaded")
except Exception as e:
    print(f"Error loading STT model: {e}")

GOOGLE_MANAGER: Optional[GoogleServicesManager] = None
GOOGLE_INITIALIZED = False

if GOOGLE_AVAILABLE:
    if os.path.exists("credentials.json"):
        try:
            GOOGLE_MANAGER = get_google_manager()
            if GOOGLE_MANAGER.initialize():
                GOOGLE_INITIALIZED = True
                print("Google services connected")
            else:
                print("Google authentication required. Visit /auth/google to authenticate.")
        except Exception as e:
            print(f"Google services error: {e}")
    else:
        print("credentials.json not found. Google integration disabled.")

print(f"Model: {'Fine-tuned' if USE_FINE_TUNED else 'Zero-shot'} | STT: {'Yes' if STT_PIPELINE else 'No'} | Google: {'Yes' if GOOGLE_INITIALIZED else 'No'}")

# ============================================
# Conversation State Management
# ============================================
class ConversationState:
    """Track conversation state for multi-turn interactions."""
    IDLE = "idle"
    AWAITING_YOUR_EMAIL = "awaiting_your_email"
    AWAITING_ATTENDEE_EMAIL = "awaiting_attendee_email"
    AWAITING_MEETING_TIME = "awaiting_meeting_time"
    AWAITING_REMINDER_EMAIL = "awaiting_reminder_email"
    CONFIRMING_MEETING = "confirming_meeting"
    # Cancel meeting flow
    AWAITING_CANCEL_EMAIL = "awaiting_cancel_email"
    AWAITING_MEETING_SELECTION = "awaiting_meeting_selection"
    # Check calendar flow
    AWAITING_CALENDAR_EMAIL = "awaiting_calendar_email"

# ============================================
# User Session Management
# ============================================
user_sessions: Dict[str, Dict] = {}

def get_user_session(user_id: str) -> Dict:
    """Get or create user session with memory and conversation state."""
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "meetings": [],
            "reminders": [],
            "emails": [],
            "messages": [],
            "last_person": None,
            "last_intent": None,
            "last_time": time.time(),
            "conversation_history": [],
            # Conversation state for multi-turn
            "state": ConversationState.IDLE,
            "pending_action": None,
            "collected_data": {}
        }
    
    session = user_sessions[user_id]
    
    # Reset if idle for 30 minutes
    if time.time() - session["last_time"] > 1800:
        session.update({
            "meetings": [],
            "reminders": [],
            "emails": [],
            "messages": [],
            "last_person": None,
            "last_intent": None,
            "conversation_history": [],
            "state": ConversationState.IDLE,
            "pending_action": None,
            "collected_data": {}
        })
    
    session["last_time"] = time.time()
    return session

# ============================================
# Intent Classification
# ============================================
def classify_intent(text: str) -> Dict[str, Any]:
    """Classify intent using the best available model."""
    
    if USE_FINE_TUNED and INTENT_MODEL is not None:
        encoding = INTENT_TOKENIZER(
            text,
            truncation=True,
            padding='max_length',
            max_length=64,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)
        
        with torch.no_grad():
            outputs = INTENT_MODEL(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()
        
        all_scores = {ID2LABEL[i]: float(probs[0][i]) for i in range(len(ID2LABEL))}
        
        return {
            "intent": ID2LABEL[pred_idx],
            "confidence": confidence,
            "all_scores": all_scores,
            "model_type": "fine_tuned"
        }
    else:
        result = ZERO_SHOT_CLASSIFIER(text, ZERO_SHOT_LABELS)
        top_label = result['labels'][0]
        confidence = result['scores'][0]
        
        intent = ZEROSHOT_TO_INTENT.get(top_label, "general_query")
        
        all_scores = {
            ZEROSHOT_TO_INTENT.get(label, label): score 
            for label, score in zip(result['labels'], result['scores'])
        }
        
        return {
            "intent": intent,
            "confidence": confidence,
            "all_scores": all_scores,
            "model_type": "zero_shot"
        }

# ============================================
# Rule-based Intent Override
# ============================================
def apply_intent_rules(text: str, predicted_intent: str) -> str:
    """Apply rule-based overrides for better accuracy."""
    text_lower = text.lower()
    
    # Check if it's an email response
    if re.match(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$', text.strip()):
        return "email_response"
    
    # Message/Notification patterns
    message_words = ["send message", "message", "text", "dm", "tell", "inform", "let him know", "let her know", "notify"]
    if any(w in text_lower for w in message_words):
        return "send_message"
    
    # Meeting patterns
    meeting_words = ["meeting", "calendar", "schedule", "appointment", "sync", "catch-up", "google meet", "meet"]
    cancel_words = ["cancel", "delete", "remove", "call off"]
    question_words = ["do i have", "what", "show", "list", "see", "any", "check"]
    book_words = ["schedule", "set up", "book", "arrange", "plan", "create"]
    
    if any(c in text_lower for c in cancel_words) and any(m in text_lower for m in meeting_words):
        return "cancel_meeting"
    
    if any(m in text_lower for m in meeting_words) and any(q in text_lower for q in question_words):
        return "check_calendar"
    
    if any(b in text_lower for b in book_words) and any(m in text_lower for m in meeting_words):
        return "schedule_meeting"
    
    if "meeting" in text_lower and any(t in text_lower for t in ["tomorrow", "today", "at", "this", "next", "morning", "evening", "afternoon"]):
        return "schedule_meeting"
    
    # Reminder patterns
    reminder_words = ["remind", "reminder", "remember to", "note to", "todo", "to-do", "task"]
    if any(r in text_lower for r in reminder_words) and not any(m in text_lower for m in meeting_words):
        if any(q in text_lower for q in ["do i have", "what reminders", "show reminders", "list reminders", "any reminders", "check reminders"]):
            return "check_reminders"
        return "add_reminder"
    
    # Email patterns
    if "@" in text or "email" in text_lower:
        return "send_email"
    
    # Confirmation patterns
    if text_lower in ["yes", "yeah", "yep", "sure", "ok", "okay", "confirm", "go ahead", "do it"]:
        return "confirmation"
    if text_lower in ["no", "nope", "cancel", "nevermind", "never mind", "stop"]:
        return "cancellation"
    
    return predicted_intent

# ============================================
# Speech-to-Text Email Conversion
# ============================================
def convert_spoken_email(text: str) -> str:
    """
    Convert spoken email from STT to proper email format.
    Handles common STT variations like:
    - "john at gmail dot com" -> "john@gmail.com"
    - "john 123 at gmail dot com" -> "john123@gmail.com"
    - "my email is john at gmail dot com" -> "john@gmail.com"
    """
    original_text = text
    text = text.strip()
    
    # Remove common prefixes from spoken email
    prefixes_to_remove = [
        r"^my email is\s*",
        r"^my email address is\s*",
        r"^email is\s*",
        r"^it's\s*",
        r"^its\s*",
        r"^the email is\s*",
        r"^sure,?\s*",
        r"^yes,?\s*",
        r"^okay,?\s*",
        r"^ok,?\s*",
    ]
    for prefix in prefixes_to_remove:
        text = re.sub(prefix, '', text, flags=re.IGNORECASE)
    
    text = text.strip()
    
    # Convert spoken numbers to digits (common in emails)
    number_words = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12'
    }
    for word, digit in number_words.items():
        text = re.sub(rf'\b{word}\b', digit, text, flags=re.IGNORECASE)
    
    # Handle common domain variations FIRST (before replacing "dot")
    # This prevents "hotmail" from becoming "hot@il" when we replace "at"
    domain_replacements = [
        (r'\bgmail\s*\.?\s*com\b', 'gmail.com'),
        (r'\bgmail\s+dot\s+com\b', 'gmail.com'),
        (r'\byahoo\s*\.?\s*com\b', 'yahoo.com'),
        (r'\byahoo\s+dot\s+com\b', 'yahoo.com'),
        (r'\boutlook\s*\.?\s*com\b', 'outlook.com'),
        (r'\boutlook\s+dot\s+com\b', 'outlook.com'),
        (r'\bhotmail\s*\.?\s*com\b', 'hotmail.com'),
        (r'\bhotmail\s+dot\s+com\b', 'hotmail.com'),
        (r'\bicloud\s*\.?\s*com\b', 'icloud.com'),
        (r'\bicloud\s+dot\s+com\b', 'icloud.com'),
        (r'\bprotonmail\s*\.?\s*com\b', 'protonmail.com'),
        (r'\bprotonmail\s+dot\s+com\b', 'protonmail.com'),
        (r'\blive\s*\.?\s*com\b', 'live.com'),
        (r'\blive\s+dot\s+com\b', 'live.com'),
        (r'\baol\s*\.?\s*com\b', 'aol.com'),
        (r'\baol\s+dot\s+com\b', 'aol.com'),
        # Handle .edu, .org, .net domains
        (r'(\w+)\s+dot\s+edu\b', r'\1.edu'),
        (r'(\w+)\s+dot\s+org\b', r'\1.org'),
        (r'(\w+)\s+dot\s+net\b', r'\1.net'),
        (r'(\w+)\s+dot\s+co\b', r'\1.co'),
        (r'(\w+)\s+dot\s+io\b', r'\1.io'),
    ]
    for pattern, replacement in domain_replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Now replace " at " with "@" (word boundary to avoid "hotmail" issues)
    # Use word boundaries - the word "at" surrounded by spaces
    text = re.sub(r'\s+at\s+', '@', text, flags=re.IGNORECASE)
    # Also handle "at" at the start after a username-like string
    text = re.sub(r'(\w)\s+at\s*$', r'\1@', text, flags=re.IGNORECASE)
    
    # Handle remaining "dot" (for domains not in our list)
    # Only replace " dot " when it looks like it's in a domain context (after @)
    if '@' in text:
        parts = text.split('@')
        if len(parts) == 2:
            local_part = parts[0]
            domain_part = parts[1]
            # Replace " dot " with "." in domain
            domain_part = re.sub(r'\s+dot\s+', '.', domain_part, flags=re.IGNORECASE)
            # Also handle "dot" at word boundaries
            domain_part = re.sub(r'\s+dot(\s|$)', r'.\1', domain_part, flags=re.IGNORECASE)
            text = f"{local_part}@{domain_part}"
    
    # Remove all spaces (email shouldn't have spaces)
    text = re.sub(r'\s+', '', text)
    
    # Remove trailing punctuation
    text = text.rstrip('.,!?')
    
    # Convert to lowercase for email
    text = text.lower()
    
    # Validate it looks like an email
    if re.match(r'^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$', text):
        return text
    
    # If we couldn't parse it into a valid email, try a more aggressive approach
    # Look for any pattern that looks like "something @ domain . tld"
    aggressive_match = re.search(
        r'([a-z0-9._%+-]+)\s*(?:@|at)\s*([a-z0-9.-]+)\s*(?:\.|dot)\s*([a-z]{2,})',
        original_text.lower()
    )
    if aggressive_match:
        email = f"{aggressive_match.group(1)}@{aggressive_match.group(2)}.{aggressive_match.group(3)}"
        email = re.sub(r'\s+', '', email)
        if re.match(r'^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$', email):
            return email
    
    # Return what we have (cleaned up) - the validation will happen downstream
    return text

# ============================================
# Action Handlers with Google Integration
# ============================================
def handle_schedule_meeting(entities: Dict, text: str, session: Dict) -> str:
    """Handle meeting scheduling with conversation flow."""
    
    # Check current state
    state = session.get("state", ConversationState.IDLE)
    collected = session.get("collected_data", {})
    
    if state == ConversationState.IDLE:
        # Starting new meeting request
        persons = entities.get('names') or []
        person = persons[0] if persons else None
        dt_iso = entities.get('datetime')
        emails = entities.get('emails') or []
        
        # Store what we have
        collected = {
            "person": person,
            "datetime": dt_iso,
            "attendee_email": emails[0] if emails else None,
            "your_email": None,
            "title": f"Meeting with {person}" if person else "Meeting"
        }
        session["collected_data"] = collected
        
        # Check what we're missing
        if not collected["your_email"]:
            session["state"] = ConversationState.AWAITING_YOUR_EMAIL
            session["pending_action"] = "schedule_meeting"
            return f"{{icon:calendar}} I'll schedule a meeting{' with ' + person if person else ''}.\n\nFirst, what's your email address?"
    
    elif state == ConversationState.AWAITING_YOUR_EMAIL:
        # Check if user is starting a new request instead of providing email
        text_lower = text.lower()
        if any(word in text_lower for word in ["schedule", "meeting", "with", "tomorrow", "today", "at"]) and len(text.split()) > 3:
            # User is starting a new meeting request, reset and process as new
            session["state"] = ConversationState.IDLE
            session["pending_action"] = None
            session["collected_data"] = {}
            # Re-extract entities from the new text
            new_entities = extract_entities_simple(text)
            persons = new_entities.get('names') or []
            person = persons[0] if persons else None
            dt_iso = new_entities.get('datetime')
            emails = new_entities.get('emails') or []
            
            collected = {
                "person": person,
                "datetime": dt_iso,
                "attendee_email": emails[0] if emails else None,
                "your_email": None,
                "title": f"Meeting with {person}" if person else "Meeting"
            }
            session["collected_data"] = collected
            
            if not collected["your_email"]:
                session["state"] = ConversationState.AWAITING_YOUR_EMAIL
                session["pending_action"] = "schedule_meeting"
                return f"{{icon:calendar}} I'll schedule a meeting{' with ' + person if person else ''}.\n\nFirst, what's your email address?"
        
        email = convert_spoken_email(text)
        
        # Validate it's a proper email format
        if re.match(r'^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$', email):
            collected["your_email"] = email
            session["collected_data"] = collected
            
            if not collected.get("attendee_email"):
                session["state"] = ConversationState.AWAITING_ATTENDEE_EMAIL
                attendee_name = collected.get("person", "the person")
                return f"{{icon:check}} Got it! Your email: {collected['your_email']}\n\nNow, what's the email of {attendee_name} you're meeting with?"
            else:
                # We have everything, proceed
                return _create_google_meeting(session)
        
        return "{{icon:x}} I couldn't understand that email. Please try again. Say something like 'john at gmail dot com' or type the email directly."
    
    elif state == ConversationState.AWAITING_ATTENDEE_EMAIL:
        # Check if user is starting a new request instead of providing email
        text_lower = text.lower()
        if any(word in text_lower for word in ["schedule", "meeting", "with", "tomorrow", "today", "at"]) and len(text.split()) > 3:
            # User is starting a new meeting request, reset and process as new
            session["state"] = ConversationState.IDLE
            session["pending_action"] = None
            session["collected_data"] = {}
            # Re-extract entities from the new text
            new_entities = extract_entities_simple(text)
            persons = new_entities.get('names') or []
            person = persons[0] if persons else None
            dt_iso = new_entities.get('datetime')
            emails = new_entities.get('emails') or []
            
            collected = {
                "person": person,
                "datetime": dt_iso,
                "attendee_email": emails[0] if emails else None,
                "your_email": None,
                "title": f"Meeting with {person}" if person else "Meeting"
            }
            session["collected_data"] = collected
            
            if not collected["your_email"]:
                session["state"] = ConversationState.AWAITING_YOUR_EMAIL
                session["pending_action"] = "schedule_meeting"
                return f"{{icon:calendar}} I'll schedule a meeting{' with ' + person if person else ''}.\n\nFirst, what's your email address?"
        
        email = convert_spoken_email(text)
        
        # Validate it's a proper email format
        if re.match(r'^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$', email):
            collected["attendee_email"] = email
            session["collected_data"] = collected
            
            if not collected.get("datetime"):
                session["state"] = ConversationState.AWAITING_MEETING_TIME
                return f"{{icon:check}} Attendee email: {collected['attendee_email']}\n\nWhen should I schedule this meeting? (e.g., 'tomorrow at 3pm', 'next Monday at 10am')"
            else:
                return _create_google_meeting(session)
        
        return "{{icon:x}} I couldn't understand that email. Please try again. Say something like 'john at gmail dot com' or type the email directly."
    
    elif state == ConversationState.AWAITING_MEETING_TIME:
        # Parse time from response
        parsed = dateparser.parse(text, settings={'PREFER_DATES_FROM': 'future'})
        if parsed:
            collected["datetime"] = parsed.isoformat()
            session["collected_data"] = collected
            return _create_google_meeting(session)
        else:
            return "{{icon:x}} I couldn't understand that time. Please try again (e.g., 'tomorrow at 3pm'):"
    
    return "Something went wrong. Let's start over. Say 'schedule a meeting' to try again."

def _create_google_meeting(session: Dict) -> str:
    """Actually create the Google Meet meeting."""
    collected = session["collected_data"]
    
    # Reset state
    session["state"] = ConversationState.IDLE
    session["pending_action"] = None
    
    if not GOOGLE_INITIALIZED:
        # Fallback: just store locally
        meeting = {
            "person": collected.get("person", "Unknown"),
            "time": collected.get("datetime"),
            "your_email": collected.get("your_email"),
            "attendee_email": collected.get("attendee_email"),
            "created_at": datetime.now().isoformat()
        }
        session["meetings"].append(meeting)
        session["collected_data"] = {}
        
        return f"""{{icon:calendar}} Meeting scheduled (locally stored):

**With:** {collected.get('person', collected.get('attendee_email', 'Unknown'))}
**Time:** {collected.get('datetime', 'TBD')}
**Your email:** {collected.get('your_email')}
**Attendee:** {collected.get('attendee_email')}

{{icon:alert}} Google Calendar is not connected. To enable Google Meet links and calendar invites, please set up Google integration.

Visit /auth/google to connect your Google account."""
    
    try:
        # Parse datetime
        meeting_time = dateparser.parse(collected["datetime"]) if collected.get("datetime") else datetime.now() + timedelta(hours=1)
        
        # Create meeting with Google
        result = GOOGLE_MANAGER.schedule_meeting(
            title=collected.get("title", "Meeting"),
            start_time=meeting_time,
            duration_minutes=60,
            attendee_emails=[collected["your_email"], collected["attendee_email"]],
            description=f"Meeting scheduled via AI Assistant",
            send_email_invite=True
        )
        
        if result.get("success"):
            # Store in session
            meeting = {
                "event_id": result.get("event_id"),
                "person": collected.get("person", collected.get("attendee_email")),
                "time": meeting_time.isoformat(),
                "meet_link": result.get("meet_link"),
                "attendees": [collected["your_email"], collected["attendee_email"]]
            }
            session["meetings"].append(meeting)
            session["collected_data"] = {}
            
            return f"""{{icon:check}} **Meeting Scheduled Successfully!**

{{icon:calendar}} **Title:** {collected.get('title', 'Meeting')}
{{icon:clock}} **Time:** {meeting_time.strftime('%B %d, %Y at %I:%M %p')}
{{icon:users}} **Attendees:** 
   • {collected['your_email']}
   • {collected['attendee_email']}

{{icon:link}} **Google Meet Link:** {result.get('meet_link', 'N/A')}

{{icon:mail}} Calendar invites have been sent to both email addresses!

[View in Calendar]({result.get('event_link', '#')})"""
        else:
            session["collected_data"] = {}
            return f"{{icon:x}} Failed to create meeting: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        session["collected_data"] = {}
        return f"{{icon:x}} Error creating meeting: {str(e)}"

def handle_add_reminder(entities: Dict, text: str, session: Dict) -> str:
    """Handle reminder creation with Google Calendar integration."""
    
    state = session.get("state", ConversationState.IDLE)
    collected = session.get("collected_data", {})
    
    if state == ConversationState.IDLE:
        dt_iso = entities.get('datetime')
        emails = entities.get('emails') or []
        
        # Extract reminder text
        reminder_text = re.sub(r'^(remind me to|set a reminder to|reminder to|remind me)', '', text, flags=re.IGNORECASE).strip()
        if not reminder_text:
            reminder_text = text
        
        collected = {
            "reminder_text": reminder_text,
            "datetime": dt_iso,
            "user_email": emails[0] if emails else None
        }
        session["collected_data"] = collected
        
        if not collected["datetime"]:
            session["state"] = ConversationState.AWAITING_MEETING_TIME
            session["pending_action"] = "add_reminder"
            return f"{{icon:bell}} I'll set a reminder for: \"{reminder_text}\"\n\nWhen should I remind you? (e.g., 'tomorrow at 5pm', 'in 2 hours')"
        
        if not collected["user_email"] and GOOGLE_INITIALIZED:
            session["state"] = ConversationState.AWAITING_REMINDER_EMAIL
            session["pending_action"] = "add_reminder"
            return f"{{icon:bell}} Reminder: \"{reminder_text}\" at {dt_iso}\n\nWhat's your email? (for calendar sync and notification)"
        
        return _create_google_reminder(session)
    
    elif state == ConversationState.AWAITING_MEETING_TIME:
        parsed = dateparser.parse(text, settings={'PREFER_DATES_FROM': 'future'})
        if parsed:
            collected["datetime"] = parsed.isoformat()
            session["collected_data"] = collected
            
            if not collected.get("user_email") and GOOGLE_INITIALIZED:
                session["state"] = ConversationState.AWAITING_REMINDER_EMAIL
                return f"{{icon:check}} Time set: {parsed.strftime('%B %d at %I:%M %p')}\n\nWhat's your email? (for calendar notification)"
            
            return _create_google_reminder(session)
        else:
            return "{{icon:x}} I couldn't understand that time. Please try again:"
    
    elif state == ConversationState.AWAITING_REMINDER_EMAIL:
        # User might skip email first
        if text.lower().strip() in ["skip", "no", "none", "no email"]:
            return _create_google_reminder(session)
        
        email = convert_spoken_email(text)
        
        if re.match(r'^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$', email):
            collected["user_email"] = email
            session["collected_data"] = collected
            return _create_google_reminder(session)
        else:
            return "{{icon:x}} I couldn't understand that email. Please try again or say 'skip':"
    
    return _create_google_reminder(session)

def _create_google_reminder(session: Dict) -> str:
    """Create reminder in Google Calendar."""
    collected = session["collected_data"]
    
    # Reset state
    session["state"] = ConversationState.IDLE
    session["pending_action"] = None
    
    reminder_time = dateparser.parse(collected["datetime"]) if collected.get("datetime") else datetime.now() + timedelta(hours=1)
    
    if not GOOGLE_INITIALIZED:
        # Store locally
        reminder = {
            "text": collected["reminder_text"],
            "time": reminder_time.isoformat() if reminder_time else None,
            "created_at": datetime.now().isoformat()
        }
        session["reminders"].append(reminder)
        session["collected_data"] = {}
        
        return f"""{{icon:bell}} **Reminder Set** (locally stored):

{{icon:file-text}} **Task:** {collected['reminder_text']}
{{icon:clock}} **Time:** {reminder_time.strftime('%B %d, %Y at %I:%M %p') if reminder_time else 'TBD'}

{{icon:alert}} Google Calendar is not connected. To sync reminders, set up Google integration."""
    
    try:
        result = GOOGLE_MANAGER.add_reminder(
            title=collected["reminder_text"],
            reminder_time=reminder_time,
            user_email=collected.get("user_email"),
            description="Reminder set via AI Assistant"
        )
        
        if result.get("success"):
            reminder = {
                "event_id": result.get("event_id"),
                "text": collected["reminder_text"],
                "time": reminder_time.isoformat()
            }
            session["reminders"].append(reminder)
            session["collected_data"] = {}
            
            return f"""{{icon:check}} **Reminder Added to Google Calendar!**

{{icon:file-text}} **Task:** {collected['reminder_text']}
{{icon:clock}} **Time:** {reminder_time.strftime('%B %d, %Y at %I:%M %p')}

You'll receive a notification at the scheduled time.

[View in Calendar]({result.get('event_link', '#')})"""
        else:
            return f"{{icon:x}} Failed to create reminder: {result.get('error')}"
            
    except Exception as e:
        return f"{{icon:x}} Error creating reminder: {str(e)}"

def check_calendar(entities: Dict, text: str, session: Dict) -> str:
    """Check calendar - asks for email first, then shows meetings for that email."""
    
    state = session.get("state", ConversationState.IDLE)
    collected = session.get("collected_data", {})
    
    if not GOOGLE_INITIALIZED:
        # Local storage fallback
        meetings = session.get("meetings", [])
        if not meetings:
            return "{{icon:calendar}} No meetings scheduled. Your calendar is clear!"
        
        response = "{{icon:calendar}} **Your Meetings** (locally stored):\n\n"
        for m in meetings:
            response += f"• {m.get('person', 'Unknown')} at {m.get('time', 'TBD')}\n"
        
        response += "\n{{icon:alert}} Connect Google Calendar for full features."
        return response
    
    if state == ConversationState.IDLE:
        # Check if email was provided in the initial request
        emails = entities.get('emails') or []
        if emails:
            email = emails[0].lower()
            return _fetch_and_show_calendar(session, email)
        
        # Ask for email
        session["state"] = ConversationState.AWAITING_CALENDAR_EMAIL
        session["pending_action"] = "check_calendar"
        session["collected_data"] = {}
        return "{{icon:calendar}} I'll show your scheduled meetings.\n\nWhat's your email address?"
    
    elif state == ConversationState.AWAITING_CALENDAR_EMAIL:
        email = convert_spoken_email(text)
        
        if re.match(r'^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$', email):
            return _fetch_and_show_calendar(session, email)
        
        return "{{icon:x}} I couldn't understand that email. Please try again."
    
    return "Something went wrong. Say 'check calendar' to try again."


def _fetch_and_show_calendar(session: Dict, email: str) -> str:
    """Fetch and display meetings for a specific email."""
    
    # Reset state
    session["state"] = ConversationState.IDLE
    session["pending_action"] = None
    session["collected_data"] = {}
    
    # Get upcoming meetings
    all_meetings = GOOGLE_MANAGER.get_upcoming_meetings(20)
    
    # Filter meetings where user is an attendee
    user_meetings = []
    for meeting in all_meetings:
        attendees = meeting.get('attendees', [])
        # Include if user is in attendees list (case-insensitive)
        if any(email.lower() == att.lower() for att in attendees):
            user_meetings.append(meeting)
        # Also include if it appears to be their own meeting (no attendees = probably organizer)
        elif not attendees:
            user_meetings.append(meeting)
    
    if not user_meetings:
        return f"{{icon:calendar}} No upcoming meetings found for **{email}**.\n\nYour calendar is clear!"
    
    response = f"{{icon:calendar}} **Upcoming meetings for {email}:**\n\n"
    
    for event in user_meetings:
        start = event.get('start', 'TBD')
        if start and start != 'TBD':
            try:
                dt = dateparser.parse(start)
                start = dt.strftime('%b %d at %I:%M %p') if dt else start
            except:
                pass
        
        meet_link = f"\n   {{icon:link}} [Join Meet]({event.get('meet_link')})" if event.get('meet_link') else ""
        attendees = ", ".join(event.get('attendees', [])[:3])
        if attendees:
            attendees = f"\n   {{icon:users}} {attendees}"
        
        response += f"• **{event.get('summary', 'Untitled')}**\n   {{icon:clock}} {start}{attendees}{meet_link}\n\n"
    
    return response

def check_reminders(entities: Dict, text: str, session: Dict) -> str:
    """Check reminders."""
    reminders = session.get("reminders", [])
    
    if not reminders:
        return "{{icon:list}} No reminders set. You're all caught up!"
    
    response = "{{icon:list}} **Your Reminders:**\n\n"
    for i, r in enumerate(reminders, 1):
        time_str = ""
        if r.get('time'):
            try:
                dt = dateparser.parse(r['time'])
                time_str = f" - {dt.strftime('%b %d at %I:%M %p')}" if dt else ""
            except:
                time_str = f" - {r['time']}"
        
        response += f"{i}. {r.get('text', 'Reminder')}{time_str}\n"
    
    return response

def send_email(entities: Dict, text: str, session: Dict) -> str:
    """Send an email."""
    emails = entities.get('emails') or []
    to_email = emails[0] if emails else None
    
    if not to_email:
        persons = entities.get('names') or []
        if persons:
            return f"{{icon:mail}} I found {persons[0]} but need an email address. What's their email?"
        return "{{icon:mail}} Please specify a recipient email address."
    
    session["emails"].append({"to": to_email, "sent_at": datetime.now().isoformat()})
    return f"{{icon:mail}} Email composed for {to_email}. (Email sending coming soon!)"

def send_message(entities: Dict, text: str, session: Dict) -> str:
    """Send a message/notification."""
    persons = entities.get('names') or []
    person = persons[0] if persons else session.get("last_person")
    
    if not person:
        return "{{icon:message-square}} Who should I message?"
    
    msg_match = re.search(r'(?:say|message|tell|text|inform)\s+(?:to\s+)?\b[a-zA-Z]+\b\s+(.*)', text, re.IGNORECASE)
    message_content = msg_match.group(1).strip() if msg_match else "your message"
    
    session["messages"].append({"to": person, "content": message_content})
    session["last_person"] = person
    
    return f"{{icon:message-square}} Message sent to {person}: \"{message_content}\""

def general_response(entities: Dict, text: str, session: Dict) -> str:
    """Handle general queries."""
    text_lower = text.lower().strip()
    
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if any(text_lower.startswith(g) for g in greetings):
        google_status = "{icon:check} Connected" if GOOGLE_INITIALIZED else "{icon:x} Not connected (visit /auth/google)"
        return f"""{{icon:hand}} Hello! I'm your AI assistant with Google integration.

**Status:**
• Google Calendar: {google_status}

**I can help you:**
• {{icon:calendar}} Schedule meetings with Google Meet links
• {{icon:bell}} Add reminders to Google Calendar
• {{icon:list}} Check your calendar
• {{icon:mail}} Send emails

What would you like to do?"""
    
    if "thank" in text_lower:
        return "{icon:smile} You're welcome! Let me know if you need anything else."
    
    if "help" in text_lower:
        return """{icon:help-circle} **I can help you with:**

• **Schedule meetings** - "Schedule a meeting with John tomorrow at 3pm"
  Creates Google Meet link & sends invites to both parties

• **Check calendar** - "What meetings do I have today?"
  Shows your Google Calendar events

• **Set reminders** - "Remind me to call mom at 5pm"
  Adds to Google Calendar with notifications

• **Send messages** - "Tell Sarah I'll be late"

Just tell me what you need!"""
    
    return "{icon:help-circle} I'm not sure how to help with that. Try asking me to schedule a meeting or set a reminder!"

# ============================================
# Cancel Meeting Handler
# ============================================
def handle_cancel_meeting(entities: Dict, text: str, session: Dict) -> str:
    """Handle cancel meeting flow - asks for email, shows meetings, lets user select."""
    
    state = session.get("state", ConversationState.IDLE)
    collected = session.get("collected_data", {})
    
    if state == ConversationState.IDLE:
        # Starting new cancel request
        if not GOOGLE_INITIALIZED:
            return "{{icon:x}} Google Calendar is not connected. Please connect to cancel meetings."
        
        # Check if email was provided in the initial request
        emails = entities.get('emails') or []
        if emails:
            email = emails[0].lower()
            collected = {"user_email": email, "meetings_list": []}
            session["collected_data"] = collected
            return _fetch_and_show_meetings_for_cancel(session, email)
        
        # Ask for email
        session["state"] = ConversationState.AWAITING_CANCEL_EMAIL
        session["pending_action"] = "cancel_meeting"
        session["collected_data"] = {"meetings_list": []}
        return "{{icon:calendar}} I'll help you cancel a meeting.\n\nWhat's your email address? (I'll show meetings where you're an attendee)"
    
    elif state == ConversationState.AWAITING_CANCEL_EMAIL:
        email = convert_spoken_email(text)
        
        if re.match(r'^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$', email):
            collected["user_email"] = email
            session["collected_data"] = collected
            return _fetch_and_show_meetings_for_cancel(session, email)
        
        return "{{icon:x}} I couldn't understand that email. Please try again or say 'cancel' to stop."
    
    elif state == ConversationState.AWAITING_MEETING_SELECTION:
        # User is selecting a meeting to cancel
        meetings = collected.get("meetings_list", [])
        
        if not meetings:
            session["state"] = ConversationState.IDLE
            session["pending_action"] = None
            return "{{icon:x}} No meetings to cancel. Start over by saying 'cancel meeting'."
        
        # Check for cancel/back commands
        text_lower = text.lower().strip()
        if text_lower in ["cancel", "back", "stop", "never mind", "nevermind"]:
            session["state"] = ConversationState.IDLE
            session["pending_action"] = None
            session["collected_data"] = {}
            return "{{icon:x}} Cancelled. Let me know if you need anything else!"
        
        # Try to parse meeting selection
        selection = None
        
        # Try direct number
        try:
            selection = int(text.strip())
        except ValueError:
            pass
        
        # Try word numbers
        if selection is None:
            number_words = {
                'one': 1, 'first': 1, '1st': 1,
                'two': 2, 'second': 2, '2nd': 2,
                'three': 3, 'third': 3, '3rd': 3,
                'four': 4, 'fourth': 4, '4th': 4,
                'five': 5, 'fifth': 5, '5th': 5,
                'six': 6, 'sixth': 6, '6th': 6,
                'seven': 7, 'seventh': 7, '7th': 7,
                'eight': 8, 'eighth': 8, '8th': 8,
                'nine': 9, 'ninth': 9, '9th': 9,
                'ten': 10, 'tenth': 10, '10th': 10,
            }
            for word, num in number_words.items():
                if word in text_lower:
                    selection = num
                    break
        
        # Validate selection
        if selection is None or selection < 1 or selection > len(meetings):
            return f"{{icon:x}} Please enter a number between 1 and {len(meetings)}, or say 'cancel' to stop."
        
        # Cancel the selected meeting
        meeting = meetings[selection - 1]
        event_id = meeting.get('id')
        meeting_title = meeting.get('summary', 'Untitled Meeting')
        
        if GOOGLE_MANAGER.cancel_meeting(event_id):
            session["state"] = ConversationState.IDLE
            session["pending_action"] = None
            session["collected_data"] = {}
            return f"{{icon:check}} Successfully cancelled: **{meeting_title}**\n\nThe meeting has been removed from your Google Calendar."
        else:
            return f"{{icon:x}} Failed to cancel the meeting. Please try again or cancel it directly in Google Calendar."
    
    return "Something went wrong. Say 'cancel meeting' to start over."


def _fetch_and_show_meetings_for_cancel(session: Dict, email: str) -> str:
    """Fetch meetings and show them for selection."""
    collected = session.get("collected_data", {})
    
    # Get upcoming meetings
    all_meetings = GOOGLE_MANAGER.get_upcoming_meetings(20)
    
    # Filter meetings where user is an attendee
    user_meetings = []
    for meeting in all_meetings:
        attendees = meeting.get('attendees', [])
        # Include if user is in attendees list (case-insensitive)
        if any(email.lower() == att.lower() for att in attendees):
            user_meetings.append(meeting)
        # Also include if it appears to be their own meeting (no attendees = probably organizer)
        elif not attendees:
            user_meetings.append(meeting)
    
    if not user_meetings:
        session["state"] = ConversationState.IDLE
        session["pending_action"] = None
        session["collected_data"] = {}
        return f"{{icon:calendar}} No upcoming meetings found for **{email}**.\n\nYour calendar is clear, or you may not be listed as an attendee on upcoming meetings."
    
    # Store meetings list for selection
    collected["meetings_list"] = user_meetings
    session["collected_data"] = collected
    session["state"] = ConversationState.AWAITING_MEETING_SELECTION
    session["pending_action"] = "cancel_meeting"
    
    # Build response with numbered list
    response = f"{{icon:calendar}} **Upcoming meetings for {email}:**\n\n"
    
    for i, meeting in enumerate(user_meetings, 1):
        title = meeting.get('summary', 'Untitled Meeting')
        start = meeting.get('start', 'TBD')
        
        # Format the date/time nicely
        if start and start != 'TBD':
            try:
                dt = dateparser.parse(start)
                start = dt.strftime('%b %d at %I:%M %p') if dt else start
            except:
                pass
        
        attendees = meeting.get('attendees', [])
        attendees_str = f"\n   {{icon:users}} {', '.join(attendees[:3])}" if attendees else ""
        meet_link = meeting.get('meet_link')
        meet_str = f"\n   {{icon:link}} Has Meet link" if meet_link else ""
        
        response += f"**{i}.** {title}\n   {{icon:clock}} {start}{attendees_str}{meet_str}\n\n"
    
    response += "---\n**Which meeting would you like to cancel?** (Enter the number, or say 'cancel' to stop)"
    
    return response


# Action mapping
ACTION_MAP = {
    "schedule_meeting": handle_schedule_meeting,
    "send_email": send_email,
    "check_calendar": check_calendar,
    "cancel_meeting": handle_cancel_meeting,
    "add_reminder": handle_add_reminder,
    "check_reminders": check_reminders,
    "send_message": send_message,
    "general_query": general_response
}

# ============================================
# Process user input with conversation state
# ============================================
def process_input(user_id: str, text: str) -> Dict[str, Any]:
    """Process user input with conversation state handling."""
    session = get_user_session(user_id)
    text = text.strip()
    
    # Check if we're in a conversation flow
    state = session.get("state", ConversationState.IDLE)
    pending_action = session.get("pending_action")
    
    # Detect if user is starting a NEW request (not responding to a question)
    # This handles cases where voice input might be a new command
    text_lower = text.lower()
    
    # Check if text contains an email - if we're waiting for email, don't treat as new request
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    contains_email = bool(re.search(email_pattern, text))
    
    # States where we're waiting for an email
    email_waiting_states = [
        ConversationState.AWAITING_YOUR_EMAIL, 
        ConversationState.AWAITING_ATTENDEE_EMAIL, 
        ConversationState.AWAITING_REMINDER_EMAIL,
        ConversationState.AWAITING_CANCEL_EMAIL,
        ConversationState.AWAITING_CALENDAR_EMAIL
    ]
    
    # If we're waiting for an email and text contains an email, it's NOT a new request
    if state in email_waiting_states and contains_email:
        is_new_request = False
    # If we're waiting for meeting selection, it's NOT a new request (user is selecting a number)
    elif state == ConversationState.AWAITING_MEETING_SELECTION:
        is_new_request = False
    else:
        # Check for new request indicators, but exclude "email" if we're in email waiting state
        new_request_indicators = [
            "schedule", "remind", "reminder", 
            "check calendar", "what meetings", "set up"
        ]
        # Don't treat "cancel" as new request if we're in cancel flow
        if state not in [ConversationState.AWAITING_CANCEL_EMAIL, ConversationState.AWAITING_MEETING_SELECTION]:
            new_request_indicators.append("cancel")
        # Only include "email" and "send" if we're NOT waiting for an email
        if state not in email_waiting_states:
            new_request_indicators.extend(["email", "send"])
        # Don't treat "meeting" as new if we're selecting a meeting to cancel
        if state != ConversationState.AWAITING_MEETING_SELECTION:
            new_request_indicators.append("meeting")
        
        is_new_request = any(indicator in text_lower for indicator in new_request_indicators) and len(text.split()) > 3
    
    # If in waiting state but user seems to be starting a new request, reset state
    if state != ConversationState.IDLE and pending_action and is_new_request:
        # Reset state and treat as new request
        session["state"] = ConversationState.IDLE
        session["pending_action"] = None
        session["collected_data"] = {}
        # Continue to new request processing below
    
    if state != ConversationState.IDLE and pending_action and not is_new_request:
        entities = extract_entities_simple(text)
        
        # Special handling: if we're waiting for an email, convert spoken email to proper format
        if state in email_waiting_states:
            # Use robust email converter
            email = convert_spoken_email(text)
            
            # If conversion produced a valid email, update entities and text
            if re.match(r'^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$', email):
                if not entities.get('emails'):
                    entities['emails'] = [email]
                text = email
        
        if pending_action == "schedule_meeting":
            response = handle_schedule_meeting(entities, text, session)
        elif pending_action == "add_reminder":
            response = handle_add_reminder(entities, text, session)
        elif pending_action == "cancel_meeting":
            response = handle_cancel_meeting(entities, text, session)
        elif pending_action == "check_calendar":
            response = check_calendar(entities, text, session)
        else:
            response = "Something went wrong. Let's start over."
            session["state"] = ConversationState.IDLE
            session["pending_action"] = None
        
        return {
            "intent": pending_action,
            "confidence": 1.0,
            "model_type": "conversation_flow",
            "entities": entities,
            "response": response,
            "state": session.get("state"),
            "google_connected": GOOGLE_INITIALIZED
        }
    
    # New request - classify intent
    classification = classify_intent(text)
    intent = classification["intent"]
    confidence = classification["confidence"]
    
    # Apply rule-based overrides (but don't override if we're in a conversation flow)
    # IMPORTANT: Don't classify as email_response if we're in a conversation flow
    # The conversation flow handler will handle emails properly
    if state == ConversationState.IDLE:
        intent = apply_intent_rules(text, intent)
    
    # Handle confirmations/cancellations
    if intent == "confirmation":
        return {
            "intent": "confirmation",
            "confidence": 1.0,
            "entities": {},
            "response": "{icon:check} Confirmed! What would you like to do next?"
        }
    if intent == "cancellation":
        session["state"] = ConversationState.IDLE
        session["pending_action"] = None
        session["collected_data"] = {}
        return {
            "intent": "cancellation",
            "confidence": 1.0,
            "entities": {},
            "response": "{icon:x} Cancelled. What would you like to do instead?"
        }
    
    # Extract entities
    entities = extract_entities_simple(text)
    
    # Handle email_response intent specially - but only if we're truly in IDLE state
    # If we're in a conversation flow, this should have been handled above
    if intent == "email_response" and state == ConversationState.IDLE:
        # This is a standalone email without context
        return {
            "intent": "email_response",
            "confidence": 1.0,
            "entities": entities,
            "response": "{{icon:mail}} I received an email address, but I'm not sure what you'd like me to do with it. Please provide more context.",
            "state": "idle",
            "google_connected": GOOGLE_INITIALIZED
        }
    
    # Execute action
    action_fn = ACTION_MAP.get(intent, general_response)
    response = action_fn(entities, text, session)
    
    # Update session
    session["last_intent"] = intent
    session["conversation_history"].append({
        "user": text,
        "assistant": response,
        "intent": intent,
        "timestamp": datetime.now().isoformat()
    })
    
    return {
        "intent": intent,
        "confidence": confidence,
        "model_type": classification.get("model_type", "unknown"),
        "entities": entities,
        "response": response,
        "state": session.get("state"),
        "google_connected": GOOGLE_INITIALIZED
    }

# ============================================
# API Models
# ============================================
class TextInput(BaseModel):
    user_id: str
    text: str

# ============================================
# API Endpoints
# ============================================
@app.get("/")
def root():
    return {
        "message": "Enhanced AI Assistant with Google Integration",
        "version": "3.0.0",
        "model_type": "fine_tuned" if USE_FINE_TUNED else "zero_shot",
        "google_connected": GOOGLE_INITIALIZED,
        "stt_available": STT_PIPELINE is not None
    }

@app.post("/infer")
def infer(req: TextInput):
    """Process text input and return assistant response."""
    return process_input(req.user_id, req.text)

@app.post("/voice")
async def voice_infer(
    audio: UploadFile = File(...),
    user_id: Optional[str] = Form(default=None)
):
    """Process voice input - accepts various audio formats."""
    if STT_PIPELINE is None:
        raise HTTPException(status_code=503, detail="Speech-to-text not available")
    
    # Default user_id if not provided
    if not user_id:
        user_id = "default_user"
    
    # Determine file extension from content type or filename
    content_type = audio.content_type or ""
    filename = audio.filename or "audio.wav"
    
    if "webm" in content_type or filename.endswith(".webm"):
        suffix = ".webm"
    elif "mp3" in content_type or filename.endswith(".mp3"):
        suffix = ".mp3"
    elif "ogg" in content_type or filename.endswith(".ogg"):
        suffix = ".ogg"
    elif "m4a" in content_type or filename.endswith(".m4a"):
        suffix = ".m4a"
    else:
        suffix = ".wav"
    
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Whisper can handle multiple formats directly
        result = STT_PIPELINE(tmp_path)
        transcription = result["text"].strip()
        
        if not transcription or len(transcription.strip()) < 1:
            return {
                "transcription": "",
                "response": "I couldn't hear anything. Please try again.",
                "intent": None,
                "confidence": 0,
                "entities": {},
                "state": "idle",
                "google_connected": GOOGLE_INITIALIZED
            }
        
        # Get current session state FIRST to check if we're in a conversation flow
        session = get_user_session(user_id)
        state = session.get("state", ConversationState.IDLE)
        pending_action = session.get("pending_action")
        
        # If we're waiting for an email, handle speech-to-text email conversion
        if state in [ConversationState.AWAITING_YOUR_EMAIL, ConversationState.AWAITING_ATTENDEE_EMAIL, 
                     ConversationState.AWAITING_REMINDER_EMAIL, ConversationState.AWAITING_CANCEL_EMAIL,
                     ConversationState.AWAITING_CALENDAR_EMAIL]:
            transcription = convert_spoken_email(transcription)
        else:
            # For non-email contexts, just clean up filler words gently
            transcription = re.sub(r'\b(um|uh|ah|er)\b\s*', '', transcription, flags=re.IGNORECASE)
            transcription = re.sub(r'\s+', ' ', transcription).strip()
        
        response = process_input(user_id, transcription)
        response["transcription"] = transcription
        return response
    except Exception as e:
        print(f"STT Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

@app.get("/context/{user_id}")
def get_context(user_id: str):
    """Get user session context."""
    return get_user_session(user_id)

@app.get("/reset/{user_id}")
def reset_user(user_id: str):
    """Reset user session."""
    if user_id in user_sessions:
        del user_sessions[user_id]
    return {"message": f"✅ Session reset for user '{user_id}'"}

@app.get("/auth/google")
def auth_google():
    """Initiate Google OAuth flow."""
    if not GOOGLE_AVAILABLE:
        return {"error": "Google integration not available. Install google-api-python-client."}
    
    if not os.path.exists("credentials.json"):
        return {
            "error": "credentials.json not found",
            "instructions": """
To set up Google integration:
1. Go to https://console.cloud.google.com
2. Create/select a project
3. Enable Google Calendar API and Gmail API
4. Create OAuth 2.0 credentials (Desktop App)
5. Download and rename to 'credentials.json'
6. Place in the Ai Agent folder
7. Restart the server
"""
        }
    
    global GOOGLE_MANAGER, GOOGLE_INITIALIZED
    GOOGLE_MANAGER = get_google_manager()
    
    if GOOGLE_MANAGER.initialize():
        GOOGLE_INITIALIZED = True
        return {"message": "✅ Successfully connected to Google services!"}
    else:
        return {"error": "Authentication failed. Check console for OAuth flow."}

@app.get("/auth/google/status")
def google_status():
    """Check Google authentication status."""
    return {
        "google_available": GOOGLE_AVAILABLE,
        "google_initialized": GOOGLE_INITIALIZED,
        "credentials_found": os.path.exists("credentials.json")
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": USE_FINE_TUNED or ZERO_SHOT_CLASSIFIER is not None,
        "stt_loaded": STT_PIPELINE is not None,
        "google_connected": GOOGLE_INITIALIZED,
        "active_sessions": len(user_sessions)
    }

# Static files for UI
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    @app.get("/ui")
    def serve_ui():
        return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
