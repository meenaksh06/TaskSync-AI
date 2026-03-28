"""
Google Services Integration Module
===================================
Handles Google Calendar, Gmail, and Google Meet integration.

Features:
- OAuth 2.0 authentication
- Create Google Calendar events with Meet links
- Send email invitations via Gmail
- Add reminders to Google Calendar

Setup Required:
1. Go to Google Cloud Console (https://console.cloud.google.com)
2. Create a new project or select existing
3. Enable APIs: Google Calendar API, Gmail API
4. Create OAuth 2.0 credentials (Desktop App)
5. Download credentials.json and place in this folder
"""

import os
import json
import pickle
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from pathlib import Path

# Google API imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ============================================
# Configuration
# ============================================

# Scopes required for Calendar and Gmail
SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/calendar.events',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.compose'
]

# File paths
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.pickle'

# ============================================
# Google Authentication
# ============================================
class GoogleAuth:
    """Handle Google OAuth 2.0 authentication."""
    
    def __init__(self, credentials_file: str = CREDENTIALS_FILE, token_file: str = TOKEN_FILE):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.creds = None
        self._load_credentials()
    
    def _load_credentials(self):
        """Load existing credentials from token file."""
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                self.creds = pickle.load(token)
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        return self.creds is not None and self.creds.valid
    
    def needs_refresh(self) -> bool:
        """Check if credentials need refresh."""
        return self.creds is not None and self.creds.expired and self.creds.refresh_token
    
    def authenticate(self) -> bool:
        """
        Authenticate with Google APIs.
        Returns True if authentication successful.
        """
        try:
            # If credentials exist and are valid
            if self.creds and self.creds.valid:
                return True
            
            # If credentials exist but expired, refresh them
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
                self._save_credentials()
                return True
            
            # Check if credentials file exists
            if not os.path.exists(self.credentials_file):
                print(f"⚠️  {self.credentials_file} not found!")
                print("   Please download OAuth credentials from Google Cloud Console")
                return False
            
            # Run OAuth flow
            flow = InstalledAppFlow.from_client_secrets_file(
                self.credentials_file, SCOPES
            )
            self.creds = flow.run_local_server(port=8080)
            self._save_credentials()
            
            return True
            
        except Exception as e:
            print(f"Authentication error: {e}")
            return False
    
    def _save_credentials(self):
        """Save credentials to token file."""
        with open(self.token_file, 'wb') as token:
            pickle.dump(self.creds, token)
    
    def get_credentials(self) -> Optional[Credentials]:
        """Get the current credentials."""
        return self.creds
    
    def logout(self):
        """Remove stored credentials."""
        if os.path.exists(self.token_file):
            os.remove(self.token_file)
        self.creds = None


# ============================================
# Google Calendar Service
# ============================================
class GoogleCalendarService:
    """Handle Google Calendar operations."""
    
    def __init__(self, auth: GoogleAuth):
        self.auth = auth
        self.service = None
        self._build_service()
    
    def _build_service(self):
        """Build the Calendar API service."""
        if self.auth.is_authenticated():
            self.service = build('calendar', 'v3', credentials=self.auth.get_credentials())
    
    def create_meeting_with_meet(
        self,
        title: str,
        start_time: datetime,
        end_time: datetime,
        attendees: List[str],
        description: str = "",
        timezone: str = "Asia/Kolkata"
    ) -> Dict[str, Any]:
        """
        Create a Google Calendar event with Google Meet link.
        
        Args:
            title: Meeting title
            start_time: Meeting start datetime
            end_time: Meeting end datetime  
            attendees: List of attendee email addresses
            description: Optional meeting description
            timezone: Timezone for the meeting
            
        Returns:
            Dict with event details including Meet link
        """
        if not self.service:
            return {"success": False, "error": "Not authenticated with Google"}
        
        try:
            # Create event with Google Meet
            event = {
                'summary': title,
                'description': description,
                'start': {
                    'dateTime': start_time.isoformat(),
                    'timeZone': timezone,
                },
                'end': {
                    'dateTime': end_time.isoformat(),
                    'timeZone': timezone,
                },
                'attendees': [{'email': email} for email in attendees],
                'conferenceData': {
                    'createRequest': {
                        'requestId': f"meet-{datetime.now().timestamp()}",
                        'conferenceSolutionKey': {'type': 'hangoutsMeet'}
                    }
                },
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'email', 'minutes': 30},
                        {'method': 'popup', 'minutes': 10},
                    ],
                },
            }
            
            # Insert event with conference data
            event_result = self.service.events().insert(
                calendarId='primary',
                body=event,
                conferenceDataVersion=1,
                sendUpdates='all'  # Send email invites to attendees
            ).execute()
            
            # Extract Meet link
            meet_link = None
            if 'conferenceData' in event_result:
                for entry_point in event_result['conferenceData'].get('entryPoints', []):
                    if entry_point.get('entryPointType') == 'video':
                        meet_link = entry_point.get('uri')
                        break
            
            return {
                "success": True,
                "event_id": event_result.get('id'),
                "event_link": event_result.get('htmlLink'),
                "meet_link": meet_link,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "attendees": attendees
            }
            
        except HttpError as error:
            return {"success": False, "error": str(error)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def add_reminder(
        self,
        title: str,
        reminder_time: datetime,
        description: str = "",
        timezone: str = "Asia/Kolkata"
    ) -> Dict[str, Any]:
        """
        Add a reminder to Google Calendar.
        
        Args:
            title: Reminder title/task
            reminder_time: When to be reminded
            description: Optional description
            timezone: Timezone
            
        Returns:
            Dict with reminder details
        """
        if not self.service:
            return {"success": False, "error": "Not authenticated with Google"}
        
        try:
            # Create an all-day event or timed reminder
            event = {
                'summary': f"⏰ Reminder: {title}",
                'description': description or f"Reminder set by AI Assistant: {title}",
                'start': {
                    'dateTime': reminder_time.isoformat(),
                    'timeZone': timezone,
                },
                'end': {
                    'dateTime': (reminder_time + timedelta(minutes=30)).isoformat(),
                    'timeZone': timezone,
                },
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'popup', 'minutes': 0},  # At time of event
                        {'method': 'email', 'minutes': 5},
                    ],
                },
                'colorId': '11',  # Red color for reminders
            }
            
            event_result = self.service.events().insert(
                calendarId='primary',
                body=event
            ).execute()
            
            return {
                "success": True,
                "event_id": event_result.get('id'),
                "event_link": event_result.get('htmlLink'),
                "reminder_time": reminder_time.isoformat(),
                "title": title
            }
            
        except HttpError as error:
            return {"success": False, "error": str(error)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_upcoming_events(self, max_results: int = 10) -> List[Dict]:
        """Get upcoming calendar events."""
        if not self.service:
            return []
        
        try:
            now = datetime.utcnow().isoformat() + 'Z'
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=now,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            return [
                {
                    'id': event.get('id'),
                    'summary': event.get('summary'),
                    'start': event.get('start', {}).get('dateTime', event.get('start', {}).get('date')),
                    'end': event.get('end', {}).get('dateTime', event.get('end', {}).get('date')),
                    'meet_link': event.get('hangoutLink'),
                    'attendees': [a.get('email') for a in event.get('attendees', [])]
                }
                for event in events
            ]
            
        except Exception as e:
            print(f"Error fetching events: {e}")
            return []
    
    def delete_event(self, event_id: str) -> bool:
        """Delete a calendar event."""
        if not self.service:
            return False
        
        try:
            self.service.events().delete(
                calendarId='primary',
                eventId=event_id
            ).execute()
            return True
        except Exception as e:
            print(f"Error deleting event: {e}")
            return False


# ============================================
# Gmail Service
# ============================================
class GmailService:
    """Handle Gmail operations for sending meeting invites."""
    
    def __init__(self, auth: GoogleAuth):
        self.auth = auth
        self.service = None
        self._build_service()
    
    def _build_service(self):
        """Build the Gmail API service."""
        if self.auth.is_authenticated():
            self.service = build('gmail', 'v1', credentials=self.auth.get_credentials())
    
    def send_meeting_invite(
        self,
        to_emails: List[str],
        subject: str,
        meeting_time: str,
        meet_link: str,
        organizer_name: str = "AI Assistant"
    ) -> Dict[str, Any]:
        """
        Send meeting invitation emails.
        
        Args:
            to_emails: List of recipient emails
            subject: Email subject
            meeting_time: Meeting time string
            meet_link: Google Meet link
            organizer_name: Name of meeting organizer
            
        Returns:
            Dict with send status
        """
        if not self.service:
            return {"success": False, "error": "Not authenticated with Gmail"}
        
        try:
            # Create HTML email body
            html_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px 10px 0 0;">
                    <h1 style="color: white; margin: 0;">📅 Meeting Invitation</h1>
                </div>
                <div style="padding: 30px; background: #f9f9f9; border-radius: 0 0 10px 10px;">
                    <h2 style="color: #333;">{subject}</h2>
                    <p style="color: #666; font-size: 16px;">You have been invited to a meeting.</p>
                    
                    <div style="background: white; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #667eea;">
                        <p style="margin: 5px 0;"><strong>🕐 Time:</strong> {meeting_time}</p>
                        <p style="margin: 5px 0;"><strong>👤 Organizer:</strong> {organizer_name}</p>
                    </div>
                    
                    <a href="{meet_link}" style="display: inline-block; background: #667eea; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; font-weight: bold; margin-top: 10px;">
                        🎥 Join Google Meet
                    </a>
                    
                    <p style="color: #999; font-size: 12px; margin-top: 30px;">
                        This meeting was scheduled using AI Assistant.
                    </p>
                </div>
            </body>
            </html>
            """
            
            results = []
            for email in to_emails:
                message = MIMEMultipart('alternative')
                message['to'] = email
                message['subject'] = f"Meeting Invite: {subject}"
                
                # Attach HTML content
                html_part = MIMEText(html_body, 'html')
                message.attach(html_part)
                
                # Encode message
                raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
                
                # Send email
                self.service.users().messages().send(
                    userId='me',
                    body={'raw': raw_message}
                ).execute()
                
                results.append({"email": email, "status": "sent"})
            
            return {
                "success": True,
                "sent_to": results
            }
            
        except HttpError as error:
            return {"success": False, "error": str(error)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def send_reminder_confirmation(
        self,
        to_email: str,
        reminder_text: str,
        reminder_time: str
    ) -> Dict[str, Any]:
        """Send confirmation email for a reminder."""
        if not self.service:
            return {"success": False, "error": "Not authenticated with Gmail"}
        
        try:
            html_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background: #22c55e; padding: 30px; border-radius: 10px 10px 0 0;">
                    <h1 style="color: white; margin: 0;">⏰ Reminder Set</h1>
                </div>
                <div style="padding: 30px; background: #f9f9f9; border-radius: 0 0 10px 10px;">
                    <h2 style="color: #333;">Your reminder has been added to Google Calendar</h2>
                    
                    <div style="background: white; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #22c55e;">
                        <p style="margin: 5px 0;"><strong>📝 Task:</strong> {reminder_text}</p>
                        <p style="margin: 5px 0;"><strong>🕐 Time:</strong> {reminder_time}</p>
                    </div>
                    
                    <p style="color: #666;">You'll receive a notification at the scheduled time.</p>
                </div>
            </body>
            </html>
            """
            
            message = MIMEMultipart('alternative')
            message['to'] = to_email
            message['subject'] = f"⏰ Reminder Set: {reminder_text}"
            
            html_part = MIMEText(html_body, 'html')
            message.attach(html_part)
            
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
            
            self.service.users().messages().send(
                userId='me',
                body={'raw': raw_message}
            ).execute()
            
            return {"success": True, "sent_to": to_email}
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================
# Main Google Services Manager
# ============================================
class GoogleServicesManager:
    """
    Main manager class for all Google services.
    Provides a unified interface for Calendar and Gmail operations.
    """
    
    def __init__(self):
        self.auth = GoogleAuth()
        self.calendar = None
        self.gmail = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize all Google services."""
        if not self.auth.authenticate():
            print("Failed to authenticate with Google")
            return False
        
        self.calendar = GoogleCalendarService(self.auth)
        self.gmail = GmailService(self.auth)
        self._initialized = True
        print("Google services initialized")
        return True
    
    def is_initialized(self) -> bool:
        """Check if services are initialized."""
        return self._initialized
    
    def schedule_meeting(
        self,
        title: str,
        start_time: datetime,
        duration_minutes: int,
        attendee_emails: List[str],
        description: str = "",
        send_email_invite: bool = True
    ) -> Dict[str, Any]:
        """
        Schedule a meeting with Google Meet and send invites.
        
        Args:
            title: Meeting title
            start_time: Meeting start time
            duration_minutes: Meeting duration
            attendee_emails: List of attendee emails
            description: Meeting description
            send_email_invite: Whether to send custom email invites
            
        Returns:
            Dict with meeting details
        """
        if not self._initialized:
            return {"success": False, "error": "Google services not initialized"}
        
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        # Create calendar event with Meet
        event_result = self.calendar.create_meeting_with_meet(
            title=title,
            start_time=start_time,
            end_time=end_time,
            attendees=attendee_emails,
            description=description
        )
        
        if not event_result.get("success"):
            return event_result
        
        # Optionally send custom email invites
        if send_email_invite and event_result.get("meet_link"):
            email_result = self.gmail.send_meeting_invite(
                to_emails=attendee_emails,
                subject=title,
                meeting_time=start_time.strftime("%B %d, %Y at %I:%M %p"),
                meet_link=event_result["meet_link"]
            )
            event_result["email_sent"] = email_result.get("success", False)
        
        return event_result
    
    def add_reminder(
        self,
        title: str,
        reminder_time: datetime,
        user_email: Optional[str] = None,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Add a reminder to Google Calendar.
        
        Args:
            title: Reminder text
            reminder_time: When to remind
            user_email: Email for confirmation (optional)
            description: Additional description
            
        Returns:
            Dict with reminder details
        """
        if not self._initialized:
            return {"success": False, "error": "Google services not initialized"}
        
        # Add to calendar
        result = self.calendar.add_reminder(
            title=title,
            reminder_time=reminder_time,
            description=description
        )
        
        # Send confirmation email if email provided
        if result.get("success") and user_email:
            self.gmail.send_reminder_confirmation(
                to_email=user_email,
                reminder_text=title,
                reminder_time=reminder_time.strftime("%B %d, %Y at %I:%M %p")
            )
            result["confirmation_sent"] = True
        
        return result
    
    def get_upcoming_meetings(self, count: int = 10) -> List[Dict]:
        """Get upcoming meetings from calendar."""
        if not self._initialized:
            return []
        return self.calendar.get_upcoming_events(count)
    
    def cancel_meeting(self, event_id: str) -> bool:
        """Cancel a meeting by event ID."""
        if not self._initialized:
            return False
        return self.calendar.delete_event(event_id)
    
    def logout(self):
        """Logout from Google services."""
        self.auth.logout()
        self._initialized = False


# ============================================
# Singleton instance for app use
# ============================================
_google_manager: Optional[GoogleServicesManager] = None

def get_google_manager() -> GoogleServicesManager:
    """Get or create the Google services manager singleton."""
    global _google_manager
    if _google_manager is None:
        _google_manager = GoogleServicesManager()
    return _google_manager


if __name__ == "__main__":
    print("Google Services Integration Test")
    
    manager = get_google_manager()
    
    if not os.path.exists(CREDENTIALS_FILE):
        print(f"{CREDENTIALS_FILE} not found. Download OAuth credentials from Google Cloud Console.")
    else:
        print("Found credentials.json, attempting authentication...")
        
        if manager.initialize():
            print("Successfully connected to Google services!")
            print("Your upcoming events:")
            
            events = manager.get_upcoming_meetings(5)
            if events:
                for event in events:
                    print(f"  - {event['summary']} - {event['start']}")
            else:
                print("  No upcoming events found.")
        else:
            print("Failed to initialize Google services.")

