"""
Advanced Entity Extraction Module
=================================
Uses spaCy NER + custom patterns for comprehensive entity extraction.
Handles: Names, Emails, Phone Numbers, Dates, Times, Organizations, Locations
"""

import re
import spacy
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import dateparser
from dateparser.search import search_dates

# ============================================
# Load spaCy Model
# ============================================
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded")
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model downloaded and loaded")

# ============================================
# Regex Patterns
# ============================================
PATTERNS = {
    # Email pattern - comprehensive
    'email': re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    ),
    
    # Phone patterns - various formats
    'phone': re.compile(
        r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}|'
        r'\b[0-9]{10}\b'
    ),
    
    # Time patterns - 12hr and 24hr formats
    'time_12hr': re.compile(
        r'\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)\b',
        re.IGNORECASE
    ),
    'time_24hr': re.compile(
        r'\b([01]?[0-9]|2[0-3]):([0-5][0-9])\b'
    ),
    
    # Duration patterns
    'duration': re.compile(
        r'\b(\d+)\s*(hour|hr|minute|min|second|sec)s?\b',
        re.IGNORECASE
    ),
    
    # URL pattern
    'url': re.compile(
        r'https?://(?:[-\w.])+(?:[:\d])*(?:/(?:[\w/_.])*(?:\?(?:[\w&=])*)?)?'
    )
}

# ============================================
# Time/Date Keywords
# ============================================
RELATIVE_TIME_MAP = {
    # Relative days
    'today': lambda: datetime.now().replace(hour=9, minute=0, second=0, microsecond=0),
    'tomorrow': lambda: (datetime.now() + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0),
    'yesterday': lambda: (datetime.now() - timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0),
    'tmrw': lambda: (datetime.now() + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0),
    'tmr': lambda: (datetime.now() + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0),
    
    # Parts of day
    'morning': lambda: datetime.now().replace(hour=9, minute=0, second=0, microsecond=0),
    'afternoon': lambda: datetime.now().replace(hour=14, minute=0, second=0, microsecond=0),
    'evening': lambda: datetime.now().replace(hour=18, minute=0, second=0, microsecond=0),
    'night': lambda: datetime.now().replace(hour=20, minute=0, second=0, microsecond=0),
    'noon': lambda: datetime.now().replace(hour=12, minute=0, second=0, microsecond=0),
    'midnight': lambda: datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
    
    # Relative time
    'now': lambda: datetime.now(),
    'later': lambda: datetime.now() + timedelta(hours=2),
    'soon': lambda: datetime.now() + timedelta(hours=1),
}

FUZZY_TIME_PATTERNS = {
    r'\b(\d{1,2})\s*(?:ish|ish)\b': lambda m: f"{m.group(1)}:00",
    r'\bafter\s+lunch\b': lambda m: "14:00",
    r'\blate\s+afternoon\b': lambda m: "16:30",
    r'\bearly\s+morning\b': lambda m: "07:00",
    r'\blate\s+evening\b': lambda m: "21:00",
    r'\blate\s+night\b': lambda m: "23:00",
    r'\bin\s+(\d+)\s*(?:hour|hr)s?\b': lambda m: (datetime.now() + timedelta(hours=int(m.group(1)))).strftime("%H:%M"),
    r'\bin\s+(\d+)\s*(?:minute|min)s?\b': lambda m: (datetime.now() + timedelta(minutes=int(m.group(1)))).strftime("%H:%M"),
}

# ============================================
# Entity Extractor Class
# ============================================
class EntityExtractor:
    """
    Advanced entity extraction using spaCy NER and custom patterns.
    """
    
    def __init__(self):
        self.nlp = nlp
        # Custom stopwords for name filtering
        self.name_stopwords = {
            'i', 'me', 'my', 'you', 'your', 'he', 'she', 'it', 'we', 'they',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'schedule', 'meeting', 'remind', 'reminder', 'email', 'send',
            'call', 'book', 'set', 'cancel', 'delete', 'check', 'show',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            'today', 'tomorrow', 'yesterday', 'morning', 'afternoon', 'evening', 'night',
            'pm', 'am', 'hello', 'hi', 'hey', 'please', 'thanks', 'thank'
        }
    
    def extract_all(self, text: str) -> Dict[str, Any]:
        """
        Extract all entities from text.
        
        Returns:
            Dict containing all extracted entities with confidence scores.
        """
        doc = self.nlp(text)
        
        result = {
            'persons': self._extract_persons(doc, text),
            'emails': self._extract_emails(text),
            'phones': self._extract_phones(text),
            'datetime': self._extract_datetime(text),
            'organizations': self._extract_organizations(doc),
            'locations': self._extract_locations(doc),
            'urls': self._extract_urls(text),
            'durations': self._extract_durations(text),
            'raw_entities': self._get_raw_spacy_entities(doc)
        }
        
        # Add summary
        result['summary'] = self._create_summary(result)
        
        return result
    
    def _extract_persons(self, doc, text: str) -> List[Dict[str, Any]]:
        """Extract person names using spaCy + patterns."""
        persons = []
        seen = set()
        
        # 1. spaCy NER for PERSON entities
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                name = ent.text.strip()
                if name.lower() not in self.name_stopwords and name not in seen:
                    persons.append({
                        'name': name,
                        'source': 'spacy_ner',
                        'confidence': 0.9,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
                    seen.add(name)
        
        # 2. Pattern matching for names after keywords
        name_patterns = [
            r'\bwith\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            r'\bto\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            r'\bfor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            r'\btell\s+([A-Z][a-z]+)\b',
            r'\bnotify\s+([A-Z][a-z]+)\b',
            r'\bmessage\s+([A-Z][a-z]+)\b',
            r'\bemail\s+([A-Z][a-z]+)\b',
            r'\bcall\s+([A-Z][a-z]+)\b',
        ]
        
        for pattern in name_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                name = match.group(1).strip()
                if name.lower() not in self.name_stopwords and name not in seen:
                    persons.append({
                        'name': name,
                        'source': 'pattern_match',
                        'confidence': 0.75,
                        'start': match.start(1),
                        'end': match.end(1)
                    })
                    seen.add(name)
        
        # 3. Lowercase name extraction (with keywords)
        lowercase_pattern = r'\b(?:with|to|for|tell|notify|message|email|call)\s+([a-z][a-z]+)\b'
        for match in re.finditer(lowercase_pattern, text.lower()):
            name = match.group(1).capitalize()
            if name.lower() not in self.name_stopwords and name not in seen:
                persons.append({
                    'name': name,
                    'source': 'lowercase_match',
                    'confidence': 0.6,
                    'start': match.start(1),
                    'end': match.end(1)
                })
                seen.add(name)
        
        # Sort by confidence
        persons.sort(key=lambda x: x['confidence'], reverse=True)
        return persons
    
    def _extract_emails(self, text: str) -> List[Dict[str, Any]]:
        """Extract email addresses."""
        emails = []
        for match in PATTERNS['email'].finditer(text):
            emails.append({
                'email': match.group(),
                'confidence': 1.0,
                'start': match.start(),
                'end': match.end()
            })
        return emails
    
    def _extract_phones(self, text: str) -> List[Dict[str, Any]]:
        """Extract phone numbers."""
        phones = []
        for match in PATTERNS['phone'].finditer(text):
            # Clean phone number
            phone = re.sub(r'[^\d]', '', match.group())
            if len(phone) >= 10:
                phones.append({
                    'phone': phone,
                    'formatted': self._format_phone(phone),
                    'confidence': 0.95,
                    'start': match.start(),
                    'end': match.end()
                })
        return phones
    
    def _format_phone(self, phone: str) -> str:
        """Format phone number."""
        if len(phone) == 10:
            return f"({phone[:3]}) {phone[3:6]}-{phone[6:]}"
        elif len(phone) == 11 and phone[0] == '1':
            return f"+1 ({phone[1:4]}) {phone[4:7]}-{phone[7:]}"
        return phone
    
    def _extract_datetime(self, text: str) -> Dict[str, Any]:
        """Extract date and time information."""
        result = {
            'date': None,
            'time': None,
            'datetime_iso': None,
            'relative_reference': None,
            'confidence': 0.0
        }
        
        text_lower = text.lower()
        parsed_dt = None
        
        # 1. Check for relative time keywords
        for keyword, time_func in RELATIVE_TIME_MAP.items():
            if keyword in text_lower:
                result['relative_reference'] = keyword
                parsed_dt = time_func()
                result['confidence'] = 0.9
                break
        
        # 2. Extract explicit time (12hr format)
        time_match = PATTERNS['time_12hr'].search(text)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2) or 0)
            ampm = time_match.group(3).lower().replace('.', '')
            
            if ampm == 'pm' and hour != 12:
                hour += 12
            elif ampm == 'am' and hour == 12:
                hour = 0
            
            base_date = parsed_dt.date() if parsed_dt else datetime.now().date()
            parsed_dt = datetime.combine(base_date, datetime.min.time().replace(hour=hour, minute=minute))
            result['time'] = f"{hour:02d}:{minute:02d}"
            result['confidence'] = max(result['confidence'], 0.95)
        
        # 3. Extract explicit time (24hr format)
        if not time_match:
            time_match_24 = PATTERNS['time_24hr'].search(text)
            if time_match_24:
                hour = int(time_match_24.group(1))
                minute = int(time_match_24.group(2))
                base_date = parsed_dt.date() if parsed_dt else datetime.now().date()
                parsed_dt = datetime.combine(base_date, datetime.min.time().replace(hour=hour, minute=minute))
                result['time'] = f"{hour:02d}:{minute:02d}"
                result['confidence'] = max(result['confidence'], 0.95)
        
        # 4. Check fuzzy time patterns
        for pattern, time_func in FUZZY_TIME_PATTERNS.items():
            match = re.search(pattern, text_lower)
            if match:
                time_str = time_func(match)
                if ':' in time_str:
                    parts = time_str.split(':')
                    hour, minute = int(parts[0]), int(parts[1])
                    base_date = parsed_dt.date() if parsed_dt else datetime.now().date()
                    parsed_dt = datetime.combine(base_date, datetime.min.time().replace(hour=hour, minute=minute))
                    result['time'] = time_str
                    result['confidence'] = max(result['confidence'], 0.8)
                break
        
        # 5. Use dateparser for complex date expressions
        if not parsed_dt or result['confidence'] < 0.7:
            try:
                parsed = dateparser.parse(
                    text,
                    settings={
                        'PREFER_DATES_FROM': 'future',
                        'RELATIVE_BASE': datetime.now()
                    }
                )
                if parsed:
                    parsed_dt = parsed
                    result['confidence'] = max(result['confidence'], 0.7)
            except:
                pass
        
        # 6. Try search_dates for multiple dates
        try:
            dates_found = search_dates(text, settings={'PREFER_DATES_FROM': 'future'})
            if dates_found and len(dates_found) > 0:
                _, parsed = dates_found[0]
                if not parsed_dt:
                    parsed_dt = parsed
                    result['confidence'] = max(result['confidence'], 0.65)
        except:
            pass
        
        # Final result
        if parsed_dt:
            result['date'] = parsed_dt.strftime('%Y-%m-%d')
            if result['time'] is None:
                result['time'] = parsed_dt.strftime('%H:%M')
            result['datetime_iso'] = parsed_dt.isoformat()
        
        return result
    
    def _extract_organizations(self, doc) -> List[Dict[str, Any]]:
        """Extract organization names."""
        orgs = []
        seen = set()
        
        for ent in doc.ents:
            if ent.label_ == 'ORG' and ent.text not in seen:
                orgs.append({
                    'name': ent.text,
                    'confidence': 0.85,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
                seen.add(ent.text)
        
        return orgs
    
    def _extract_locations(self, doc) -> List[Dict[str, Any]]:
        """Extract location names."""
        locations = []
        seen = set()
        
        for ent in doc.ents:
            if ent.label_ in ('GPE', 'LOC', 'FAC') and ent.text not in seen:
                locations.append({
                    'name': ent.text,
                    'type': ent.label_,
                    'confidence': 0.85,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
                seen.add(ent.text)
        
        return locations
    
    def _extract_urls(self, text: str) -> List[Dict[str, Any]]:
        """Extract URLs."""
        urls = []
        for match in PATTERNS['url'].finditer(text):
            urls.append({
                'url': match.group(),
                'confidence': 1.0,
                'start': match.start(),
                'end': match.end()
            })
        return urls
    
    def _extract_durations(self, text: str) -> List[Dict[str, Any]]:
        """Extract duration mentions."""
        durations = []
        for match in PATTERNS['duration'].finditer(text):
            value = int(match.group(1))
            unit = match.group(2).lower()
            
            # Normalize unit
            if unit in ('hr', 'hour'):
                unit = 'hours'
            elif unit in ('min', 'minute'):
                unit = 'minutes'
            elif unit in ('sec', 'second'):
                unit = 'seconds'
            
            durations.append({
                'value': value,
                'unit': unit,
                'total_minutes': self._to_minutes(value, unit),
                'confidence': 0.9,
                'start': match.start(),
                'end': match.end()
            })
        
        return durations
    
    def _to_minutes(self, value: int, unit: str) -> int:
        """Convert duration to minutes."""
        if unit == 'hours':
            return value * 60
        elif unit == 'minutes':
            return value
        elif unit == 'seconds':
            return value // 60
        return value
    
    def _get_raw_spacy_entities(self, doc) -> List[Dict[str, str]]:
        """Get all raw spaCy entities."""
        return [
            {
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_)
            }
            for ent in doc.ents
        ]
    
    def _create_summary(self, result: Dict) -> Dict[str, Any]:
        """Create a summary of extracted entities."""
        return {
            'has_person': len(result['persons']) > 0,
            'has_email': len(result['emails']) > 0,
            'has_phone': len(result['phones']) > 0,
            'has_datetime': result['datetime']['datetime_iso'] is not None,
            'has_organization': len(result['organizations']) > 0,
            'has_location': len(result['locations']) > 0,
            'primary_person': result['persons'][0]['name'] if result['persons'] else None,
            'primary_email': result['emails'][0]['email'] if result['emails'] else None,
            'primary_datetime': result['datetime']['datetime_iso']
        }


# ============================================
# Simplified extraction function (for API use)
# ============================================
_extractor = None

def get_extractor():
    """Get or create singleton extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor()
    return _extractor

def extract_entities(text: str) -> Dict[str, Any]:
    """
    Simple function to extract entities from text.
    
    Args:
        text: Input text to extract entities from
        
    Returns:
        Dict with extracted entities
    """
    extractor = get_extractor()
    return extractor.extract_all(text)

def extract_entities_simple(text: str) -> Dict[str, Any]:
    """
    Extract entities in simplified format (backward compatible).
    
    Returns:
        Dict with: names, emails, datetime (simplified)
    """
    full_result = extract_entities(text)
    
    return {
        'names': [p['name'] for p in full_result['persons']] or None,
        'emails': [e['email'] for e in full_result['emails']] or None,
        'datetime': full_result['datetime']['datetime_iso'],
        'phones': [p['formatted'] for p in full_result['phones']] or None,
        'organizations': [o['name'] for o in full_result['organizations']] or None,
        'locations': [l['name'] for l in full_result['locations']] or None
    }


# ============================================
# Test the extractor
# ============================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 ENTITY EXTRACTION TEST")
    print("="*60 + "\n")
    
    test_sentences = [
        "Schedule a meeting with John tomorrow at 3pm",
        "Send an email to sarah@company.com about the project",
        "Remind me to call Mike at 5pm today",
        "Book a meeting with Alice and Bob for next Monday morning",
        "Email john.doe@gmail.com the report by Friday afternoon",
        "Set up a call with the Google team at 10:30am",
        "Tell Sarah I'll be late by 30 minutes",
        "Meeting with Dr. Smith at Microsoft headquarters tomorrow at 2pm",
        "Cancel my 3pm appointment with Jane",
        "Call me back at 555-123-4567 in 2 hours",
        "tmrw at 5ish let's meet with Bob",
        "Schedule meeting with john after lunch",
        "Remind me in 30 minutes to check emails"
    ]
    
    extractor = EntityExtractor()
    
    for sentence in test_sentences:
        print(f"📝 Input: \"{sentence}\"")
        result = extractor.extract_all(sentence)
        
        print(f"   👤 Persons: {[p['name'] for p in result['persons']]}")
        print(f"   📧 Emails: {[e['email'] for e in result['emails']]}")
        print(f"   📅 DateTime: {result['datetime']['datetime_iso']}")
        print(f"   🏢 Organizations: {[o['name'] for o in result['organizations']]}")
        print(f"   📍 Locations: {[l['name'] for l in result['locations']]}")
        print(f"   📞 Phones: {[p['formatted'] for p in result['phones']]}")
        print()

