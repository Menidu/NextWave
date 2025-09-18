import re
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any


WEEKDAYS = {
    'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
    'friday': 4, 'saturday': 5, 'sunday': 6
}

DATE_FORMATS = [
    "%Y-%m-%d",
    "%B %d, %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%d %b %Y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%d/%m/%y",
    "%m/%d/%y",
    "%d-%m-%y",
    "%m-%d-%y",
]


def extract_email(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, re.I)
    return match.group(0) if match else None


def clean_json_response(content: str) -> str:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    start_idx = content.find('{')
    end_idx = content.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        content = content[start_idx:end_idx + 1]
    return content


def parse_date_flexible(date_input: str) -> Tuple[Optional[str], Optional[str]]:
    if not date_input or not date_input.strip():
        return None, "Date cannot be empty"
    date_input = date_input.strip().lower()
    today = datetime.now()
    try:
        relative_dates = {
            "today": 0,
            "tomorrow": 1,
            "day after tomorrow": 2,
            "yesterday": -1,
        }
        if date_input in relative_dates:
            days_offset = relative_dates[date_input]
            if days_offset < 0:
                return None, "Leave cannot be scheduled for past dates"
            target_date = today + timedelta(days=days_offset)
            return target_date.strftime("%Y-%m-%d"), None

        next_weekday_match = re.search(r'next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', date_input)
        if next_weekday_match:
            target_weekday = WEEKDAYS[next_weekday_match.group(1)]
            days_ahead = target_weekday - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            target_date = today + timedelta(days=days_ahead)
            return target_date.strftime("%Y-%m-%d"), None

        this_weekday_match = re.search(r'this\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', date_input)
        if this_weekday_match:
            target_weekday = WEEKDAYS[this_weekday_match.group(1)]
            days_ahead = target_weekday - today.weekday()
            if days_ahead < 0:
                return None, f"This {this_weekday_match.group(1)} has already passed. Did you mean next {this_weekday_match.group(1)}?"
            target_date = today + timedelta(days=days_ahead)
            return target_date.strftime("%Y-%m-%d"), None

        in_days_match = re.search(r'in\s+(\d+)\s+days?', date_input)
        if in_days_match:
            days = int(in_days_match.group(1))
            target_date = today + timedelta(days=days)
            return target_date.strftime("%Y-%m-%d"), None

        in_weeks_match = re.search(r'in\s+(\d+)\s+weeks?', date_input)
        if in_weeks_match:
            weeks = int(in_weeks_match.group(1))
            target_date = today + timedelta(weeks=weeks)
            return target_date.strftime("%Y-%m-%d"), None

        ordinal_cleaned = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_input)
        for fmt in DATE_FORMATS:
            try:
                parsed_date = datetime.strptime(ordinal_cleaned, fmt)
                if parsed_date.date() < today.date():
                    return None, "Leave cannot be scheduled for past dates"
                return parsed_date.strftime("%Y-%m-%d"), None
            except ValueError:
                continue
        return None, "Date format not recognized. Prefer YYYY-MM-DD."
    except Exception as e:
        return None, f"Unable to parse date: {str(e)}"


def validate_leave_dates(start_date: str, end_date: str) -> Tuple[bool, Optional[str]]:
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        if end.date() < start.date():
            return False, "End date cannot be before start date."
        duration = (end - start).days + 1
        if duration > 365:
            return False, "Leave duration cannot exceed 365 days."
        return True, None
    except ValueError as e:
        return False, "Invalid date format in validation"


def normalize_leave_dates(leave: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    updated = dict(leave)
    errors = []
    for key in ["startDate", "endDate"]:
        if updated.get(key):
            parsed, err = parse_date_flexible(str(updated[key]))
            if err:
                errors.append(f"{key.replace('Date',' date')}: {err}")
            else:
                updated[key] = parsed
    if errors:
        return updated, "; ".join(errors)
    if updated.get("startDate") and updated.get("endDate"):
        ok, err = validate_leave_dates(updated["startDate"], updated["endDate"])
        if not ok:
            return updated, err
    return updated, None


