import json
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MainAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.1,  # Lower temperature for consistency
        )
        
        # Store conversation state for each user
        self.user_states: Dict[str, Dict[str, Any]] = {}
        
        # Leave application fields
        self.required_fields = ["startDate", "endDate", "leaveType", "reason"]
        
        # Enhanced date parsing patterns
        self.weekdays = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        # Comprehensive date formats
        self.date_formats = [
            "%Y-%m-%d",           # 2024-12-25 (ISO - most reliable)
            "%B %d, %Y",          # December 25, 2024
            "%d %B %Y",           # 25 December 2024
            "%b %d, %Y",          # Dec 25, 2024
            "%d %b %Y",           # 25 Dec 2024
            "%d/%m/%Y",           # 25/12/2024
            "%m/%d/%Y",           # 12/25/2024
            "%d-%m-%Y",           # 25-12-2024
            "%m-%d-%Y",           # 12-25-2024
            "%d/%m/%y",           # 25/12/24
            "%m/%d/%y",           # 12/25/24
            "%d-%m-%y",           # 25-12-24
            "%m-%d-%y",           # 12-25-24
        ]
    
    def get_or_create_user_state(self, user_email: str) -> Dict[str, Any]:
        """Get or create user state"""
        if user_email not in self.user_states:
            self.user_states[user_email] = {
                "currentIntent": "general",
                "leaveData": {},
                "conversationHistory": [],
                "isCollectingLeaveData": False,
                "lastMessage": None,
                "invalidAttempts": {},  # Track invalid format attempts per field
                "lastFieldAsked": None,  # Track last field we asked about
            }
            logger.info(f"Created new user state for {user_email}")
        return self.user_states[user_email]

    def clean_json_response(self, content: str) -> str:
        """Clean LLM response to extract pure JSON"""
        content = content.strip()
        # Remove markdown code blocks
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        # Find the first { and last } to extract JSON object
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            content = content[start_idx:end_idx + 1]
        return content

    def parse_date_flexible(self, date_input: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Enhanced date parsing with comprehensive format support
        Returns (normalized_date, error_message)
        """
        if not date_input or not date_input.strip():
            return None, "Date cannot be empty"
        date_input = date_input.strip().lower()
        today = datetime.now()
        try:
            # Handle relative dates first
            relative_dates = {
                "today": 0,
                "tomorrow": 1,
                "day after tomorrow": 2,
                "yesterday": -1
            }
            if date_input in relative_dates:
                days_offset = relative_dates[date_input]
                if days_offset < 0:
                    return None, "Leave cannot be scheduled for past dates"
                target_date = today + timedelta(days=days_offset)
                return target_date.strftime("%Y-%m-%d"), None
            # Handle "next [weekday]" patterns
            next_weekday_match = re.search(r'next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', date_input)
            if next_weekday_match:
                target_weekday = self.weekdays[next_weekday_match.group(1)]
                days_ahead = target_weekday - today.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                target_date = today + timedelta(days=days_ahead)
                return target_date.strftime("%Y-%m-%d"), None
            # Handle "this [weekday]" patterns
            this_weekday_match = re.search(r'this\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', date_input)
            if this_weekday_match:
                target_weekday = self.weekdays[this_weekday_match.group(1)]
                days_ahead = target_weekday - today.weekday()
                if days_ahead < 0:
                    return None, f"This {this_weekday_match.group(1)} has already passed. Did you mean next {this_weekday_match.group(1)}?"
                target_date = today + timedelta(days=days_ahead)
                return target_date.strftime("%Y-%m-%d"), None
            # Handle "in X days/weeks" patterns
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
            # Handle ordinal numbers (1st, 2nd, 3rd, etc.)
            ordinal_cleaned = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_input)
            # Try to parse with different formats
            for fmt in self.date_formats:
                try:
                    parsed_date = datetime.strptime(ordinal_cleaned, fmt)
                    if parsed_date.date() < today.date():
                        return None, "Leave cannot be scheduled for past dates"
                    return parsed_date.strftime("%Y-%m-%d"), None
                except ValueError:
                    continue
            return None, self._get_date_format_error_message()
        except Exception as e:
            logger.error(f"Date parsing error for '{date_input}': {str(e)}")
            return None, f"Unable to parse date: {str(e)}"

    def _get_date_format_error_message(self) -> str:
        """Get a helpful error message with date format examples"""
        return """❌ **Date format not recognized. Please use one of these formats:**

📅 **Natural language:**
• today, tomorrow
• next Monday, this Friday  
• in 3 days, in 2 weeks

📅 **Specific dates:**
• 2024-12-25 (recommended format)
• December 25, 2024 or Dec 25, 2024
• 25/12/2024 or 12/25/2024
• December 25th, 2024

**💡 Tip:** Use "YYYY-MM-DD" format for best results (e.g., 2024-12-25)"""

    def validate_leave_dates(self, start_date: str, end_date: str) -> Tuple[bool, Optional[str]]:
        """Validate that end date is after start date"""
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            if end.date() < start.date():
                return False, "End date cannot be before start date. Please check your dates."
            duration = (end - start).days + 1
            if duration > 365:
                return False, "Leave duration cannot exceed 365 days. Please check your dates."
            return True, None
        except ValueError as e:
            logger.error(f"Date range validation error: {str(e)}")
            return False, "Invalid date format in validation"

    def extract_via_rules(self, user_message: str, current_leave_data: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based extraction for reliable pattern matching"""
        message = user_message.lower()
        extracted = {}
        
        # Extract leave types using keywords
        leave_type_patterns = {
            'sick': ['sick', 'ill', 'medical', 'health', 'doctor', 'hospital'],
            'vacation': ['vacation', 'holiday', 'trip', 'travel', 'tour'],
            'personal': ['personal', 'family', 'private', 'urgent'],
            'casual': ['casual', 'day off', 'rest'],
            'emergency': ['emergency', 'urgent', 'crisis'],
            'maternity': ['maternity', 'pregnancy', 'birth'],
            'paternity': ['paternity', 'father', 'newborn']
        }
        
        # Only extract leave type if not already set
        if not current_leave_data.get("leaveType"):
            for leave_type, keywords in leave_type_patterns.items():
                if any(keyword in message for keyword in keywords):
                    extracted['leaveType'] = leave_type
                    logger.info(f"Extracted leave type via rules: {leave_type}")
                    break
        
        # Extract date ranges using regex patterns
        if not current_leave_data.get("startDate") or not current_leave_data.get("endDate"):
            date_range_patterns = [
                r'from\s+([^to]+?)\s+to\s+([^\s,\.!?]+)',
                r'between\s+([^and]+?)\s+and\s+([^\s,\.!?]+)',
                r'starting\s+([^until]+?)\s+until\s+([^\s,\.!?]+)',
                r'([a-zA-Z0-9\s]+?)\s+to\s+([a-zA-Z0-9\s]+?)(?:\s|$|[,.!?])',
            ]
            
            for pattern in date_range_patterns:
                match = re.search(pattern, message)
                if match:
                    start_text = match.group(1).strip()
                    end_text = match.group(2).strip()
                    
                    # Basic validation - ensure reasonable length
                    if 3 <= len(start_text) <= 50 and 3 <= len(end_text) <= 50:
                        if not current_leave_data.get("startDate"):
                            extracted['startDate'] = start_text
                        if not current_leave_data.get("endDate"):
                            extracted['endDate'] = end_text
                        logger.info(f"Extracted date range via rules: {start_text} to {end_text}")
                        break
        
        # Extract single dates if no range found
        if not current_leave_data.get("startDate") and not extracted.get("startDate"):
            single_date_patterns = [
                r'tomorrow',
                r'today',
                r'next\s+\w+',
                r'this\s+\w+',
                r'in\s+\d+\s+days?',
                r'\d{4}-\d{2}-\d{2}',
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
                r'[a-zA-Z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}'
            ]
            
            for pattern in single_date_patterns:
                match = re.search(pattern, message)
                if match:
                    extracted['startDate'] = match.group(0).strip()
                    logger.info(f"Extracted single date via rules: {match.group(0)}")
                    break
        
        return extracted if extracted else {}
    
    async def classify_intent(self, user_message: str, user_email: str) -> str:
        """Enhanced intent classification with rule-based pre-filtering"""
        
        # Rule-based classification first (faster and more reliable)
        leave_keywords = [
            'leave', 'vacation', 'time off', 'sick', 'holiday', 'absence',
            'pto', 'day off', 'days off', 'break', 'medical leave',
            'personal leave', 'emergency leave', 'maternity', 'paternity'
        ]
        
        message_lower = user_message.lower()
        if any(keyword in message_lower for keyword in leave_keywords):
            logger.info("Intent classified as leave_application via keywords")
            return "leave_application"
        
        # LLM-based classification with improved prompt
        system_prompt = """You are an intent classifier. Classify this message as either:

1. LEAVE - if about taking time off, vacation, sick days, or any work absence
2. GENERAL - for everything else

Respond with only "LEAVE" or "GENERAL" - no other text."""
        
        try:
            for attempt in range(2):  # Max 2 attempts
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message),
                ]
                response = await self.llm.ainvoke(messages)
                
                # Clean and validate response
                intent = response.content.strip().upper()
                intent = intent.replace('"', '').replace("'", "").replace(".", "").replace(",", "")
                
                if "LEAVE" in intent:
                    logger.info(f"Intent classified as leave_application via LLM (attempt {attempt + 1})")
                    return "leave_application"
                elif "GENERAL" in intent:
                    logger.info(f"Intent classified as general via LLM (attempt {attempt + 1})")
                    return "general"
                
                logger.warning(f"Unclear LLM intent response: '{intent}' (attempt {attempt + 1})")
                
        except Exception as error:
            logger.error(f"Error classifying intent: {error}")
        
        # Default to general for safety
        logger.info("Intent classification defaulted to general")
        return "general"
    
    def is_leave_data_complete(self, leave_data: Dict[str, Any]) -> bool:
        """Check if all required leave data is collected and validated"""
        if not all(leave_data.get(field) and str(leave_data[field]).strip() != "" for field in self.required_fields):
            return False
        
        # Additional validation for dates
        start_date = leave_data.get("startDate")
        end_date = leave_data.get("endDate")
        
        if start_date and end_date:
            # Try to parse and validate the dates
            start_parsed, start_error = self.parse_date_flexible(start_date)
            end_parsed, end_error = self.parse_date_flexible(end_date)
            
            if start_error or end_error:
                return False
            
            # Update with normalized dates
            leave_data["startDate"] = start_parsed
            leave_data["endDate"] = end_parsed
            
            # Validate date range
            valid, _ = self.validate_leave_dates(start_parsed, end_parsed)
            return valid
        
        return True
    
    async def extract_leave_info(self, user_message: str, current_leave_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced leave information extraction with hybrid approach"""
        
        # First try rule-based extraction (faster and more reliable)
        rule_based_data = self.extract_via_rules(user_message, current_leave_data)
        if rule_based_data:
            logger.info(f"Extracted data via rules: {rule_based_data}")
            return rule_based_data
        
        # Fall back to LLM-based extraction
        system_prompt = f"""Extract leave information from the user message. Return a JSON object with only the fields you can identify.

Available fields:
- startDate: extract exactly as written
- endDate: extract exactly as written  
- leaveType: casual, sick, vacation, personal, emergency, maternity, paternity, etc.
- reason: brief description

Current data: {json.dumps(current_leave_data)}

CRITICAL RULES:
1. Return ONLY raw JSON - no markdown, no code blocks, no extra text
2. Start response with {{ and end with }}
3. Only extract NEW information from current message
4. Don't overwrite existing data unless clearly correcting
5. Preserve date text exactly as written
6. Return {{}} if no new information found

Examples:
"vacation from Dec 25 to Dec 30" → {{"startDate": "Dec 25", "endDate": "Dec 30", "leaveType": "vacation"}}
"sick leave tomorrow" → {{"startDate": "tomorrow", "leaveType": "sick"}}
"december 15th" → {{"startDate": "december 15th"}}
"personal reasons" → {{"reason": "personal reasons"}}"""
        
        try:
            for attempt in range(2):  # Max 2 attempts
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message),
                ]
                
                response = await self.llm.ainvoke(messages)
                
                # Clean the response to extract pure JSON
                content = self.clean_json_response(response.content)
                
                # Validate JSON format
                if not content or not content.startswith("{") or not content.endswith("}"):
                    logger.warning(f"LLM did not return valid JSON format (attempt {attempt + 1}): {response.content[:100]}")
                    continue
                
                # Parse JSON
                extracted_data = json.loads(content)
                
                # Ensure it's a dictionary
                if not isinstance(extracted_data, dict):
                    logger.warning(f"LLM returned non-dict JSON (attempt {attempt + 1}): {content}")
                    continue
                
                logger.info(f"Extracted data via LLM (attempt {attempt + 1}): {extracted_data}")
                return extracted_data
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}, Content: '{content}'")
        except Exception as error:
            logger.error(f"Error extracting leave info: {error}")
        
        logger.info("No data extracted, returning empty dict")
        return {}
    
    async def generate_response(self, user_message: str, user_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate conversational response with enhanced validation"""
        leave_data = user_state["leaveData"]
        is_collecting_leave_data = user_state["isCollectingLeaveData"]
        invalid_attempts = user_state.get("invalidAttempts", {})
        
        if is_collecting_leave_data:
            # Extract new information using hybrid approach
            extracted_data = await self.extract_leave_info(user_message, leave_data)
            
            # Validate and normalize dates if they were extracted
            validated_data = {}
            validation_errors = []
            
            for field, value in extracted_data.items():
                if field in ["startDate", "endDate"] and value:
                    parsed_date, error = self.parse_date_flexible(value)
                    if error:
                        validation_errors.append(f"**{field.replace('Date', ' Date')}:** {error}")
                        # Track invalid attempts
                        invalid_attempts[field] = invalid_attempts.get(field, 0) + 1
                    else:
                        validated_data[field] = parsed_date
                        # Reset invalid attempts on success
                        invalid_attempts.pop(field, None)
                else:
                    validated_data[field] = value
            
            # If there are validation errors, ask for correction
            if validation_errors:
                error_msg = "\n".join(validation_errors)
                return {
                    "response": error_msg,
                    "shouldInvokeLeaveAgent": False,
                    "leaveData": leave_data.copy(),
                }
            
            # Update leave data with validated information
            leave_data.update(validated_data)
            
            # If both dates are present, validate the range
            if leave_data.get("startDate") and leave_data.get("endDate"):
                valid_range, range_error = self.validate_leave_dates(
                    leave_data["startDate"], leave_data["endDate"]
                )
                if not valid_range:
                    return {
                        "response": f"❌ **Date Range Issue:**\n\n{range_error}\n\nPlease provide valid start and end dates.",
                        "shouldInvokeLeaveAgent": False,
                        "leaveData": leave_data.copy(),
                    }
            
            # Check if all data is complete
            if self.is_leave_data_complete(leave_data):
                user_state["isCollectingLeaveData"] = False
                user_state["invalidAttempts"] = {}  # Reset invalid attempts
                
                # Format dates nicely for display
                try:
                    start_display = datetime.strptime(leave_data['startDate'], "%Y-%m-%d").strftime("%B %d, %Y")
                    end_display = datetime.strptime(leave_data['endDate'], "%Y-%m-%d").strftime("%B %d, %Y")
                except:
                    start_display = leave_data['startDate']
                    end_display = leave_data['endDate']
                
                return {
                    "response": f"""✅ **Perfect! I have all the information needed for your leave application:**
                    
📅 **Leave Details:**
• **Start Date:** {start_display}
• **End Date:** {end_display}
• **Leave Type:** {leave_data['leaveType'].title()}
• **Reason:** {leave_data['reason']}

🔄 **Processing your leave application now...**""",
                    "shouldInvokeLeaveAgent": True,
                    "leaveData": leave_data.copy(),
                }
            else:
                # Ask for missing information with enhanced prompts
                missing_fields = [
                    field for field in self.required_fields
                    if not leave_data.get(field) or str(leave_data[field]).strip() == ""
                ]
                
                field_prompts = {
                    "startDate": """📅 **When would you like your leave to start?**
                    
You can say:
• Specific dates: "2024-12-23" or "December 23, 2024"
• Relative dates: "tomorrow", "next Monday", "in 3 days"
• Ranges: "from December 23 to December 27" """,
                    
                    "endDate": """📅 **When should your leave end?**
                    
You can say:
• Specific dates: "2024-12-27" or "December 27, 2024"  
• Relative dates: "next Friday", "in 5 days"
• Or mention both dates together: "from start date to end date" """,
                    
                    "leaveType": """📋 **What type of leave is this?**
                    
Common types:
• Casual leave
• Sick leave
• Vacation / Annual leave
• Personal leave
• Emergency leave
• Maternity/Paternity leave""",
                    
                    "reason": "📝 **What's the reason for your leave?** (A brief description is fine)"
                }
                
                next_field = missing_fields[0]
                prompt = field_prompts[next_field]
                
                # Add encouragement if we got some data
                if validated_data:
                    updated_fields = [f.replace('Date', ' date') for f in validated_data.keys()]
                    prompt = f"✅ Great! I got your {', '.join(updated_fields)}.\n\n{prompt}"
                
                # Add additional help if user has failed multiple times
                if invalid_attempts.get(next_field, 0) >= 2:
                    prompt += "\n\n💡 **Having trouble?** You can also say 'help with dates' for more examples."
                
                user_state["lastFieldAsked"] = next_field
                return {
                    "response": prompt,
                    "shouldInvokeLeaveAgent": False,
                    "leaveData": leave_data.copy(),
                }
        else:
            # General conversation
            system_prompt = """You are a helpful workplace assistant. Be friendly, professional, and concise. 
            If the user mentions anything related to leave, vacation, time off, or sick days, gently guide them to provide more details about their leave request.
            If they ask for help with date formats, provide examples of acceptable date formats."""
            
            try:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message),
                ]
                response = await self.llm.ainvoke(messages)
                
                # If user asks for date help, provide specific guidance
                if "date" in user_message.lower() and ("help" in user_message.lower() or "format" in user_message.lower()):
                    return {
                        "response": self._get_date_format_error_message(),
                        "shouldInvokeLeaveAgent": False,
                        "leaveData": None,
                    }
                
                return {
                    "response": response.content,
                    "shouldInvokeLeaveAgent": False,
                    "leaveData": None,
                }
            except Exception as error:
                logger.error(f"Error generating response: {error}")
                return {
                    "response": "I'm here to help! How can I assist you today?",
                    "shouldInvokeLeaveAgent": False,
                    "leaveData": None,
                }
    
    async def process_message(self, user_message: str, user_email: str) -> Dict[str, Any]:
        """Main processing method with enhanced error handling"""
        try:
            user_state = self.get_or_create_user_state(user_email)
            
            logger.info(f"Processing message from {user_email}: '{user_message}' | Collecting: {user_state['isCollectingLeaveData']}")
            
            # Handle special commands
            if user_message.lower().strip() in ["reset", "start over", "cancel"]:
                user_state["isCollectingLeaveData"] = False
                user_state["leaveData"] = {}
                user_state["invalidAttempts"] = {}
                user_state["lastFieldAsked"] = None
                logger.info(f"User state reset for {user_email}")
                return {
                    "response": "🔄 **Reset complete!** How can I help you today?",
                    "shouldInvokeLeaveAgent": False,
                    "leaveData": None,
                }
            
            # Classify intent (but if already collecting leave data, treat as leave-related)
            if user_state["isCollectingLeaveData"]:
                intent = "leave_application"
            else:
                intent = await self.classify_intent(user_message, user_email)
            
            # Update user state based on intent
            if intent == "leave_application" and not user_state["isCollectingLeaveData"]:
                user_state["isCollectingLeaveData"] = True
                user_state["leaveData"] = {}
                user_state["invalidAttempts"] = {}
                
                # Extract initial leave information
                extracted_data = await self.extract_leave_info(user_message, {})
                
                # Validate extracted dates
                validated_data = {}
                validation_errors = []
                
                for field, value in extracted_data.items():
                    if field in ["startDate", "endDate"] and value:
                        parsed_date, error = self.parse_date_flexible(value)
                        if error:
                            validation_errors.append(f"**{field.replace('Date', ' Date')}:** {error}")
                        else:
                            validated_data[field] = parsed_date
                    else:
                        validated_data[field] = value
                
                user_state["leaveData"].update(validated_data)
                
                # If there are validation errors, ask for correction
                if validation_errors:
                    error_msg = "\n".join(validation_errors)
                    return {
                        "response": f"👋 I'd be happy to help with your leave application!\n\n{error_msg}",
                        "shouldInvokeLeaveAgent": False,
                        "leaveData": user_state["leaveData"].copy(),
                    }
                
                # Check if we have some data and what's missing
                if user_state["leaveData"]:
                    missing_fields = [
                        field for field in self.required_fields
                        if not user_state["leaveData"].get(field) or str(user_state["leaveData"][field]).strip() == ""
                    ]
                    
                    if not missing_fields:
                        # We have everything, check date range
                        if user_state["leaveData"].get("startDate") and user_state["leaveData"].get("endDate"):
                            valid_range, range_error = self.validate_leave_dates(
                                user_state["leaveData"]["startDate"], 
                                user_state["leaveData"]["endDate"]
                            )
                            if not valid_range:
                                return {
                                    "response": f"❌ **Date Range Issue:**\n\n{range_error}",
                                    "shouldInvokeLeaveAgent": False,
                                    "leaveData": user_state["leaveData"].copy(),
                                }
                    
                    if missing_fields:
                        field_prompts = {
                            "startDate": "📅 When would you like your leave to start?",
                            "endDate": "📅 When should your leave end?",
                            "leaveType": "📋 What type of leave is this? (casual, sick, vacation, etc.)",
                            "reason": "📝 What's the reason for your leave?",
                        }
                        
                        next_field = missing_fields[0]
                        user_state["lastFieldAsked"] = next_field
                        return {
                            "response": f"Great! I got some information. {field_prompts[next_field]}",
                            "shouldInvokeLeaveAgent": False,
                            "leaveData": user_state["leaveData"].copy(),
                        }
                else:
                    return {
                        "response": """👋 **I'd be happy to help you apply for leave!**
                        
📅 To get started, please tell me:
• When you'd like to start your leave
• When it should end
• What type of leave it is
• The reason for your leave

**Example:** "I want to take vacation from December 23 to December 27 for family time" """,
                        "shouldInvokeLeaveAgent": False,
                        "leaveData": {},
                    }
            
            # Generate response
            result = await self.generate_response(user_message, user_state)
            
            # Update conversation history
            user_state["conversationHistory"].append({
                "user": user_message,
                "assistant": result["response"],
                "timestamp": datetime.now().isoformat(),
            })
            
            # Keep only last 10 messages to prevent memory issues
            if len(user_state["conversationHistory"]) > 10:
                user_state["conversationHistory"] = user_state["conversationHistory"][-10:]
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing message for {user_email}: {str(e)}")
            return {
                "response": "I apologize, but I encountered an error. Please try again or say 'reset' to start over.",
                "shouldInvokeLeaveAgent": False,
                "leaveData": None,
            }
    
    def reset_user_state(self, user_email: str) -> None:
        if user_email in self.user_states:
            del self.user_states[user_email]
            logger.info(f"Reset user state for {user_email}")

    def get_user_analytics(self, user_email: str) -> Dict[str, Any]:
        if user_email not in self.user_states:
            return {"error": "User not found"}
        user_state = self.user_states[user_email]
        leave_data = user_state.get("leaveData", {})
        return {
            "total_messages": len(user_state.get("conversationHistory", [])),
            "is_collecting_leave_data": user_state.get("isCollectingLeaveData", False),
            "current_leave_data_completeness": sum(1 for field in self.required_fields 
                                                  if leave_data.get(field)),
            "invalid_attempts": user_state.get("invalidAttempts", {}),
            "last_field_asked": user_state.get("lastFieldAsked"),
            "current_intent": user_state.get("currentIntent", "general"),
        }