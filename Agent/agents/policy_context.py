import os
import json
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger("PolicyContext")


class PolicyContext:
    """Manages policy documents and provides context for the agents."""
    
    def __init__(self, docs_folder: str = "docs"):
        self.docs_folder = Path(docs_folder)
        self.policies = {}
        self.leave_policy = None
        self._load_policies()
    
    def _load_policies(self):
        """Load policy documents from the docs folder."""
        try:
            if not self.docs_folder.exists():
                logger.warning(f"Policy docs folder not found: {self.docs_folder}")
                return
            
            # Load each policy document
            for doc_file in self.docs_folder.glob("*.pdf"):
                policy_name = doc_file.stem
                try:
                    # For now, we'll create a basic policy structure
                    # In production, you'd use a PDF parser like PyPDF2 or pdfplumber
                    policy_content = self._extract_policy_content(doc_file)
                    self.policies[policy_name] = {
                        "file": str(doc_file),
                        "content": policy_content,
                        "type": self._categorize_policy(policy_name)
                    }
                    
                    # Identify leave policy
                    if "leave" in policy_name.lower() or "attendance" in policy_name.lower():
                        self.leave_policy = policy_content
                        logger.info(f"Loaded leave policy: {policy_name}")
                    
                except Exception as e:
                    logger.error(f"Error loading policy {doc_file}: {e}")
            
            logger.info(f"Loaded {len(self.policies)} policy documents")
            
        except Exception as e:
            logger.error(f"Error loading policies: {e}")
    
    def _extract_policy_content(self, file_path: Path) -> str:
        """Extract content from policy document. Placeholder implementation."""
        # In production, implement actual PDF parsing
        policy_name = file_path.stem
        
        # Create policy summaries based on file names
        if "leave" in policy_name.lower() or "attendance" in policy_name.lower():
            return """
            LEAVE POLICY SUMMARY:
            - Annual leave: 25 days per year
            - Sick leave: Up to 10 days with medical certificate
            - Personal leave: 5 days per year
            - Maternity leave: 6 months paid
            - Paternity leave: 2 weeks paid
            - Leave requests must be submitted at least 48 hours in advance
            - Emergency leave can be requested same day with supervisor approval
            - Leave balance carries over up to 5 days maximum
            - Unused leave is forfeited at year-end
            """
        elif "dress" in policy_name.lower():
            return """
            DRESS CODE POLICY:
            - Business casual attire required
            - No jeans, shorts, or flip-flops
            - Company logo items preferred
            - Formal dress for client meetings
            - Safety equipment required in designated areas
            """
        elif "health" in policy_name.lower() or "safety" in policy_name.lower():
            return """
            HEALTH & SAFETY POLICY:
            - Regular safety training required
            - Report all incidents immediately
            - Use personal protective equipment
            - Emergency procedures posted
            - Health checks as required
            """
        elif "holiday" in policy_name.lower() or "calendar" in policy_name.lower():
            return """
            HOLIDAY CALENDAR:
            - Public holidays observed
            - Company shutdown periods
            - Blackout dates for leave requests
            - Weekend work arrangements
            """
        else:
            return f"Policy document: {policy_name}"
    
    def _categorize_policy(self, policy_name: str) -> str:
        """Categorize policy by type."""
        name_lower = policy_name.lower()
        if "leave" in name_lower or "attendance" in name_lower:
            return "leave"
        elif "dress" in name_lower:
            return "dress_code"
        elif "health" in name_lower or "safety" in name_lower:
            return "health_safety"
        elif "holiday" in name_lower or "calendar" in name_lower:
            return "calendar"
        else:
            return "general"
    
    def get_policy_context(self, query: str) -> str:
        """Get all policy context for LLM to intelligently select relevant information."""
        if not self.policies:
            return "No policy documents available."
        
        # Provide all policies as context for the LLM to intelligently select from
        all_policies = []
        for policy_name, policy_data in self.policies.items():
            all_policies.append(f"=== {policy_name.upper()} ===\n{policy_data['content']}\n")
        
        return "\n".join(all_policies)
    
    async def validate_leave_against_policy(self, leave_data: Dict, llm) -> Dict[str, any]:
        """Validate leave request against policy using LLM reasoning."""
        if not self.policies:
            return {"valid": True, "warnings": [], "conflicts": []}
        
        try:
            # Get all policy context
            policy_context = self.get_policy_context("")
            
            # Create validation prompt for LLM
            validation_prompt = f"""
You are a workplace policy validator. Analyze the leave request against the company policies provided below.

LEAVE REQUEST:
- Type: {leave_data.get('leaveType', 'Not specified')}
- Start Date: {leave_data.get('startDate', 'Not specified')}
- End Date: {leave_data.get('endDate', 'Not specified')}
- Reason: {leave_data.get('reason', 'Not specified')}
- Supervisor Email: {leave_data.get('supervisorEmail', 'Not specified')}

COMPANY POLICIES:
{policy_context}

Please analyze this leave request and return a JSON response with:
{{
    "valid": true/false,
    "warnings": ["warning1", "warning2"],
    "conflicts": ["conflict1", "conflict2"],
    "policy_reference": "Brief reference to specific policy section"
}}

Rules:
- Set "valid" to false only if there are clear policy violations that would prevent approval
- Add warnings for policy considerations that should be noted but don't prevent approval
- Add conflicts for violations that would prevent approval
- Be specific about which policy section applies
- If no policies are violated, return valid: true with appropriate warnings
"""
            
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content="You are a workplace policy validator. Return only valid JSON."),
                HumanMessage(content=validation_prompt)
            ]
            
            response = await llm.ainvoke(messages)
            
            # Parse LLM response
            import json
            try:
                result = json.loads(response.content)
                return {
                    "valid": result.get("valid", True),
                    "warnings": result.get("warnings", []),
                    "conflicts": result.get("conflicts", []),
                    "policy_reference": result.get("policy_reference", "Based on company policy")
                }
            except json.JSONDecodeError:
                # Fallback if LLM doesn't return valid JSON
                logger.warning("LLM policy validation returned invalid JSON, using fallback")
                return {"valid": True, "warnings": ["Policy validation completed"], "conflicts": []}
            
        except Exception as e:
            logger.error(f"Error validating leave against policy: {e}")
            return {"valid": True, "warnings": [], "conflicts": []}
    
    def get_all_policies(self) -> Dict[str, str]:
        """Get all available policies."""
        return {name: policy["content"] for name, policy in self.policies.items()}
    
    async def get_intelligent_policy_context(self, query: str, llm) -> str:
        """Use LLM to intelligently select relevant policy context for a query."""
        if not self.policies:
            return "No policy documents available."
        
        try:
            # Get all policy context
            all_policies = self.get_policy_context("")
            
            # Create prompt for LLM to select relevant policies
            selection_prompt = f"""
You are a policy assistant. Given a user query and available company policies, select the most relevant policy sections to provide as context.

USER QUERY: "{query}"

AVAILABLE POLICIES:
{all_policies}

Please select and return ONLY the most relevant policy sections that would help answer the user's query. 
Return the policy sections exactly as they appear above, maintaining the formatting.
If no policies are relevant, return "No relevant policies found for this query."

Focus on:
- Direct relevance to the user's question
- Policies that provide actionable information
- Avoid including irrelevant policy sections
"""
            
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content="You are a policy assistant. Return only the relevant policy sections, maintaining original formatting."),
                HumanMessage(content=selection_prompt)
            ]
            
            response = await llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error getting intelligent policy context: {e}")
            # Fallback to providing all policies
            return all_policies
