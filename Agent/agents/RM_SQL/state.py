from typing import TypedDict, Optional


class RMState(TypedDict):
    query: str
    response: str  # Simple text response from agent
    error: Optional[str]