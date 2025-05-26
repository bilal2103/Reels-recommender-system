from pydantic import BaseModel
from typing import List

class SetInitialPreferencesRequest(BaseModel):
    userId: str
    initialPreferences: List[str]
