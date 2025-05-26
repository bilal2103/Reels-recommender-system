from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Optional

class User(BaseModel):
    id: Optional[str] = None
    email: str
    password: str
    interactions: Dict[str, int]
    initialPreferences: List[str]

    model_config = ConfigDict(arbitrary_types_allowed=True)

