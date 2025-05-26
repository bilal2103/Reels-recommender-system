from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Union, Any

class Reel(BaseModel):
    id: Optional[str] = None
    path: str
    category: str
    textualEmbeddings: Optional[Union[List[float], List[List[float]]]] = None
    aggregatedEmbeddings: Optional[Union[List[float], List[List[float]]]] = None
    videoEmbeddings: Optional[List[List[float]]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
