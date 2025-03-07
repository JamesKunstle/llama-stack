from typing import Optional

from pydantic import BaseModel


class ilabPostTrainingConfig(BaseModel):
    torch_seed: Optional[int] = None