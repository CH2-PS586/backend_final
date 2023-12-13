from typing import Optional
from pydantic import BaseModel
from datetime import datetime

class FileBase(BaseModel):
    filename: str
    file_size: int
    category: str
    label: Optional[str] = None
    gcp_bucket_url: str
    owner_id: int

class FileCreate(FileBase):
    pass

class FileUpdate(FileBase):
    pass

class FileInDB(FileBase):
    id: int
    created_at: datetime
    updated_at: datetime
    gcp_bucket_url: str
    owner_id: int

class FileOut(FileInDB):
    owner_id: Optional[int]

    class Config:
        orm_mode = True
        
class CreateUserRequest(BaseModel):
    username: str
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
	username : str or None=None