# 文件路径: state/pet_profile.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Any
# 确保这里导入路径是你移动后的位置
from common.species_enum import SpeciesEnum 

class PetProfile(BaseModel):
    # mandatory    
    name: Optional[str] = Field(None, description="The name of the pet")
    species: Optional[SpeciesEnum] = Field(None, description="The biological species of the pet (e.g., dog, cat)")
    breed: Optional[str] = Field(None, description="The breed of the pet (e.g., Golden Retriever, Cockatiel)")
    symptoms: List[str] = Field(
        default_factory=list, 
        description="List of symptom descriptions from the user. Return empty list if no specific symptoms are mentioned."
    )
    # optional
    age: Optional[str] = Field(None, description="Age of the pet")
    sex: Optional[str] = Field(None, description="Sex/Gender of the pet (male or female)")
    weight: Optional[str] = Field(None, description="Weight of the pet")
    language: Optional[str] = Field(None, description="The primary language used by the user in the conversation (e.g., 'Chinese', 'English', 'Spanish').")

    @field_validator('symptoms', mode='before')
    @classmethod
    def sanitize_symptoms(cls, v: Any) -> List[str]:
        """
        (逻辑保持不变)
        """
        if v is None:
            return []
        
        if isinstance(v, str):
            if v.lower() in ["none", "null", "n/a", "no symptoms", ""]:
                return []
            return [s.strip() for s in v.split(',') if s.strip()]

        if isinstance(v, list):
            cleaned_list = []
            for item in v:
                if isinstance(item, list):
                    for sub_item in item:
                        if isinstance(sub_item, str) and sub_item.strip():
                            cleaned_list.append(sub_item.strip())
                elif isinstance(item, str):
                    s = item.strip()
                    if s and s.lower() not in ["none", "n/a"]:
                        cleaned_list.append(s)
            return cleaned_list
            
        return []

    @field_validator('name', 'species', 'breed', 'age', 'sex', 'weight', 'language', mode='before')
    @classmethod
    def empty_string_to_none(cls, v):
        """
        (逻辑保持不变)
        """
        if v is None:
            return None
        if isinstance(v, str):
            cleaned = v.strip().lower()
            if cleaned in ["", "none", "null", "n/a", "unknown", "not provided"]:
                return None
        return v
    
    @property
    def summarization(self) -> str:
        # 1. Handle the list safely
        symptoms_str = ". ".join(self.symptoms) if self.symptoms else "None reported"
        
        # 2. Handle the Enum safely (get the .value, otherwise handle None)
        species_str = self.species.value if self.species else "Unknown"

        # 3. Use Parentheses for cleaner multi-line f-strings
        # We also use 'or' to provide fallback text for None values
        return (
            f"📋 Pet Profile Summary:\n"
            f"-----------------------\n"
            f"Name:     {self.name or 'Unknown'}\n"
            f"Species:  {species_str}\n"
            f"Breed:    {self.breed or 'Unknown'}\n"
            f"Age:      {self.age or 'N/A'}\n"
            f"Sex:      {self.sex or 'N/A'}\n"
            f"Weight:   {self.weight or 'N/A'}\n"
            f"Symptoms: {symptoms_str}\n"
            f"Language: {self.language}\n"
        )
