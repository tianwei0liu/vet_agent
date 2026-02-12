from enum import Enum
from typing import List, Dict, ClassVar
from types import MappingProxyType
from pydantic import BaseModel, Field, model_validator
from common.species_enum import SpeciesEnum

class PetRecord(BaseModel):
    """
    PetRecord Data Contract
    """
    id: int
    text: str = Field(..., description="Original user observation")
    condition: str = Field(..., description="Original pet condition")
    
    # Metadata fields
    species: SpeciesEnum
    specific_breed: str
    symptom_keywords: List[str]

    # --- 封装优化 ---
    # 1. 使用 ClassVar 标注这是一个类属性，而不是 Pydantic 的字段
    # 2. 命名以 _ 开头，表示这是“私有”的实现细节，外部不应访问
    # 3. 使用 MappingProxyType 包裹字典，使其在运行时成为【只读】，防止被意外修改
    _NORMALIZATION_MAP: ClassVar[Dict[str, SpeciesEnum]] = MappingProxyType({
        # --- Bird ---
        "bird": SpeciesEnum.BIRD, "parrot": SpeciesEnum.BIRD, 
        "cockatiel": SpeciesEnum.BIRD, "budgie": SpeciesEnum.BIRD, 
        "finch": SpeciesEnum.BIRD, "canary": SpeciesEnum.BIRD, 
        "avian": SpeciesEnum.BIRD,
        
        # --- Cat ---
        "cat": SpeciesEnum.CAT, "kitten": SpeciesEnum.CAT,
        
        # --- Dog ---
        "dog": SpeciesEnum.DOG, "puppy": SpeciesEnum.DOG,

        # --- Other mammals ---
        "ferret": SpeciesEnum.FERRET, 
        "guinea pig": SpeciesEnum.GUINEA_PIG, "guinea_pig": SpeciesEnum.GUINEA_PIG,
        "hamster": SpeciesEnum.HAMSTER,
        "rabbit": SpeciesEnum.RABBIT, "bunny": SpeciesEnum.RABBIT,

        # --- Fallback ---
        "unknown": SpeciesEnum.UNKNOWN, "pet": SpeciesEnum.UNKNOWN,
        "other": SpeciesEnum.UNKNOWN, "none": SpeciesEnum.UNKNOWN,
        "n/a": SpeciesEnum.UNKNOWN
    })

    @model_validator(mode='before')
    @classmethod
    def robust_cleaning(cls, data: dict):
        """
        Cleaning logic encapsulated within the class.
        """
        # --- 1. Species Normalization ---
        raw_species = str(data.get("species", "unknown")).lower().strip()
        
        # 【关键修改】通过 cls._NORMALIZATION_MAP 访问，逻辑高度内聚
        # 即使 key 不存在，也不会报错，而是返回 UNKNOWN
        normalized_species = cls._NORMALIZATION_MAP.get(raw_species, SpeciesEnum.UNKNOWN)
        data["species"] = normalized_species

        # --- 2. Specific Breed Logic ---
        raw_breed = str(data.get("specific_breed", "")).lower().strip()
        is_breed_invalid = not raw_breed or raw_breed in ["unknown", "generic", "none", "n/a", "pet"]
        
        if is_breed_invalid:
            # Fallback to the normalized species value (e.g., "cat")
            data["specific_breed"] = normalized_species.value
        else:
            data["specific_breed"] = raw_breed

        # --- 3. Symptoms Cleaning ---
        raw_symptoms = data.get("symptom_keywords", [])
        if raw_symptoms:
            data["symptom_keywords"] = sorted(list(set(
                [str(s).lower().strip() for s in raw_symptoms if s]
            )))
        else:
            data["symptom_keywords"] = []

        if "condition" not in data:
            data["condition"] = "Unknown"

        return data

    @property
    def dense_search_content(self) -> str:
        symptoms_str = ", ".join(self.symptom_keywords)
        return (
            f"category: {self.species.value}. "
            f"specific breed: {self.specific_breed}. "
            f"symptoms: {symptoms_str}. "
            f"observation: {self.text}"
        )

    @property
    def sparse_search_content(self) -> str:
        symptoms_str = " ".join(self.symptom_keywords)
        return f"{self.species.value} {self.specific_breed} {symptoms_str} {self.text}"
    
    @property
    def payload(self) -> dict:
        return {
            "id": self.id,
            "species": self.species.value,
            "specific_breed": self.specific_breed,
            "symptom_keywords": self.symptom_keywords,
            "text": self.text,
            "condition": self.condition
        }
