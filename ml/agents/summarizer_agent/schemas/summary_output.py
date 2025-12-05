from pydantic import BaseModel, Field

class SummaryOutput(BaseModel):
    summary: str = Field(
        description="Concise 100-200 word summary of the scientific text. "
                    "Must preserve key findings, methods, and conclusions. "
                    "Written in third person, past tense. "
                    "No preamble or meta-commentary."
    )
    key_concepts: list[str] = Field(
        default_factory=list,
        description="List of 3-5 key technical concepts or terms from the text"
    )