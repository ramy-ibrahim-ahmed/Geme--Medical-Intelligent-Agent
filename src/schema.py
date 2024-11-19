from pydantic import BaseModel, Field


class Query(BaseModel):
    query: str = Field(..., max_length=1024)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What are Allergies?",
                }
            ]
        }
    }
