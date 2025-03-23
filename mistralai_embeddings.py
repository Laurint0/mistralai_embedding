from typing import List, Optional, Type
from cat.mad_hatter.decorators import hook
from pydantic import ConfigDict, SecretStr
from cat.factory.embedder import EmbedderSettings
from mistralai.client import Mistral  # Usa il client ufficiale di MistralAI

class CustomMistralAIEmbeddings:
    """Classe per ottenere embeddings da Mistral AI"""

    def __init__(self, mistral_api_key: str):
        self.client = Mistral(api_key=mistral_api_key)
        self.model = "mistral-embed"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings per una lista di documenti"""
        response = self.client.embeddings.create(model=self.model, inputs=texts)
        return [data.embedding for data in response.data]

    def embed_query(self, text: str) -> List[float]:
        """Genera un embedding per una singola query"""
        response = self.client.embeddings.create(model=self.model, inputs=[text])
        return response.data[0].embedding

class MistralAIEmbedderConfig(EmbedderSettings):
    """Configurazione per l'embedder MistralAI."""

    mistral_api_key: Optional[SecretStr]  # Chiave API per MistralAI
    _pyclass: Type = CustomMistralAIEmbeddings  # Usa la classe personalizzata!

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "MistralAI Embedder",
            "description": "Configuration for MistralAI Embeddings",
            "link": "https://www.mistral.ai/capabilities/embeddings/",
        }
    )

@hook
def factory_allowed_embedders(allowed, cat) -> List:
    """Hook per aggiungere l'embedder MistralAI alla lista degli embedders consentiti."""
    allowed.append(MistralAIEmbedderConfig)
    return allowed
#fatto da laurint :D
