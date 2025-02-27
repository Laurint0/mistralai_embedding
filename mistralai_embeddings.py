from typing import List, Optional, Type
from cat.mad_hatter.decorators import hook
from pydantic import ConfigDict, SecretStr
from cat.factory.embedder import EmbedderSettings
from langchain_mistralai.embeddings import MistralAIEmbeddings

class MistralAIEmbedderConfig(EmbedderSettings):
    """Configurazione per l'embedder MistralAI."""

    mistral_api_key: Optional[SecretStr]  # Chiave API per MistralAI
    model: str = "mistral-embed"          # Modello predefinito per embeddings
    max_retries: int = 5                  # Numero massimo di tentativi in caso di errore
    timeout: int = 120                    # Timeout in secondi

    _pyclass: Type = MistralAIEmbeddings  # Classe Python associata

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "MistralAI Embedder",
            "description": "Configuration for MistralAI Embeddings",
            "link": "https://www.mistral.ai",
        }
    )

@hook
def factory_allowed_embedders(allowed, cat) -> List:
    """
    Hook per aggiungere l'embedder MistralAI alla lista degli embedders consentiti.
    
    :param allowed: Lista degli embedders attualmente consentiti.
    :param cat: Istanza del Cheshire Cat.
    :return: Lista aggiornata degli embedders consentiti.
    """
    allowed.append(MistralAIEmbedderConfig)
    return allowed
