import json
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anyscale import Anyscale
from llama_index.llms.cohere import Cohere
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

from constants import (
    EmbeddingModels, 
    EmbeddingModelsMap,
    LLMTypes,
    LLMs,
    LLMsMap,
)


def get_api_keys(
    api_keys_file: str = 'api_keys.json',
    llm_type: str = LLMTypes.ANY_SCALE.value,
    idx = 0
):
    all_api_keys = json.load(open(api_keys_file))
    api_key = all_api_keys[llm_type][idx]
    return api_key


def set_embed_model(
    model_name: str = EmbeddingModelsMap[EmbeddingModels.DEFAULT_EMBED_MODEL.value],
):

    embed_llm = HuggingFaceEmbedding(
        model_name=model_name,
        max_length=1024,
    )
    Settings.embed_model = embed_llm


def set_llm(
    model_type: str = None,
    model_name: str = None,
    api_key = None
):
    api_key = get_api_keys(llm_type=model_type) if api_key is None else api_key
    if model_type == LLMTypes.ANY_SCALE.value:
        llm = Anyscale(
            model=model_name,
            api_key=api_key,
        )
    elif model_type == LLMTypes.COHERE.value:
        llm = Cohere(
            model=model_name,
            api_key=api_key,
        )
    elif model_type == LLMTypes.OPENAI.value:
        llm = OpenAI(
            model=model_name,
            api_key=api_key
        )

    Settings.llm = llm


def set_llm_and_embed(
    llm_type: str = None,
    llm_name: str = None,
    embed_model_name: str = None,
    api_key = None
):
    
    set_llm(llm_type, llm_name, api_key) if llm_type and llm_name else set_llm()
    set_embed_model(embed_model_name) if embed_model_name else set_embed_model()