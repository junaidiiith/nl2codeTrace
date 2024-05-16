import json
from typing import Union
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anyscale import Anyscale
from llama_index.llms.cohere import Cohere
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
):
    all_api_keys = json.load(open(api_keys_file))
    api_key = all_api_keys[llm_type][0]
    return api_key


def get_embed_model(
    model_name: str = EmbeddingModelsMap[EmbeddingModels.DEFAULT_EMBED_MODEL.value],
):

    embed_llm = HuggingFaceEmbedding(
        model_name=model_name,
        max_length=1024,
    )
    return embed_llm


def get_llm(
    model_type: str = LLMTypes.ANY_SCALE.value,
    model_name: str = LLMsMap[LLMs.LLAMA3.value],
):
    api_key = get_api_keys(llm_type=model_type)
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

    return llm


def set_llm_and_embed(
    llm_type: str = None,
    llm_name: str = None,
    embed_model_name: str = None,
):
    
    llm = get_llm(llm_type, llm_name) if llm_type and llm_name else get_llm()
    embed_llm = get_embed_model(embed_model_name) if embed_model_name else get_embed_model()
    Settings.llm = llm
    Settings.embed_model = embed_llm