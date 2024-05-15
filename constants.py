from enum import Enum

PIPELINE_CHUNK_SIZE = 512
PIPELINE_CHUNK_OVERLAP = 50

class AnyScaleLLMs(Enum):
    MISTRAL_8X7B = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    MISTRAL_8X22B = "mistralai/Mixtral-8x22B-Instruct-v0.1"
    MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.1"
    CODELLAMA = "codellama/CodeLlama-70b-Instruct-hf"
    LLAMA3 = "meta-llama/Meta-Llama-3-70B-Instruct"
    GOOGLE_GEMMA = "google/gemma-7b-it"
    DEFAULT_LLM = LLAMA3


class CohereLLMs(Enum):
    COMMANDR_PLUS = 'command-r-plus'
    COMMANDR = 'command-r'
    COMMAND = 'command'
    DEFAULT_LLM = COMMANDR


class GPTModels(Enum):
    GPT_4 = "gpt-4-1106-preview"
    GPT_3 = "gpt-3.5-turbo"


class EmbeddingModels(Enum):
    HF_BGE_LARGE = "BAAI/bge-large-en-v1.5"
    HF_BGE_BASE = "BAAI/bge-base-en-v1.5"
    HF_BGE_SMALL = "BAAI/bge-small-en-v1.5"
    AS_THENLPER = "thenlper/gte-large"
    COHERE_EMBED = 'embed-multilingual-v3.0'
    DEFAULT_EMBED_MODEL = HF_BGE_SMALL


class LLMTypes(Enum):
    ANY_SCALE = "anyscale"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    OPENAI = "openai"
    DEFAULT_LLM = ANY_SCALE