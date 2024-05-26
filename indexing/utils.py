import kuzu
from llama_index.graph_stores.kuzu import KuzuGraphStore
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import Settings
from llama_index.core import Document
from llama_index.core.schema import TextNode

from tqdm.auto import tqdm

from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    KeywordExtractor
)

from llama_index.core.node_parser import LangchainNodeParser
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter
)

import chromadb
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore



from llama_index.core.extractors.metadata_extractors import (
    DEFAULT_SUMMARY_EXTRACT_TEMPLATE,
    DEFAULT_QUESTION_GEN_TMPL,
    DEFAULT_TITLE_NODE_TEMPLATE
)

from prompts.templates import CODE_KEYWORD_EXTRACT_TEMPLATE_TMPL

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

PIPELINE_CHUNK_SIZE = 100


def get_transformations(
    llm=None,
    embed=False,
    embed_model=None,
    summary_extractor=False,
    summary_template=DEFAULT_SUMMARY_EXTRACT_TEMPLATE,
    num_questions=None, 
    questions_template=DEFAULT_QUESTION_GEN_TMPL,
    num_title_nodes=None,
    title_template=DEFAULT_TITLE_NODE_TEMPLATE,
    qa_prompt=None,
    keyword_extractor=False,
    keyword_extraction_template=CODE_KEYWORD_EXTRACT_TEMPLATE_TMPL,
    num_keywords=None
):
    llm = llm if llm is not None else Settings.llm
    transformations = list()

    if embed:
        transformations.append(Settings.embed_model if embed_model is None else embed_model)

    if summary_extractor:
        
        summary_extractor = SummaryExtractor(
            llm=llm,
            prompt_template=summary_template
        )
            
        transformations.append(summary_extractor)
    
    if num_title_nodes is not None:
        title_extractor = TitleExtractor(
            llm=llm,
            nodes=num_title_nodes,
            prompt_template=title_template
        )
        transformations.append(title_extractor)

    if num_questions is not None:
        qa_extractor = QuestionsAnsweredExtractor(
            llm=llm,
            questions=num_questions,
            prompt_template=questions_template
        )
        if qa_prompt is not None:
            qa_extractor.prompt_template = qa_prompt
        
        transformations.append(qa_extractor)
    
    if keyword_extractor:
        prompt_template=CODE_KEYWORD_EXTRACT_TEMPLATE_TMPL \
            if keyword_extraction_template else keyword_extraction_template
        kw_extractor = KeywordExtractor(
            llm=llm,
            prompt_template=prompt_template,
            keywords=num_keywords
        )
        transformations.append(kw_extractor)
    
    assert len(transformations) > 0, "No transformations provided"

    return transformations


def get_pipeline_chunks(
        nodes: Union[Document, TextNode],
        transformations: list=None,
        parser: LangchainNodeParser = None,
        pipeline_chunk_size:int = PIPELINE_CHUNK_SIZE
    ):

    transformations = transformations if transformations is not None else list()
    pipeline_chunks = list()

    for i in tqdm(range(0, len(nodes), pipeline_chunk_size), desc='Creating nodes'):
        docs = nodes[i:i+pipeline_chunk_size]
        if parser:
            docs = parser.get_nodes_from_documents(docs, show_progress=True)
        
        pipeline = IngestionPipeline(transformations=transformations)
        pipeline_chunks.append((pipeline, docs))
    
    return pipeline_chunks


def run_pipeline(pipeline, docs, show_progress=True):
    try:
        index_nodes = pipeline.run(
            nodes=docs, 
            show_progress=show_progress
        )
    except Exception as e:
        print(e)
        index_nodes = []
        
    return index_nodes


def run_pipeline_multithreaded(
        nodes, 
        transformations: list = list(),
        num_threads=4,
        pipeline_chunk_size=PIPELINE_CHUNK_SIZE,
        show_progress=True
    ) -> List[TextNode]:

    pipeline_chunks = get_pipeline_chunks(
        nodes,
        transformations=transformations,
        pipeline_chunk_size=pipeline_chunk_size
    )

    indexed_nodes = list()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = list()
        for i, (pipeline_docs) in enumerate(pipeline_chunks):
            pipeline, docs = pipeline_docs
            futures.append(
                executor.submit(run_pipeline, pipeline, docs, show_progress)
            )
        
        for future in futures:
            indexed_nodes += future.result()

    return indexed_nodes


def get_parser(
        language: Language = None
):
    chunk_size = CHUNK_SIZE
    chunk_overlap = CHUNK_OVERLAP

    if language:
        splitter = RecursiveCharacterTextSplitter.from_language(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            language=language
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    parser = LangchainNodeParser(splitter)
    return parser


def get_vector_storage_context(chroma_db_path, collection_name):
    db = chromadb.PersistentClient(path=f"{chroma_db_path}")
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context, vector_store


def get_kuzu_graph_store(collection_name):
    db = kuzu.Database(collection_name)
    graph_store = KuzuGraphStore(db)

    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    return storage_context
