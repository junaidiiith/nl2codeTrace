from concurrent.futures import ThreadPoolExecutor
import os
import pickle
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.indices import VectorStoreIndex
from llama_index.core import Settings

from tqdm.auto import tqdm

from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
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

from constants import (
    PIPELINE_CHUNK_SIZE,
    PIPELINE_CHUNK_OVERLAP
)


def get_transformations(
    llm,
    summary_extractor=False,
    summary_template=DEFAULT_SUMMARY_EXTRACT_TEMPLATE,
    num_questions=None, 
    questions_template=DEFAULT_QUESTION_GEN_TMPL,
    num_title_nodes=None,
    title_template=DEFAULT_TITLE_NODE_TEMPLATE,
    qa_prompt=None
):

    transformations = list()

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

    return transformations


def get_pipeline_chunks(
        nodes,
        parser,
        embed_model=None,
        transformations=None
    ):

    embed_model = embed_model if embed_model is not None else Settings.embed_model
    pipeline_chunks = list()

    for i in tqdm(range(0, len(nodes), PIPELINE_CHUNK_SIZE), desc='Creating nodes'):
        docs = nodes[i:i+PIPELINE_CHUNK_SIZE]
        splitted_docs = parser.get_nodes_from_documents(docs, show_progress=True)
        transformations = list() if transformations is None else transformations
        transformations += [embed_model]

        pipeline = IngestionPipeline(
            transformations=transformations if transformations else list()
        )
        pipeline_chunks.append((pipeline, splitted_docs))
    
    return pipeline_chunks


def run_pipeline(pipeline, docs, save_loc='tmp.pkl', show_progress=True):
    if os.path.exists(save_loc):
        with open(save_loc, 'rb') as f:
            return pickle.load(f)
    try:
        # print(pipeline.transformations[0].llm)
        index_nodes = pipeline.run(
            nodes=docs, 
            show_progress=show_progress
        )
    except Exception as e:
        print(e)
        print(f"Error in {save_loc}")
        index_nodes = []
        
    return index_nodes



def run_pipeline_multithreaded(
        nodes, 
        embed_model=None,
        transformations=None,
        num_threads=4,
        pickle_dir='tmp',
        show_progress=True
    ):

    os.makedirs(pickle_dir, exist_ok=True)
    pipeline_chunks = get_pipeline_chunks(
        nodes,
        embed_model=embed_model,
        transformations=transformations
    )

    indexed_nodes = list()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = list()
        for i, (pipeline_docs) in enumerate(pipeline_chunks):
            pipeline, docs = pipeline_docs
            pickle_path = f"{pickle_dir}/pipeline_{i}.pkl"
            futures.append(
                executor.submit(run_pipeline, pipeline, docs, pickle_path, show_progress)
            )
        
        for future in futures:
            indexed_nodes += future.result()

    return indexed_nodes


def get_parser(
    language=Language.JAVA,
):


    chunk_size = PIPELINE_CHUNK_SIZE
    chunk_overlap = PIPELINE_CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language,
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