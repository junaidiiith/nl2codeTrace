from indexing.utils import run_pipeline_multithreaded
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices import KeywordTableIndex
from llama_index.core import StorageContext


def create_vector_index(
        nodes, 
        storage_context, 
        embed_model, 
        transformations=None,
        num_threads=4,
        pickle_dir='tmp',
        show_progress=True
    ):

    indexed_nodes = run_pipeline_multithreaded(
        nodes, 
        embed_model,
        transformations=transformations,
        num_threads=num_threads,
        pickle_dir=pickle_dir,
        show_progress=show_progress
    )

    vector_index = VectorStoreIndex(
        nodes=indexed_nodes, 
        storage_context=storage_context,
        show_progress=show_progress
    )

    return indexed_nodes, vector_index


def create_keyword_table_index(
        docs, 
        storage_context: StorageContext
):
    
    keyword_index = KeywordTableIndex.from_documents(
        docs,
        storage_context=storage_context,
        show_progress=True
    )
    return keyword_index