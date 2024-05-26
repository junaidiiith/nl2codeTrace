from typing import List
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
import json
import os
import pickle

from llama_index.core.schema import TextNode, MetadataMode
from evaluation import evaluate_indices, evaluate_query_engines, generate_qa_dataset, get_questions_from_nodes
from req2nodes import get_requirements_nodes
from retrievers import custom_query_engine
from api_models import set_llm_and_embed
from indexing.utils import get_parser
from indexing.indices import create_keyword_table_index, create_knowledge_graph_index, get_graph_store
from code2graph import get_code_graph_nodes
from indexing.code_index import create_code_graph_nodes_index
from indexing.utils import get_vector_storage_context, get_kuzu_graph_store
from indexing.indices import create_vector_index
from llama_index.core.schema import Document
from prompts.templates import CODE_KEYWORD_EXTRACT_TEMPLATE
from prompts.templates import CODE_KG_KEYWORD_EXTRACT_TEMPLATE
from argparse import ArgumentParser
from constants import (
    EmbeddingModels,
    LLMTypes, 
    LLMs,
    LLMsMap,
    EmbeddingModelsMap,
    KG_INDEX_LABEL,
    KW_TABLE_LABEL,
    KG_INDEX_CUSTOM_LABEL,
    VECTOR_INDEX_LABEL
)
from llama_index.core.indices import KnowledgeGraphIndex, KeywordTableIndex


from parameters import parse_args


def set_models(llm_type, llm_name, embed_model):
    llm_name = LLMsMap[llm_name]
    embed_model_name = EmbeddingModelsMap[embed_model]
    print(f"Using LLM: {llm_name}")
    print(f"Using Embedding Model: {embed_model_name}")

    set_llm_and_embed(
        llm_type=llm_type,
        llm_name=llm_name,
        embed_model_name=embed_model_name,
    )

def get_vector_index(docs, indices_path, dataset_name):
    print(f"Creating Vector Index...")
    storage_context, _ = get_vector_storage_context(
        chroma_db_path=f'{indices_path}',
        collection_name=f'{dataset_name}',
    )

    _, vector_index = create_vector_index(
        nodes=docs, 
        storage_context=storage_context, 
        num_threads=8,
        save_path=f'{indices_path}',
        show_progress=True
    )
    print(f"Vector Index created.")
    return vector_index


def get_kg_links_index(
        docs, 
        graph_nodes, 
        dataset_name,
        include_embeddings=False,
    ):
    graph_storage_context = get_graph_store(
        db_name=f'nl2code{dataset_name}',
    )
    index = create_knowledge_graph_index(
        docs, 
        graph_nodes, 
        graph_storage_context,
        include_embeddings=include_embeddings
    )
    return index


def get_kw_table_index(
        docs, 
        indices_path, 
        dataset_name
    ) -> KeywordTableIndex:
    print(f"Creating Keyword Table Index...")
    storage_context, _ = get_vector_storage_context(
        chroma_db_path=f'{indices_path}',
        collection_name=f'{dataset_name}',
    )

    _, kw_index = create_keyword_table_index(
        docs=docs, 
        storage_context=storage_context, 
        num_keywords=20,
        num_threads=8,
        save_path=f'{indices_path}',
        show_progress=True
    )
    print(f"Keyword Table Index created.")
    return kw_index


def get_kg_index(
        docs: List[TextNode],
        dataset_name
    ) -> KnowledgeGraphIndex:

    graph_storage_index = get_graph_store(
        db_name=f'nl2code{dataset_name}',
    )
    
    docs = [Document(text=doc.get_content(MetadataMode.LLM)) for doc in docs]

    kg_index = KnowledgeGraphIndex.from_documents(
        documents=docs,
        storage_context=graph_storage_index,
    )
    return kg_index


def execute_dataset(args, dataset_name):
    args.dataset_name = dataset_name

    base_dir = args.base_dir
    dataset_name = args.dataset_name
    dataset_dir = f'{base_dir}/{dataset_name}'
    indices_path = f"{args.chroma_db_dir}/{dataset_name}"
    os.makedirs(indices_path, exist_ok=True)

    solutions_file = args.solutions_file
    solutions_file_path = f'{dataset_dir}/{dataset_name}_{solutions_file}'

    call_graph_file = f'{dataset_name.lower()}_{args.call_graph_file}'


    graph_nodes = get_code_graph_nodes(
        base_dir=dataset_dir,
        all_code_files_path=args.all_code_files_path,
        callgraph_file_name=call_graph_file,
    )

    print(f"Indexing code nodes...")
    code_nodes = create_code_graph_nodes_index(
        graph_nodes=graph_nodes
    )
    print(f"Number of code nodes: {len(code_nodes)}")
    parser = get_parser()
    print(f"Getting nodes from documents...")
    docs = parser.get_nodes_from_documents(code_nodes)
    print(f"Number of nodes: {len(docs)}")

    
    vector_index = get_vector_index(docs, indices_path, dataset_name)
    print(f"Creating KG Links Index...")
    kg_links_index = get_kg_links_index(
        docs, 
        graph_nodes,
        dataset_name,
        include_embeddings=args.include_embeddings
    )
    print(f"KG Links Index created.")

    print(f"Creating KG Index...")
    kg_index = get_kg_index(docs, dataset_name)
    kw_index = get_kw_table_index(docs, indices_path, dataset_name)
    print(f"Indices created.")

    query_engines = {
        'vector_index': vector_index.as_query_engine(),
        'kg_links_index': kg_links_index.as_query_engine(),
        'kg_index': kg_index.as_query_engine(),
        'kw_index': kw_index.as_query_engine(),
        'vector_kg_links': custom_query_engine(
            vector_index.as_retriever(),
            kg_links_index.as_retriever(),
        ),
        'vector_kw_table': custom_query_engine(
            vector_index.as_retriever(),
            kw_index.as_retriever(),
        ),
        'vector_kg': custom_query_engine(
            vector_index.as_retriever(),
            kg_index.as_retriever(),
        ),
    }

    req_nodes = get_requirements_nodes(
        dataset_dir,
        all_req_files_path=args.all_req_filenames,
    )

    correctness_results = evaluate_query_engines(
        req_nodes=req_nodes,
        query_engines=query_engines,
        solutions_file=solutions_file_path,
    )

    indices = {
        'vector_index': vector_index,
        'kg_links_index': kg_links_index,
        'kg_index': kg_index,
        'kw_index': kw_index,
    }

    questions = get_questions_from_nodes(code_nodes)
    print(f"Number of questions: {len(questions)}")
    eval_results = evaluate_indices(questions, indices)
    
    results = {
        'correctness': correctness_results,
        'evaluation': eval_results,
    }
    with open(f'results/results.json', 'w') as f:
        json.dump(results, f, indent=4)


def main():
    configs = [
        ('eTour', 'bge_small'),
        ('iTrust', 'bge_small'),
        ('smos', 'bge_small'),
        ('eANCI', 'bge_m3'),

    ]
    args = parse_args()
    for dataset_name, embed_model in configs:
        set_models(
            llm_type=args.llm_type,
            llm_name=args.llm,
            embed_model=embed_model,
        )

        execute_dataset(args, dataset_name)
        break


if __name__ == '__main__':
    main()