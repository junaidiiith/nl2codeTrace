import os
import pickle
from evaluation import evaluate_query_engines
from req2nodes import get_requirements_nodes
from retrievers import custom_query_engine
from api_models import set_llm_and_embed
from indexing.indices import create_kg_index, create_keyword_table_index
from code2graph import get_code_graph_nodes
from indexing.code_index import create_code_graph_nodes_index
from indexing.utils import get_vector_storage_context, get_kuzu_graph_store
from indexing.indices import create_vector_index
from argparse import ArgumentParser
from constants import (
    EmbeddingModels,
    LLMTypes, 
    LLMs,
    LLMsMap,
    EmbeddingModelsMap
)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='data_repos/ftlr/datasets')
    parser.add_argument('--dataset_name', type=str, default='eTour')
    parser.add_argument('--all_code_files_path', type=str, default='all_code_filenames.txt')
    parser.add_argument('--all_req_filenames', type=str, default='all_req_filenames.txt')
    parser.add_argument('--call_graph_file', type=str, default='etour_method_callgraph.json')
    parser.add_argument('--solutions_file', type=str, default='etour_solution_links_english.txt')

    parser.add_argument('--chroma_db_dir', type=str, default='indices')
    parser.add_argument('--collection_name', type=str, default='ftlr_etour')

    parser.add_argument('--num_threads', type=int, default=8)
    parser.add_argument('--save_path', type=str, default='eTour')

    parser.add_argument('--show_progress', type=bool, default=True)


    parser.add_argument('--embed_model', type=str, choices=list(EmbeddingModels._value2member_map_.keys()), default='bge_small')
    parser.add_argument('--llm_type', type=str, choices=list(LLMTypes._value2member_map_.keys()), default='anyscale')
    parser.add_argument('--llm', type=str, choices=list(LLMs._value2member_map_.keys()), default='llama3')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.chroma_db_dir, exist_ok=True)
    base_dir = args.base_dir
    dataset_name = args.dataset_name
    dataset_dir = f'{base_dir}/{dataset_name}'

    indices_path = f"{args.chroma_db_dir}/{dataset_name}"
    os.makedirs(indices_path, exist_ok=True)

    solutions_file = args.solutions_file
    solutions_file_path = f'{dataset_dir}/{solutions_file}'

    llm_name = LLMsMap[args.llm]
    embed_model_name = EmbeddingModelsMap[args.embed_model]
    print(f"Using LLM: {llm_name}")
    print(f"Using Embedding Model: {embed_model_name}")

    set_llm_and_embed(
        llm_type=args.llm_type,
        llm_name=llm_name,
        embed_model_name=embed_model_name,
    )


    graph_nodes = get_code_graph_nodes(
        base_dir=dataset_dir,
        all_code_files_path=args.all_code_files_path,
        callgraph_path=args.call_graph_file
    )

    nodes = create_code_graph_nodes_index(graph_nodes)

    if os.path.exists(f"{indices_path}/vector_index.pkl"):
        print("Vector Index already exists. Skipping indexing")
        vector_index = pickle.load(open(f"{indices_path}/vector_index.pkl", 'rb'))
    else:
        storage_context, _ = get_vector_storage_context(
            chroma_db_path=f'{indices_path}',
            collection_name=f'{dataset_name}',
        )
        _, vector_index = create_vector_index(
        nodes=nodes, 
        storage_context=storage_context, 
        num_threads=8,
        save_path=f'{dataset_name}',
        show_progress=True
    )
    
    if os.path.exists(f"{dataset_dir}/kg_index.pkl"):
        print("KG Index already exists. Skipping indexing")
        kg_index = pickle.load(open(f"{indices_path}/kg_index.pkl", 'rb'))
    else:
        storage_context = get_kuzu_graph_store(f"{indices_path}/{dataset_name}_graph")
        kg_index = create_kg_index(
            docs=nodes,
            storage_context=storage_context,
            save_path=f'{dataset_name}',
        )
    
    if os.path.exists(f"{dataset_dir}/kw_table.pkl"):
        print("KG Table Index already exists. Skipping indexing")
        kg_table = pickle.load(open(f"{indices_path}/kw_table.pkl", 'rb'))
    else:
        kg_table = create_keyword_table_index(
            docs=nodes,
            storage_context=storage_context,
            save_path=f'{dataset_name}',
        )

    kw_table_qe = custom_query_engine(
        vector_retriever=vector_index.as_retriever(),
        kg_retriever=kg_table.as_retriever(),
    )

    kg_qe = custom_query_engine(
        vector_retriever=vector_index.as_retriever(),
        kg_retriever=kg_index.as_retriever(),
    )

    query_engines = {
        'vector_qe': vector_index.as_query_engine(),
        'kg_qe': kg_index.as_query_engine(),
        'kw_qe': kg_table.as_query_engine(),
        'kw_vi_table': kw_table_qe,
        'kg_vi': kg_qe,
    }

    req_nodes = get_requirements_nodes(
        dataset_dir,
        all_req_files_path=args.all_req_filenames,
    )
    evaluate_query_engines(
        req_nodes=req_nodes,
        query_engines=query_engines,
        solutions_file=solutions_file_path,
    )

if __name__ == '__main__':
    main()