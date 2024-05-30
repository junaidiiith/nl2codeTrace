from tqdm.auto import tqdm
import asyncio
import json
import os
import pickle
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from llama_index.core import Settings
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.indices import VectorStoreIndex
from collections import defaultdict
from api_models import get_api_keys
from code2graph import CLASS_NAME_LABEL, get_docs_nxg
from typing import List
from typing import Union
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, Document
from constants import BM25_INDEX_RETREIVER, VECTOR_INDEX_RETREIVER
from evaluation import evaluate_from_retriever, evaluate_response, get_solutions
from indexing.constants import (
    CLASS_NAME_LABEL,
    DOCSTRING_LABEL,
    ATTRIBUTES_LABEL,
    ATTRIBUTE_NAME_LABEL,
    ATTRIBUTES_TYPE_LABEL,
    METHOD_NAME_LABEL,
    METHODS_LABEL,
)
from llama_index.core import QueryBundle
from retrievers import get_reachable_nodes


def get_dataset_nodes(
    dataset_name,
    path: str = 'indexed_nodes',
    use_mc: bool = False,
    use_summary: bool = False
):
    indexed_nodes = pickle.load(open(f'{path}/{dataset_name}{"_mc" if use_mc else ""}{"_sm" if use_summary else ""}.pkl', 'rb'))
    print(f"Number of indexed nodes: {len(indexed_nodes)}")
    req_nodes = pickle.load(open(f'similar_requirements/{dataset_name}.pkl', 'rb'))
    print(f"Number of requirements nodes: {len(req_nodes)}")
    return indexed_nodes, req_nodes

def get_graph_node_str(graph_node):
    content = f"Class {graph_node[CLASS_NAME_LABEL]}\n"
    if DOCSTRING_LABEL in graph_node and graph_node[DOCSTRING_LABEL]:
        content += f"Docstring: {graph_node[DOCSTRING_LABEL]}\n"
        
    # if ATTRIBUTES_LABEL in graph_node and len(graph_node[ATTRIBUTES_LABEL]):
    #     content += f"Attributes: \n"
    #     for attr in graph_node[ATTRIBUTES_LABEL]:
    #         content += f"{attr[ATTRIBUTE_NAME_LABEL]}: {attr[ATTRIBUTES_TYPE_LABEL]}\n"
    
    # method_names = [i[METHOD_NAME_LABEL] for i in graph_node[METHODS_LABEL]] 
    # if len(method_names):
    #     content += f"Methods: {', '.join(method_names)}"
    return content


def get_graph_node_str(graph_node):
    content = f"Class {graph_node[CLASS_NAME_LABEL]}\n"
    if DOCSTRING_LABEL in graph_node and graph_node[DOCSTRING_LABEL]:
        content += f"Docstring: {graph_node[DOCSTRING_LABEL]}\n"
        
    # if ATTRIBUTES_LABEL in graph_node and len(graph_node[ATTRIBUTES_LABEL]):
    #     content += f"Attributes: \n"
    #     for attr in graph_node[ATTRIBUTES_LABEL]:
    #         content += f"{attr[ATTRIBUTE_NAME_LABEL]}: {attr[ATTRIBUTES_TYPE_LABEL]}\n"
    
    # method_names = [i[METHOD_NAME_LABEL] for i in graph_node[METHODS_LABEL]] 
    # if len(method_names):
    #     content += f"Methods: {', '.join(method_names)}"
    return content


class NL2CodeTracer(BaseRetriever):
    def __init__(
        self, 
        dataset_name,
        retrieval_distance: int = 1,
        similarity_threshold: float = 0.6,
        similarity_top_k: int = 2,
        base_dir='data_repos/ftlr/datasets',
        chroma_db_dir='indices',
        solutions_file='solution_links_english.txt',
        call_graph_file='method_callgraph.json',
        all_code_files_path='all_code_filenames.txt',
        all_req_file_names='all_req_filenames.txt',
        results_dir='results',
    ):

        self.similarity_top_k = similarity_top_k
        self.results_dir = results_dir
        self.sem_evaluator = SemanticSimilarityEvaluator(
            embed_model=Settings.embed_model,
            similarity_threshold=similarity_threshold,
        )
        self.retrieval_distance = retrieval_distance
        os.makedirs('results', exist_ok=True)
        self.dataset_name = dataset_name

        self.all_code_files_path = all_code_files_path
        self.all_req_file_names = all_req_file_names

        self.base_dir = base_dir
        self.dataset_dir = f'{base_dir}/{dataset_name}'
        indices_path = f"{chroma_db_dir}/{dataset_name}"
        os.makedirs(indices_path, exist_ok=True)

        self.solutions_file_path = f'{self.dataset_dir}/{dataset_name.lower()}_{solutions_file}'
        self.call_graph_file = f'{dataset_name.lower()}_{call_graph_file}'
        self.class_names2node_map = None
    

    def set_dataset_data(
            self, 
            use_mc=False, 
            use_summary=False,
            use_similar_q=False
        ):
        self.use_mc = use_mc
        self.use_summary = use_summary
        self.use_similar_q = use_similar_q
        
        indexed_nodes, req_nodes = get_dataset_nodes(
            self.dataset_name, 
            'indexed_nodes',
            use_mc=use_mc,
            use_summary=use_summary
        )

        if self.use_similar_q:
            for i in range(len(req_nodes)):
                similar_qs = '\n'.join(req_nodes[i].metadata['similar_queries'])
                req_nodes[i].text += f"\n\nSet of similar requirements as context: \n{similar_qs}"

        self.indexed_nodes = indexed_nodes
        self.req_nodes = req_nodes
        
        self.set_class_node_maps()
        self.set_docs_nxg_and_graph_nodes()
        self.set_solution_links()
    

    def set_solution_links(self):
        solutions = get_solutions(self.solutions_file_path)
        self.req_file_to_node_map = {
            n.metadata['file_name']: solutions[n.metadata['file_name']]
            for n in self.req_nodes
        }
    
    def set_class_node_maps(self):
        self.node_map = {n.hash: n for n in self.indexed_nodes}
        self.class_names2node_map = defaultdict(list)
        for n in self.indexed_nodes:
            self.class_names2node_map[n.metadata[CLASS_NAME_LABEL]].append(n.hash)


    def set_docs_nxg_and_graph_nodes(self):
        self.docs_nxg, self.graph_class_nodes = get_docs_nxg(
            dataset_dir=self.dataset_dir,
            all_code_files_path=self.all_code_files_path,
            callgraph_file_name=self.call_graph_file
        )
    
    async def get_semantic_score(self, query_str, doc_str):
        result = await self.sem_evaluator.aevaluate(response=doc_str, reference=query_str)
        return result


    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
            
        # print("Query: ", query_bundle.query_str)
        retrieved_nodes: List[NodeWithScore] = []
        for retriever in self.retrievers:
            retrieved_nodes.extend(retriever.retrieve(query_bundle.query_str))
        
        retrieved_nodes = list({n.node_id: n for n in retrieved_nodes}.values())
        
        class_names = list(set([n.metadata['Class Name'] for n in retrieved_nodes]))
        assert all(n in self.docs_nxg for n in class_names)

        # print("Class names", class_names)

        reachable_classes = list(
            set(sum([
                get_reachable_nodes(self.docs_nxg, class_name, self.retrieval_distance)\
                for class_name in class_names], []
            ))
        )
        node_with_scores = list()
        for c in reachable_classes:
            if c not in self.graph_class_nodes or c not in self.class_names2node_map:
                continue
            class_node_str = get_graph_node_str(self.graph_class_nodes[c])
            doc = self.node_map[self.class_names2node_map[c][0]]
            if 'section_summary' in doc.metadata:
                class_node_str += f"\n\nSummary: {doc.metadata['section_summary']}"

            node = Document(
                text=class_node_str, 
                metadata=doc.metadata,
                excluded_llm_metadata_keys=doc.excluded_llm_metadata_keys,
                excluded_embed_metadata_keys=doc.excluded_embed_metadata_keys
            )
            sim_result = asyncio.run(self.get_semantic_score(query_bundle.query_str, class_node_str))
            
            if sim_result.passing:
                node_with_score = NodeWithScore(node=node, score=sim_result.score)
                node_with_scores.append(node_with_score)
        
        print("Retrieved nodes: ", len(node_with_scores))
        return node_with_scores


    def set_retrivers(
        self,
        types: List[str],
    ):
        self.retrievers: Union[BaseRetriever, List[BaseRetriever]] = []
        if VECTOR_INDEX_RETREIVER in types:
            vector_index = VectorStoreIndex(nodes=self.indexed_nodes, show_progress=True)
            vector_retriever = vector_index.as_retriever(similarity_top_k=self.similarity_top_k)
            self.retrievers.append(vector_retriever)
        
        if BM25_INDEX_RETREIVER in types:
            bm25_retriever = BM25Retriever.from_defaults(
            nodes=self.indexed_nodes, 
            similarity_top_k=self.similarity_top_k,
        )
            self.retrievers.append(bm25_retriever)
    

    def evaluate_retrievers(self, retrievers: List[str], both=False):
        result_file_name = f'{self.results_dir}/{self.dataset_name}_'
        retrievers_str = '_'.join(retrievers)
        result_file_name += retrievers_str
        result_file_name += '_mc' if self.use_mc else ''
        result_file_name += '_sm' if self.use_summary else ''
        result_file_name += '_eq' if self.use_similar_q else ''
        result_file_name += f'_simk_{self.similarity_top_k}'
        result_file_name = f'{result_file_name}_results.json'
        print("Result file name: ", result_file_name)
        if os.path.exists(result_file_name):
            return
        self.set_retrivers(retrievers)
        
        
        qes = {f'{retrievers_str}_nl': RetrieverQueryEngine(self)}
        retrievers_dict = {f'{retrievers_str}_nl': self}
        
        if both:
            qes[retrievers_str] = RetrieverQueryEngine(self.retrievers[0])
            retrievers_dict[retrievers_str] = self.retrievers[0]


        retrieval_results = evaluate_from_retriever(
            req_nodes=self.req_nodes,
            retrievers=retrievers_dict,
            solutions_file=self.solutions_file_path,
            dataset_name=self.dataset_name,
            results_dir=self.results_dir,
        )
        correctness_results = evaluate_response(
            req_nodes=self.req_nodes,
            query_engines=qes,
            solutions_file=self.solutions_file_path,
            dataset_name=self.dataset_name,
            results_dir=self.results_dir,
        )
        results = {
            'retrieval_results': retrieval_results,
            'correctness_results': correctness_results
        }

        with open(result_file_name, 'w') as f:
            json.dump(results, f, indent=4)


    def trace(self):
        self.evaluate_retrievers([VECTOR_INDEX_RETREIVER], both=True)
        self.evaluate_retrievers([BM25_INDEX_RETREIVER], both=True)
        self.evaluate_retrievers([VECTOR_INDEX_RETREIVER, BM25_INDEX_RETREIVER])



if __name__ == '__main__':
    from parameters import parse_args
    import os
    from constants import (
        LLMsMap,
        EmbeddingModelsMap,
    )

    from api_models import set_llm_and_embed
    args = parse_args()

    # datasets = [
    #     ('smos', 'bge_m3'),
    #     ('eTour', 'bge_m3'), 
    #     ('eANCI', 'bge_m3'), 
    #     ('iTrust', 'bge_m3'),
    # ]
    use_mcs = [True, False]
    use_summaries = [True, False]
    use_similar_qs = [True, False]
    k = [4, 6, 8]
    os.makedirs(args.results_dir, exist_ok=True)
    dataset, embed_model = args.dataset_name, args.embed_model
    
    solutions_file = args.solutions_file
    api_key = get_api_keys(llm_type=args.llm_type, idx=args.api_key_index)
    llm_name = LLMsMap[args.llm]
    embed_model_name = EmbeddingModelsMap[embed_model]
    results_dir = args.results_dir

    print(f"Using LLM: {llm_name}")
    print(f"Using Embedding Model: {embed_model_name}")
    print(f"Dataset: {dataset}")

    set_llm_and_embed(
        llm_type=args.llm_type,
        llm_name=llm_name,
        embed_model_name=embed_model_name,
        api_key=api_key
    )
    if dataset == 'smos':
        solutions_file = 'solution_links_italian.txt'

    configs = [(mc, sm, eq, k) 
            for mc in use_mcs 
            for sm in use_summaries 
            for eq in use_similar_qs
            for k in k
        ]
    for config in tqdm(configs, desc="Config"):
        use_mc, use_summary, use_similar_q, k = config
        nl2code_tracer = NL2CodeTracer(
            dataset, 
            results_dir=results_dir,
            solutions_file=solutions_file,
            similarity_top_k=k
        )
        nl2code_tracer.set_dataset_data(
            use_mc=use_mc, 
            use_summary=use_summary, 
            use_similar_q=use_similar_q
        )
        print("Config: ", config)
        nl2code_tracer.trace()
        print("Done")
    
    print("Done with dataset")
    