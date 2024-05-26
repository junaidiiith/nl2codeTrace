from llama_index.core import Document
from collections import defaultdict
import json
from typing import List, Union
from querying import query_parallel, CLASS_TRACE_TEMPLATE

from tqdm.auto import tqdm
import json
import re
from typing import Dict, List, Tuple
import uuid
from llama_index.core.schema import TextNode
from llama_index.core.schema import MetadataMode
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.evaluation import (
    BatchEvalRunner,
    ContextRelevancyEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator
)
from llama_index.core.indices import (
    VectorStoreIndex,
    KeywordTableIndex,
    KnowledgeGraphIndex
)
from indexing.utils import run_pipeline_multithreaded, get_transformations
from prompts.templates import REQ2CODE_QA_TEMPLATE



class QADataset(BaseModel):
    """Embedding QA Finetuning Dataset.

    Args:
        queries (Dict[str, str]): Dict id -> query.
        corpus (Dict[str, str]): Dict id -> string.
        relevant_docs (Dict[str, List[str]]): Dict query id -> list of doc ids.

    """

    queries: Dict[str, str]  # dict id -> query
    corpus: Dict[str, str]  # dict id -> string
    relevant_docs: Dict[str, List[str]]  # query id -> list of doc ids
    mode: str = "text"

    @property
    def query_docid_pairs(self) -> List[Tuple[str, List[str]]]:
        """Get query, relevant doc ids."""
        return [
            (query, self.relevant_docs[query_id])
            for query_id, query in self.queries.items()
        ]

    def save_json(self, path: str) -> None:
        """Save json."""
        with open(path, "w") as f:
            json.dump(self.dict(), f, indent=4)

    @classmethod
    def from_json(cls, path: str) -> "QADataset":
        """Load json."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def generate_qa_dataset(
    nodes: List[TextNode],
    qa_generate_prompt_tmpl: str = REQ2CODE_QA_TEMPLATE,
    num_questions_per_chunk: int = 3,
    pipeline_chunk_size=10,
):
    
    if 'questions_this_excerpt_can_answer' not in nodes[0].metadata.keys() == set():
        nodes = run_pipeline_multithreaded(
            nodes,
            transformations=get_transformations(
                num_questions=num_questions_per_chunk,
                questions_template=qa_generate_prompt_tmpl,
            ),
            pipeline_chunk_size=pipeline_chunk_size
        )

    queries = {}
    relevant_docs = {}
    for node in tqdm(nodes):
        node_id = node.node_id
        questions_response = node.metadata['questions_this_excerpt_can_answer']
        result = questions_response.strip().split("\n")
        questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
        ]
        questions = [question for question in questions if len(question) > 0]

        for question in questions:
            question_id = str(uuid.uuid4())
            queries[question_id] = question
            relevant_docs[question_id] = [node_id]

    node_dict = {
        node.node_id: node.get_content(metadata_mode=MetadataMode.NONE)
        for node in nodes
    }

    return QADataset(
        queries=queries, corpus=node_dict, relevant_docs=relevant_docs
    )



def get_post_processing_results(req_results):
    ok = True
    llm_results = defaultdict(set)
    for file_name, result in req_results:
        try:
            class_names_list = json.loads(result.response)
            for class_name in class_names_list:
                llm_results[file_name].add(class_name)
        except Exception as e:
            print(file_name, result.response)
            ok = False

    for k, value in llm_results.items():
        llm_results[k] = list(value)
    
    if not ok:
        print("Some results are invalid")
    else:
        print("All results are valid")

    return llm_results


def get_solutions(file_name):
    gts = [line for line in open(file_name).read().split('\n') if line]
    solutions = defaultdict(list)
    for gt in gts:
        gt_split = gt.split(': ')
        file_name = gt_split[0]
        class_name = gt_split[1].split('.java')[0]
        solutions[file_name].append(class_name)
    return solutions


def compare_solutions(solutions, llm_results, result_file_name='results.json'):
    results = list()
    tp, fp = 0, 0
    tn, fn = 0, 0
    
    for file_name, classes in solutions.items():
        if file_name in llm_results:
            tp += len(set(classes).intersection(set(llm_results[file_name])))
            fp += len(set(llm_results[file_name]) - set(classes))
            fn += len(set(classes) - set(llm_results[file_name]))
            
            result = {
                "file_name": file_name,
                "expected_classes": sorted(classes),
                "llm_classes": sorted(llm_results[file_name])
            }
            results.append(result)
    with open(f"results/{result_file_name}", 'w') as f:
        json.dump(results, f, indent=4)
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)


def get_results(
        query_engines, 
        query_template: str,
        req_nodes,
        base_dir: str = 'etour_solution_links_english.txt',
    ):
    for config, query_engine in query_engines.items():
        print(f"Evaluating for {config}")
        req_results = query_parallel(
            query_engine, 
            query_template, 
            req_nodes, 
            num_threads=8
        )
        llm_results = get_post_processing_results(req_results)
        print(f"Results for {config}")

        solutions = get_solutions(f'{base_dir}/etour_solution_links_english.txt')
        compare_solutions(solutions, llm_results)


def evaluate_query_engines(
        req_nodes: List[Document], 
        query_engines: dict,
        solutions_file: str
    ):
    results = dict()
    solutions = get_solutions(solutions_file)
    for config, query_engine in query_engines.items():
        print(f"Evaluating for {config}")
        req_results = query_parallel(
            query_engine, 
            CLASS_TRACE_TEMPLATE, 
            req_nodes, 
            num_threads=8
        )
        llm_results = get_post_processing_results(req_results)
        print(f"Results for {config}")
        config_results = compare_solutions(solutions, llm_results)
        results[config] = config_results

        with open(f'results/{config}_correctness_results.json', 'w') as f:
            json.dump(config_results, f, indent=4)

    return results



def get_questions_from_nodes(nodes):
    dataset_generator = RagDatasetGenerator.from_documents(
        nodes,
        num_questions_per_chunk=3,  # set the number of questions per nodes
        show_progress=True,
        question_gen_query=REQ2CODE_QA_TEMPLATE,
        workers=8,
    )
    rag_dataset = dataset_generator.generate_questions_from_nodes()
    questions = [e.query for e in rag_dataset.examples]
    return questions


def evaluate_index(
        questions: List, 
        index: Union[VectorStoreIndex, KeywordTableIndex, KeywordTableIndex],
        num_workers: int = 8
    ):
    
    runner = BatchEvalRunner(
        {
            "faithfulness": FaithfulnessEvaluator(), 
            "relevancy": RelevancyEvaluator(),
            # "context_relevancy": ContextRelevancyEvaluator(),
        },
        workers=num_workers,
    )

    eval_results = runner.evaluate_queries(
        index.as_query_engine(), queries=questions
    )
    extract_result = lambda results, key: sum(result.passing for result in results[key]) / len(results[key])
    faithfulness = extract_result(eval_results, "faithfulness")
    relevancy = extract_result(eval_results, "relevancy")
    # context_relevancy = extract_result(eval_results, "context_relevancy")
    result = {
        "faithfulness": faithfulness,
        "relevancy": relevancy,
        # "context_relevancy": context_relevancy,
    }

    return result


def evaluate_indices(
        questions: List[str], 
        indices: Dict[str, Union[VectorStoreIndex, KeywordTableIndex, KnowledgeGraphIndex]],
        num_workers: int = 8
    ):
    results = dict()
    for index_name, index in indices.items():
        print(f"Evaluating {index_name}")
        eval_results = evaluate_index(questions, index, num_workers)
        results[index_name] = eval_results

        with open(f'results/{index_name}_evaluation_results.json', 'w') as f:
            json.dump(eval_results, f, indent=4)

    return results