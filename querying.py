from concurrent.futures import ThreadPoolExecutor
from llama_index.core import Document
from tqdm.auto import tqdm

from prompts.templates import (
    CLASS_TRACE_TEMPLATE
)
import evaluation

from collections import defaultdict
from typing import List
from llama_index.core.query_engine import BaseQueryEngine



def query_parallel(
        query_engine: BaseQueryEngine,
        query_template: str, 
        req_nodes: List[Document], 
        num_threads=8
    ):
    progress_bar = tqdm(total=len(req_nodes), desc="Processing", unit="Requirement")
    futures = list()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for req_node in req_nodes:
            future = executor.submit(
                query_engine.query,
                query_template.format(requirement=req_node.text)
            )
            futures.append((req_node.metadata["file_name"], future))

        results = list()
        for file_name, future in futures:
            results.append((file_name, future.result()))
            progress_bar.update(1)
    return results
