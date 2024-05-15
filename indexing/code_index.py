from llama_index.core import Document
from tqdm.auto import tqdm
from code2graph import (
    FILE_NAME_LABEL,
    CLASS_NAME_LABEL,
    DOCSTRING_LABEL,
    ATTRIBUTES_LABEL,
    ATTRIBUTE_NAME_LABEL,
    ATTRIBUTES_TYPE_LABEL,
    METHOD_NAME_LABEL,
    METHOD_SIGNATURE_LABEL,
    METHOD_DOCSTRING,
    CALLS_LABEL,
    CALLED_BY_LABEL,
)


def create_class_node_doc(graph_node):
    content = f"{graph_node['type']} Name: " + graph_node[CLASS_NAME_LABEL] + "\n"
    
    if ATTRIBUTES_LABEL in graph_node and len(graph_node[ATTRIBUTES_LABEL]):
        content += f"Attributes: \n"
        for attr in graph_node[ATTRIBUTES_LABEL]:
            content += f"{attr[ATTRIBUTE_NAME_LABEL]}: {attr[ATTRIBUTES_TYPE_LABEL]}\n"
    
    if DOCSTRING_LABEL in graph_node:
        content += f"\n{graph_node[DOCSTRING_LABEL]}\n"
    
    doc = Document(
        text=content,
        metadata = {
            FILE_NAME_LABEL: graph_node[FILE_NAME_LABEL],
            "type": "Class"
        },
        excluded_embed_metadata_keys=["type"],
        excluded_llm_metadata_keys=["type"]
    )
    return doc


def create_method_node_doc(graph_node, show_calls=False):
    content = f"Class Name: {graph_node[CLASS_NAME_LABEL]}\n"
    content += f"{graph_node['type']} Name: {graph_node[METHOD_NAME_LABEL]}\n"
    content += f"Signature: {graph_node[METHOD_SIGNATURE_LABEL]}\n"
    
    if METHOD_DOCSTRING in graph_node:
        content += f"\n{graph_node[METHOD_DOCSTRING]}\n"
    
    if show_calls:
        if CALLS_LABEL in graph_node:
            content += f"\nCalls: \n"
            for call in graph_node[CALLS_LABEL]:
                content += f"{call}\n"
        
        if CALLED_BY_LABEL in graph_node:
            content += f"\nCalled By: \n"
            for called_by in graph_node[CALLED_BY_LABEL]:
                content += f"{called_by}\n"
    
    doc = Document(
        text=content,
        metadata = {
            FILE_NAME_LABEL: graph_node[FILE_NAME_LABEL],
            CALLS_LABEL: ", ".join(graph_node[CALLS_LABEL]) if CALLS_LABEL in graph_node else None,
            CALLED_BY_LABEL: ", ".join(graph_node[CALLED_BY_LABEL]) if CALLED_BY_LABEL in graph_node else None,
            "type": "Method",
            METHOD_SIGNATURE_LABEL: graph_node[METHOD_SIGNATURE_LABEL],
        },
        excluded_embed_metadata_keys=[METHOD_SIGNATURE_LABEL, CALLS_LABEL, CALLED_BY_LABEL, "type"],
        excluded_llm_metadata_keys=[METHOD_SIGNATURE_LABEL, CALLS_LABEL, CALLED_BY_LABEL, "type"],
    )
    return doc


def create_code_graph_nodes_index(graph_nodes):
    docs = [
        create_class_node_doc(graph_node) \
        if graph_node["type"] == "Class" else create_method_node_doc(graph_node) \
        for graph_node in tqdm(graph_nodes.values(), desc="Creating Documents")
    ]
    return docs