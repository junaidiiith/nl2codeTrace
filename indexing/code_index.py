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
    METHODS_LABEL,
    extract_call_graph_links
)


def create_class_node_doc(
        graph_node, 
        add_method_calls=True
    ):

    content = f"{graph_node['type']} Name: " + graph_node[CLASS_NAME_LABEL] + "\n"
    
    if ATTRIBUTES_LABEL in graph_node and len(graph_node[ATTRIBUTES_LABEL]):
        content += f"Attributes: \n"
        for attr in graph_node[ATTRIBUTES_LABEL]:
            content += f"{attr[ATTRIBUTE_NAME_LABEL]}: {attr[ATTRIBUTES_TYPE_LABEL]}\n"
    
    if DOCSTRING_LABEL in graph_node:
        content += f"\n{graph_node[DOCSTRING_LABEL]}\n"
    
    content += "\nMethods: \n"
    
    method_nodes = graph_node[METHODS_LABEL]
    for method_node in method_nodes:
        content += f"\nMethod Name: {method_node[METHOD_NAME_LABEL]}\n"
        content += f"Signature: {method_node[METHOD_SIGNATURE_LABEL]}\n"
        content += f"Class Name: {graph_node[CLASS_NAME_LABEL]}\n"
        if METHOD_DOCSTRING in method_node:
            content += f"Docstring: \n{method_node[METHOD_DOCSTRING]}\n"

        if add_method_calls:
            if CALLS_LABEL in method_node and len(method_node[CALLS_LABEL]):
                
                content += f"\nCalls: \n"
                for call in method_node[CALLS_LABEL]:
                    content += f"{method_node[METHOD_NAME_LABEL]} calls {call}\n"
            
            if CALLED_BY_LABEL in method_node and len(method_node[CALLED_BY_LABEL]):
                content += f"\nCalled By: \n"
                for called_by in method_node[CALLED_BY_LABEL]:
                    content += f"{method_node[METHOD_NAME_LABEL]} called by {called_by}\n"

        content += "\n\n"

    doc = Document(
        text=content,
        metadata = {
            FILE_NAME_LABEL: graph_node[FILE_NAME_LABEL],
            CLASS_NAME_LABEL: graph_node[CLASS_NAME_LABEL],
        },
        excluded_embed_metadata_keys=[FILE_NAME_LABEL, 'excerpt_keywords', 'questions_this_excerpt_can_answer'],
        excluded_llm_metadata_keys=[FILE_NAME_LABEL, 'questions_this_excerpt_can_answer']
    )
    return doc


def create_code_graph_nodes_index(
        graph_nodes, 
        add_method_calls=True
    ):
    docs = [
        create_class_node_doc(graph_node, add_method_calls) \
        for graph_node in tqdm(graph_nodes.values(), desc="Creating Documents")
        if graph_node["type"] == "Class"
    ]
    
    print("Number of documents indexed:", len(docs))
    return docs