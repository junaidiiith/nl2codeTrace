import json
import os
import javalang

import javalang.tree
from tqdm.auto import tqdm

FILE_NAME_LABEL = "File Name"
CLASS_NAME_LABEL = "Class Name"
DOCSTRING_LABEL = "Docstring"
ATTRIBUTES_LABEL = "Attributes"
ATTRIBUTE_NAME_LABEL = "Attribute Name"
ATTRIBUTES_TYPE_LABEL = "Attribute Type"
METHODS_LABEL = "Methods"
METHOD_NAME_LABEL = "Method Name"
METHOD_SIGNATURE_LABEL = "Signature"
METHOD_DOCSTRING = "Method Docstring"
METHOD_PARAMETERS_LABEL = "Method Parameters"
METHOD_RETURN_LABEL = "Method Return"
PARAMETERS = "Parameters"
PARAM_NAME_LABEL = "Parameter Name"
PARAM_TYPE_LABEL = "Parameter Type"
PARAM_DESCRIPTION_LABEL = "Description"
CALLS_LABEL = "calls"
CALLED_BY_LABEL = "called_by"


def parse_java_file(file_path):
    with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as file:
        content = file.read()

    try:
        tree = javalang.parse.parse(content)
    except (javalang.parser.JavaSyntaxError, javalang.tokenizer.LexerError) as e:
        return {
            FILE_NAME_LABEL: file_path.split('/')[-1],
            CLASS_NAME_LABEL: file_path.split('/')[-1].split('.')[0],
        }
        

    # Find the main class declaration
    class_decl = next(
        (type_decl for type_decl in tree.types \
         if isinstance(type_decl, javalang.tree.ClassDeclaration) or \
            isinstance(type_decl, javalang.tree.InterfaceDeclaration) or \
            isinstance(type_decl, javalang.tree.EnumDeclaration)), None
    )

    if class_decl is None:
        return {
            FILE_NAME_LABEL: file_path.split('/')[-1],
            CLASS_NAME_LABEL: file_path.split('/')[-1].split('.')[0],
        }

    class_info = {
        CLASS_NAME_LABEL: class_decl.name,
        DOCSTRING_LABEL: class_decl.documentation if class_decl.documentation else "",
        ATTRIBUTES_LABEL: [],
        METHODS_LABEL: [],
        FILE_NAME_LABEL: file_path.split('/')[-1]
    }

    # print(f"Processing {class_info[CLASS_NAME_LABEL]}")

    

    # Extract attributes and methods
    for _, node in class_decl.filter(javalang.tree.FieldDeclaration):
        for field in node.declarators:
            class_info[ATTRIBUTES_LABEL].append({
                ATTRIBUTE_NAME_LABEL: field.name,
                ATTRIBUTES_TYPE_LABEL: node.type.name,
            })
    

    for _, node in class_decl.filter(javalang.tree.MethodDeclaration):
        method_info = {
            METHOD_NAME_LABEL: node.name,
            METHOD_DOCSTRING: node.documentation if node.documentation else None,
            METHOD_PARAMETERS_LABEL: [],
            METHOD_RETURN_LABEL: node.return_type.name if node.return_type else "void"
        }
        for param in node.parameters:
            method_info[METHOD_PARAMETERS_LABEL].append({
                PARAM_NAME_LABEL: param.name,
                PARAM_TYPE_LABEL: param.type.name,
            })
        class_info[METHODS_LABEL].append(method_info)

    # if isinstance(class_decl, javalang.tree.EnumDeclaration):
    #     print(class_info)
    return class_info


def get_graph_nodes(class_info_objects):
    graph_nodes = dict()
    for class_info in class_info_objects:
        class_name = class_info[CLASS_NAME_LABEL]
        file_name = class_info[FILE_NAME_LABEL]
        node = {
            CLASS_NAME_LABEL: class_name,
            FILE_NAME_LABEL: file_name,
            "type": "Class"
        }
        if DOCSTRING_LABEL in class_info:
            node[DOCSTRING_LABEL] = class_info[DOCSTRING_LABEL]
        if ATTRIBUTES_LABEL in class_info:
            node[ATTRIBUTES_LABEL] = class_info[ATTRIBUTES_LABEL]
        graph_nodes[class_info[CLASS_NAME_LABEL]] = node

        if METHODS_LABEL not in class_info:
            continue

        for method_info in class_info[METHODS_LABEL]:
            method_name = method_info[METHOD_NAME_LABEL]
            method_key = f'{class_name}.{method_name}'
            params_str = f"({','.join([param[PARAM_TYPE_LABEL] for param in method_info[METHOD_PARAMETERS_LABEL]])})"
            method_key_str = f'{method_key}{params_str}'

            node = {
                CLASS_NAME_LABEL: class_name,
                FILE_NAME_LABEL: class_info[FILE_NAME_LABEL],
                METHOD_NAME_LABEL: method_info[METHOD_NAME_LABEL],
                "type": "Method",
                METHOD_SIGNATURE_LABEL: method_key_str
            }

            if METHOD_DOCSTRING in method_info:
                node[METHOD_DOCSTRING] = method_info[METHOD_DOCSTRING]
            
            graph_nodes[method_key_str] = node

    print("Number of nodes in the graph:", len(graph_nodes))
    return graph_nodes


def get_class_info_objects(
        base_dir: str,
        all_code_files_path: str = 'all_code_filenames.txt',
    ):
    all_code_files_path = os.path.join(base_dir, all_code_files_path)
    all_code_files_names = [f_name.strip() for f_name in open(all_code_files_path).read()\
                            .split('\n') if f_name.strip()]

    code_file_paths = dict()
    for root, _, files in os.walk(os.path.join(base_dir, 'code')):
        for f_name in files:
            if f_name in all_code_files_names:
                code_file_paths[f_name] = os.path.join(root, f_name)

    class_info_objects = list()
    for f_name, f_path in tqdm(code_file_paths.items()):
        class_info = parse_java_file(f_path)
        class_info_objects.append(class_info)

    return class_info_objects


def add_method_calls(graph_nodes, cdg_file_path):
    cdg = json.load(open(cdg_file_path))
    present, absent = 0, 0
    for node in list(graph_nodes.values()):
        if node["type"] == "Method":
            method_key = node[METHOD_SIGNATURE_LABEL]
            if method_key in cdg:
                graph_nodes[method_key][CALLS_LABEL] = cdg[method_key]["calls"]
                graph_nodes[method_key][CALLED_BY_LABEL] = cdg[method_key]["called_by"]

                for node_call in cdg[method_key]["calls"] + cdg[method_key]["called_by"]:
                    if node_call not in cdg:
                        absent += 1
                        print(f"Node {node_call} not found")
                    else:
                        present += 1
                        cdg_call_node = cdg[node_call]
                        if node_call not in graph_nodes:
                            graph_nodes[node_call] = {
                                "type": "Method",
                                METHOD_SIGNATURE_LABEL: node_call,
                                CLASS_NAME_LABEL: cdg_call_node['class_name'],
                                METHOD_NAME_LABEL: cdg_call_node['method_name'],
                                FILE_NAME_LABEL: node[FILE_NAME_LABEL],
                            }
                        
                        graph_nodes[node_call][CALLED_BY_LABEL] = cdg_call_node["called_by"]
                        graph_nodes[node_call][CALLS_LABEL] = cdg_call_node["calls"]

    print(f"Present: {present}, Absent: {absent}")
    print("Number of nodes in the graph:", len(graph_nodes))


def get_signature_contents(signature):
    class_name = signature.split('.')[0]
    method_name = signature.split('.')[1].split('(')[0]
    params = signature\
        .replace(class_name, '')\
        .replace(method_name, '')\
        .replace('(', '')\
        .replace(')', '')\
        .replace('.', '')
    
    return {
        CLASS_NAME_LABEL: class_name,
        METHOD_NAME_LABEL: method_name,
        PARAMETERS: params
    }

def extract_call_graph_links(graph_nodes):
    call_graph_links = dict()
    for graph_node in graph_nodes.values():
        class_name = graph_node[CLASS_NAME_LABEL]
        if METHODS_LABEL not in graph_node:
            continue
        
        links = list()
        for method_node in graph_node[METHODS_LABEL]:
            src = get_signature_contents(method_node[METHOD_SIGNATURE_LABEL])
            
            calls = method_node[CALLS_LABEL] if CALLS_LABEL in method_node else []
            called_by = method_node[CALLED_BY_LABEL] if CALLED_BY_LABEL in method_node else []
            
            for call in calls + called_by:
                dest = get_signature_contents(call)
                links.append(
                    {
                        'src_signature': src,
                        'dest_signature': dest
                    }
                )
        
        call_graph_links[class_name] = links
        
    return call_graph_links



def add_methods_to_class_nodes(graph_nodes: dict):
    class_nodes = [node for node in graph_nodes.values() if node["type"] == "Class"]
    for class_node in class_nodes:
        class_name = class_node[CLASS_NAME_LABEL]
        class_node[METHODS_LABEL] = list()
        for method_node in graph_nodes.values():
            if method_node["type"] == "Method" and method_node[CLASS_NAME_LABEL] == class_name:
                class_node[METHODS_LABEL].append(method_node)

        graph_nodes[class_name] = class_node


def get_code_graph_nodes(
        base_dir: str,
        all_code_files_path: str,
        callgraph_file_name: str,
    ):
    print("Extracting class info objects")
    class_info_objects = get_class_info_objects(base_dir, all_code_files_path)
    graph_nodes = get_graph_nodes(class_info_objects)
    cdg_path = os.path.join(base_dir, callgraph_file_name)
    print("Extracting call graph links")
    add_method_calls(graph_nodes, cdg_path)
    print("Adding method calls")
    add_methods_to_class_nodes(graph_nodes)
    print("Done!")

    return graph_nodes


def get_code_nodes(
        base_dir: str,
        all_code_files_path: str,
    ):
    all_code_files_path = os.path.join(base_dir, all_code_files_path)
    all_code_files = [f_name.strip() for f_name in open(all_code_files_path)]

    nodes = [
        open(os.path.join(base_dir, 'code', f_name)).read()
        for f_name in tqdm(all_code_files)
    ]

    return nodes


if __name__ == '__main__':
    for dataset in ['iTrust', 'eTour', 'eANCI', 'smos']:
        print(f"Processing {dataset}")
        graph_nodes = get_code_graph_nodes(
            base_dir=f'data_repos/ftlr/datasets/{dataset}',
            all_code_files_path='all_code_filenames.txt',
            callgraph_path=f'{dataset.lower()}_method_callgraph.json',
        )
        print(f"Finished processing {dataset}")
