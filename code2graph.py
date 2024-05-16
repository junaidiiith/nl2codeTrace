import json
import os
import javalang

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
PARAM_NAME_LABEL = "Parameter Name"
PARAM_TYPE_LABEL = "Parameter Type"
PARAM_DESCRIPTION_LABEL = "Description"
CALLS_LABEL = "calls"
CALLED_BY_LABEL = "called_by"


def parse_java_file(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        content = file.read()

    # Parse the file into an AST
    try:
        tree = javalang.parse.parse(content)
    except javalang.parser.JavaSyntaxError as e:
        # raise e
        return {
            FILE_NAME_LABEL: file_path.split('/')[-1],
            CLASS_NAME_LABEL: file_path.split('/')[-1].split('.')[0],
        }
        

    # Find the main class declaration
    class_decl = next(
        (type_decl for type_decl in tree.types \
         if isinstance(type_decl, javalang.tree.ClassDeclaration) or \
            isinstance(type_decl, javalang.tree.InterfaceDeclaration)), None
    )

    class_info = {
        CLASS_NAME_LABEL: class_decl.name,
        DOCSTRING_LABEL: class_decl.documentation if class_decl.documentation else "",
        ATTRIBUTES_LABEL: [],
        METHODS_LABEL: [],
        FILE_NAME_LABEL: file_path.split('/')[-1]
    }

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

    return class_info


def get_graph_nodes(class_info_objects):
    graph_nodes = dict()
    for class_info in class_info_objects:
        node = {
            CLASS_NAME_LABEL: class_info[CLASS_NAME_LABEL],
            FILE_NAME_LABEL: class_info[FILE_NAME_LABEL],
            "type": "Class"
        }
        if DOCSTRING_LABEL in class_info:
            node[DOCSTRING_LABEL] = class_info[DOCSTRING_LABEL]
        if ATTRIBUTES_LABEL in class_info:
            node[ATTRIBUTES_LABEL] = class_info[ATTRIBUTES_LABEL]
        graph_nodes[class_info[CLASS_NAME_LABEL]] = node

        class_name = class_info[CLASS_NAME_LABEL]
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
    all_code_files = [f_name.strip() for f_name in open(all_code_files_path)]

    class_info_objects = list()
    for f_name in tqdm(all_code_files):
        class_info = parse_java_file(os.path.join(base_dir, 'code', f_name))
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


def get_code_graph_nodes(
        base_dir: str,
        all_code_files_path: str = 'all_code_filenames.txt',
        callgraph_path: str = 'etour_method_callgraph.json'
    ):
    class_info_objects = get_class_info_objects(base_dir, all_code_files_path)
    graph_nodes = get_graph_nodes(class_info_objects)
    cdg_path = os.path.join(base_dir, callgraph_path)
    add_method_calls(graph_nodes, cdg_path)
    return graph_nodes