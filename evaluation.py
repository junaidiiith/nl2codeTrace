from collections import defaultdict
import json
from querying import query_parallel


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
    with open(result_file_name, 'w') as f:
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