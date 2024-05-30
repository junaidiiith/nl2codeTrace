import pickle
from api_models import get_api_keys, set_llm_and_embed
from constants import EmbeddingModelsMap, LLMsMap
from parameters import parse_args
from req2nodes import get_requirements_nodes
from run import get_code_nodes
from indexing.utils import create_semantically_similar_nodes


def generate_similar_nodes(args, dataset_name):
    base_dir = args.base_dir
    dataset_dir = f'{base_dir}/{dataset_name}'

    req_nodes = get_requirements_nodes(
        dataset_dir,
        all_req_files_path=args.all_req_filenames,
    )
    req_nodes = create_semantically_similar_nodes(
        req_nodes,
        args.num_similar_nodes
    )


    with open(f'similar_requirements/{dataset_name}.pkl', 'wb') as f:
        pickle.dump(req_nodes, f)


def main():
    configs = [
        ('iTrust', 'bge_large'),
        ('smos', 'bge_m3'),
        ('eANCI', 'bge_m3'),
        ('eTour', 'bge_large')
    ]
    args = parse_args()

    for i, (dataset_name, embed_model) in enumerate(configs):
        api_key = get_api_keys(llm_type=args.llm_type, idx=i)
        args.embed_model = embed_model
        llm_name = LLMsMap[args.llm]
        embed_model_name = EmbeddingModelsMap[embed_model]
        print(f"Using LLM: {llm_name}")
        print(f"Using Embedding Model: {embed_model_name}")

        set_llm_and_embed(
            llm_type=args.llm_type,
            llm_name=llm_name,
            embed_model_name=embed_model_name,
            api_key=api_key
        )

        generate_similar_nodes(args, dataset_name)

if __name__ == '__main__':
    main()