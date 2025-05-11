import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *

FAIL_MSG = 'Failed to obtain answer via API.'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, index_set=None, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'

    # To reuse records in MMBench_V11
    if dataset_name in ['MMBench', 'MMBench_CN']:
        v11_pred = f'{work_dir}/{model_name}_{dataset_name}_V11.xlsx'
        if osp.exists(v11_pred):
            try:
                reuse_inds = load('http://opencompass.openxlab.space/utils/mmb_reuse.pkl')
                data = load(v11_pred)
                ans_map = {x: y for x, y in zip(data['index'], data['prediction']) if x in reuse_inds}
                dump(ans_map, out_file)
            except Exception as err:
                print(type(err), err)

    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if ignore_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    if index_set is not None:
        res = {k: v for k, v in res.items() if k in index_set}
    os.remove(out_file)
    return res


def infer_data(model, model_name, work_dir, support_dataset, query_dataset, out_file, verbose=False, api_nproc=4, use_vllm=False, rag_method=None, num_shots=0, previous_query_data_cot=None):
    rank, world_size = get_rank_and_world_size()
    query_dataset_name = query_dataset.dataset_name
    if support_dataset is None:
        prev_file = f'{work_dir}/{model_name}_{query_dataset_name}_PREV.pkl'
    else:
        support_dataset_name = support_dataset.dataset_name
        prev_file = f'{work_dir}/{model_name}_{support_dataset_name}_{query_dataset_name}_{rag_method}_{num_shots}_PREV.pkl'
        support_sheet_indices = list(range(rank, len(support_dataset), world_size))
        support_lt = len(support_sheet_indices)
        support_data = support_dataset.data.iloc[support_sheet_indices]
        support_data_indices = [i for i in support_data['index']]

    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    query_sheet_indices = list(range(rank, len(query_dataset), world_size))
    query_lt = len(query_sheet_indices)
    query_data = query_dataset.data.iloc[query_sheet_indices]
    query_data_indices = [i for i in query_data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(query_lt):
        idx = query_data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in query_data_indices}
        dump(res, out_file)
        return

    # Data need to be inferred
    query_data = query_data[~query_data['index'].isin(res)]
    query_lt = len(query_data)

    kwargs = {}
    if model_name is not None and 'Llama-4' in model_name:
        kwargs = {'use_vllm': use_vllm}
    model = supported_VLM[model_name](**kwargs) if isinstance(model, str) else model
    print("model: ", model_name, " | post_process: ", model.post_process)

    is_api = getattr(model, 'is_api', False)
    if is_api:
        query_lt, query_indices = len(query_data), list(query_data['index'])
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=query_dataset,  # TODO: add support_dataset
            index_set=set(query_indices),
            api_nproc=api_nproc)
        for idx in query_indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in query_data_indices}
        dump(res, out_file)
        return model
    else:
        model.set_dump_image(query_dataset.dump_image)
    
    if support_dataset is not None:
        if rag_method == 'jices':
            from vlmeval.demo_select.jices import JICES
            cached_features_path = f"cache/{model_name}/{rag_method}/support/{support_dataset_name}.pkl"
            query_cached_features_path = f"cache/{model_name}/{rag_method}/query/{query_dataset_name}.pkl"

            if previous_query_data_cot is not None:
                use_ans_feat = True
            else:
                use_ans_feat = False
            
            retriever = JICES(
                dataset=support_dataset,
                query_dataset=query_dataset,
                eval_model=model,
                device=model.model.device,
                batch_size=8,
                cached_features_path=cached_features_path,
                query_cached_features_path=query_cached_features_path,
                use_ans_feat=use_ans_feat,
            )
        else:
            retriever = None

    for i in tqdm(range(query_lt)):
        if i > 2: # stop the verbose after 2 queries
            model.verbose = False

        idx = query_data.iloc[i]['index']
        if idx in res:
            continue

        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(query_dataset_name):
            struct = model.build_prompt(query_data.iloc[i], dataset=query_dataset_name)
        else:
            if previous_query_data_cot is None:
                struct = query_dataset.build_prompt(query_data.iloc[i])
            else:
                previous_cot = ''
                for d in previous_query_data_cot:
                    previous_cot += d[i] + '\n'
                struct = query_dataset.build_prompt(query_data.iloc[i], previous_cot=previous_cot)
        
        demo_msgs = []
        if support_dataset is not None:
            if rag_method == 'random':
                random_support_id = np.random.choice(len(support_data), num_shots, replace=False)
                if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(support_dataset_name):
                    for id in random_support_id:
                        demo_msgs += model.build_prompt(support_data.iloc[id], dataset=support_dataset_name)
                else:
                    for id in random_support_id:
                        demo_msgs += support_dataset.build_prompt(support_data.iloc[id], use_answer=True)
            elif retriever is not None:
                for demo in retriever.find(idx, num_shots):
                    demo_msgs += demo
        
        # print(demo_msgs+struct)

        response = model.generate(message=demo_msgs+struct, dataset=query_dataset_name)
        torch.cuda.empty_cache()

        # if verbose:
        #     print(response, flush=True)

        res[idx] = response
        if (i + 1) % 10 == 0:
            dump(res, out_file)

    res = {k: res[k] for k in query_data_indices}
    dump(res, out_file)
    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(
    model, work_dir, model_name, support_dataset, query_dataset, verbose=False, api_nproc=4, ignore_failed=False, use_vllm=False, rag_method=None, num_shots=0, previous_query_data_cot=None
):
    rank, world_size = get_rank_and_world_size()
    query_dataset_name = query_dataset.dataset_name
    if support_dataset is None:
        result_file = osp.join(work_dir, f'{model_name}_{query_dataset_name}.xlsx')
        prev_file = f'{work_dir}/{model_name}_{query_dataset_name}_PREV.pkl'
        tmpl = osp.join(work_dir, '{}' + f'{world_size}_{model_name}_{query_dataset_name}.pkl')
    else:
        support_dataset_name = support_dataset.dataset_name
        result_file = osp.join(work_dir, f'{model_name}_{support_dataset_name}_{query_dataset_name}_{rag_method}_{num_shots}.xlsx')
        prev_file = f'{work_dir}/{model_name}_{support_dataset_name}_{query_dataset_name}_{rag_method}_{num_shots}_PREV.pkl'
        tmpl = osp.join(work_dir, '{}' + f'{world_size}_{model_name}_{support_dataset_name}_{query_dataset_name}_{rag_method}_{num_shots}.pkl')
    
    if osp.exists(result_file):
        if world_size > 1:
            dist.barrier()
        
        if rank == 0:
            data = load(result_file)

            if 'prediction' in data.columns and 'rationale' not in data.columns:
                # change the column name "prediction" to "rationale"
                data.rename(columns={'prediction': 'rationale'}, inplace=True)
                import re
                def extract_answer_content(output_str):
                    # Try to find the content within <answer> tags, if can not find, return None
                    answer_pattern = r"<CONCLUSION>\s*(.*?)\s*<\/CONCLUSION>"
                    match = re.search(answer_pattern, output_str, re.DOTALL)

                    if match:
                        return match.group(1).strip()
                    return output_str

                def replace_last_dot(input_string):
                    if input_string.endswith("."):
                        return input_string[:-1]
                    else:
                        return input_string
                    
                data['prediction'] = data['rationale'].apply(lambda x: replace_last_dot(extract_answer_content(x)))
                dump(data, result_file)

            if 'rationale' in data.columns:
                results = {k: {'prediction': v, 'rationale': q} for k, v, q in zip(data['index'], data['prediction'], data['rationale'])}
                if not ignore_failed:
                    results = {k: v for k, v in results.items() if FAIL_MSG not in str(v['prediction'])}
            else:
                results = {k: v for k, v in zip(data['index'], data['prediction'])}
                if not ignore_failed:
                    results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()
    
    out_file = tmpl.format(rank)

    model = infer_data(
        model=model, work_dir=work_dir, model_name=model_name, support_dataset=support_dataset, query_dataset=query_dataset,
        out_file=out_file, verbose=verbose, api_nproc=api_nproc, use_vllm=use_vllm, rag_method=rag_method, num_shots=num_shots, previous_query_data_cot=previous_query_data_cot)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = query_dataset.data
        for x in data['index']:
            assert x in data_all
        
        if isinstance(data_all[data['index'][0]], dict):
            data['prediction'] = [str(data_all[x]['prediction']) for x in data['index']]
            data['rationale'] = [str(data_all[x]['rationale']) for x in data['index']]
        else:
            data['prediction'] = [str(data_all[x]) for x in data['index']]
        
        # ??????????
        # if 'image' in data:
        #     data.pop('image')

        dump(data, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    if world_size > 1:
        dist.barrier()
    return model
