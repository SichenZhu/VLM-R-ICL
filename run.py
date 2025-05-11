import json
import re

import torch
import torch.distributed as dist

from vlmeval.config import supported_VLM
from vlmeval.dataset.video_dataset_config import supported_video_datasets
from vlmeval.dataset import build_dataset
from vlmeval.inference import infer_data_job
from vlmeval.inference_video import infer_data_job_video
from vlmeval.inference_mt import infer_data_job_mt
from vlmeval.smp import *
from vlmeval.utils.result_transfer import MMMU_result_transfer, MMTBench_result_transfer


def build_model_from_config(cfg, model_name):
    import vlmeval.api
    import vlmeval.vlm
    config = cp.deepcopy(cfg[model_name])
    if config == {}:
        return supported_VLM[model_name]()
    assert 'class' in config
    cls_name = config.pop('class')
    if hasattr(vlmeval.api, cls_name):
        return getattr(vlmeval.api, cls_name)(**config)
    elif hasattr(vlmeval.vlm, cls_name):
        return getattr(vlmeval.vlm, cls_name)(**config)
    else:
        raise ValueError(f'Class {cls_name} is not supported in `vlmeval.api` or `vlmeval.vlm`')


def build_dataset_from_config(cfg, dataset_name):
    import vlmeval.dataset
    import inspect
    config = cp.deepcopy(cfg[dataset_name])
    if config == {}:
        return supported_video_datasets[dataset_name]()
    assert 'class' in config
    cls_name = config.pop('class')
    if hasattr(vlmeval.dataset, cls_name):
        cls = getattr(vlmeval.dataset, cls_name)
        sig = inspect.signature(cls.__init__)
        valid_params = {k: v for k, v in config.items() if k in sig.parameters}
        if cls.MODALITY == 'VIDEO':
            if valid_params.get('fps', 0) > 0 and valid_params.get('nframe', 0) > 0:
                raise ValueError('fps and nframe should not be set at the same time')
            if valid_params.get('fps', 0) <= 0 and valid_params.get('nframe', 0) <= 0:
                raise ValueError('fps and nframe should be set at least one valid value')
        return cls(**valid_params)
    else:
        raise ValueError(f'Class {cls_name} is not supported in `vlmeval.dataset`')


def parse_args():
    help_msg = """\
You can launch the evaluation by setting either --data and --model or --config.

--data and --model:
    Each Arg should be a list of strings, specifying the names of datasets and models.
    To find all supported model names, please refer to the `vlmeval/config.py` of check the output of the command \
        `vlmutil mlist all` in the terminal (you should first have vlmeval installed).
    To find all supported dataset names, please refer to the `vlmeval/dataset/__init__.py` file. The python script \
        to print all supported dataset names is as follows:
        ```python
        from vlmeval.dataset import SUPPORTED_DATASETS
        print(SUPPORTED_DATASETS)
        ```
        or you can check the output of the command `vlmutil dlist all` in the terminal.
    To find all supported video dataset default settings, please refer to the \
        `vlmeval/dataset/video_dataset_config.py` file.

--config:
    Launch the evaluation by specifying the path to the config json file. Sample Json Content:
    ```json
    {
        "model": {
            "GPT4o_20240806_T00_HIGH": {
                "class": "GPT4V",
                "model": "gpt-4o-2024-08-06",
                "temperature": 0,
                "img_detail": "high"
            },
            "GPT4o_20240806_T10_Low": {
                "class": "GPT4V",
                "model": "gpt-4o-2024-08-06",
                "temperature": 1.0,
                "img_detail": "low"
            },
            "GPT4o_20241120": {}
        },
        "data": {
            "MME-RealWorld-Lite": {
                "class": "MMERealWorld",
                "dataset": "MME-RealWorld-Lite"
            },
            "MMBench_DEV_EN_V11": {
                "class": "ImageMCQDataset",
                "dataset": "MMBench_DEV_EN_V11"
            },
            "MMBench_Video_8frame_nopack": {},
            "Video-MME_16frame_subs": {
                "class": "VideoMME",
                "dataset": "Video-MME",
                "nframe": 16,
                "use_subtitle": true,
            }
        }
    }
    ```
    Currently, only `model` and `data` are supported fields. The content of each field is a dictionary.
    For `model`, the key is the name of the model, and the value is a dictionary containing the following keys:
    - `class`: The class name of the model, which should be a class in `vlmeval.vlm` or `vlmeval.api`.
    - Other keys are specific to the model, please refer to the corresponding class.
    - Tip: The defined model in the `supported_VLM` of `vlmeval/config.py` can be used as a shortcut.
    For `data`, the key is the name of the dataset (should be the same as the `dataset` field in most cases, \
        except for video datasets), and the value is a dictionary containing the following keys:
    - `class`: The class name of the dataset, which should be a class in `vlmeval.dataset`.
    - `dataset`: The name of the dataset, which should be a string that is accepted by the `dataset` argument of the \
        corresponding class.
    - Other keys are specific to the dataset, please refer to the corresponding class.
    - Tip: The defined dataset in the `supported_video_datasets` of `vlmeval/dataset/video_dataset_config.py` \
        can be used as a shortcut.

    The keys in the `model` and `data` fields will be used for naming the prediction files and evaluation results.
    When launching with `--config`, args for API VLMs, such as `--retry`, `--verbose`, will be ignored.
"""
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawTextHelpFormatter)
    # Essential Args, Setting the Names of Datasets and Models
    parser.add_argument('--query_data', type=str, nargs='+', help='Names of Query Datasets')
    parser.add_argument('--support_data', type=str, nargs='+', help='Names of Support Datasets')
    parser.add_argument('--num_shots', type=int, nargs='+', default=0, help='Number of shots for few-shot learning')
    parser.add_argument('--rag_method', type=str, default='none', help='RAG method for few-shot learning')
    parser.add_argument('--icl_rationale', action='store_true', help='Use ICL rationale for few-shot learning')
    parser.add_argument('--multi_step_icl', action='store_true', help='Use multi-step ICL for few-shot learning')

    parser.add_argument('--model', type=str, nargs='+', help='Names of Models')
    parser.add_argument('--config', type=str, help='Path to the Config Json File')
    # Work Dir
    parser.add_argument('--work-dir', type=str, default='./outputs', help='select the output directory')
    # Infer + Eval or Infer Only
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer'])
    # API Kwargs, Apply to API VLMs and Judge API LLMs
    parser.add_argument('--api-nproc', type=int, default=4, help='Parallel API calling')
    parser.add_argument('--retry', type=int, default=None, help='retry numbers for API VLMs')
    parser.add_argument('--judge-args', type=str, default=None, help='Judge arguments in JSON format')
    # Explicitly Set the Judge Model
    parser.add_argument('--judge', type=str, default=None)
    # Logging Utils
    parser.add_argument('--verbose', action='store_true')
    # Configuration for Resume
    # Ignore: will not rerun failed VLM inference
    parser.add_argument('--ignore', action='store_true', help='Ignore failed indices. ')
    # Reuse: will reuse the existing prediction files
    parser.add_argument('--reuse', action='store_true')
    # Reuse-aux: if set, when reuse is True, will also reuse the auxiliary evaluation files
    parser.add_argument('--reuse-aux', type=bool, default=True, help='reuse auxiliary evaluation files')
    parser.add_argument(
        '--use-vllm', action='store_true', help='use vllm to generate, the flag is only supported in Llama4 for now')

    args = parser.parse_args()
    return args


def main_build_dataset(model_name, query_dataset_name, use_config, cfg, rank, world_size, logger):
    if use_config:
        if world_size > 1:
            if rank == 0:
                query_dataset = build_dataset_from_config(cfg['query_data'], query_dataset_name)
            dist.barrier()
        query_dataset = build_dataset_from_config(cfg['query_data'], query_dataset_name)
        if query_dataset is None:
            return query_dataset
    else:
        query_dataset_kwargs = {}
        if query_dataset_name in ['MMLongBench_DOC', 'DUDE', 'DUDE_MINI', 'SLIDEVQA', 'SLIDEVQA_MINI']:
            query_dataset_kwargs['model'] = model_name

        # If distributed, first build the dataset on the main process for doing preparation works
        if world_size > 1:
            if rank == 0:
                query_dataset = build_dataset(query_dataset_name, **query_dataset_kwargs)
            dist.barrier()

        query_dataset = build_dataset(query_dataset_name, **query_dataset_kwargs)
        if query_dataset is None:
            return query_dataset
    
    return query_dataset


def main_inference(model, model_name, support_dataset, query_dataset, support_dataset_name, query_dataset_name, 
                   prev_pred_roots, pred_root, pred_root_meta, result_file, result_file_base, args, rank, world_size, logger, shot, commit_id, previous_query_data_cot=None):
    # Reuse the previous prediction file if exists
    if rank == 0 and len(prev_pred_roots):
        prev_result_files = []
        prev_pkl_file_list = []
        for root in prev_pred_roots:  # [::-1]
            if osp.exists(osp.join(root, result_file_base)):
                if args.reuse_aux:
                    prev_result_files = fetch_aux_files(osp.join(root, result_file_base))
                else:
                    prev_result_files = [osp.join(root, result_file_base)]
                break
            elif commit_id in root and len(ls(root)) and root != pred_root:
                temp_files = ls(root, match=[query_dataset_name, '.pkl'])  # TODO
                if len(temp_files):
                    prev_pkl_file_list.extend(temp_files)
                    break
        if not args.reuse:
            prev_result_files = []
            prev_pkl_file_list = []
        if len(prev_result_files):
            for prev_result_file in prev_result_files:
                src = prev_result_file
                tgt = osp.join(pred_root, osp.basename(src))
                if not osp.exists(tgt):
                    shutil.copy(src, tgt)
                    logger.info(f'--reuse is set, will reuse the prediction file {src}.')
                else:
                    logger.warning(f'File already exists: {tgt}')

        elif len(prev_pkl_file_list):
            for fname in prev_pkl_file_list:
                target_path = osp.join(pred_root, osp.basename(fname))
                if not osp.exists(target_path):
                    shutil.copy(fname, target_path)
                    logger.info(f'--reuse is set, will reuse the prediction pickle file {fname}.')
                else:
                    logger.warning(f'File already exists: {target_path}')

    if world_size > 1:
        dist.barrier()

    if model is None:
        model = model_name  # which is only a name

    # Perform the Inference
    if query_dataset.MODALITY == 'VIDEO':
        model = infer_data_job_video(
            model,
            work_dir=pred_root,
            model_name=model_name,
            dataset=query_dataset,
            result_file_name=result_file_base,
            verbose=args.verbose,
            api_nproc=args.api_nproc,
            use_vllm=args.use_vllm)
    elif query_dataset.TYPE == 'MT':
        model = infer_data_job_mt(
            model,
            work_dir=pred_root,
            model_name=model_name,
            dataset=query_dataset,
            verbose=args.verbose,
            api_nproc=args.api_nproc,
            ignore_failed=args.ignore,
            use_vllm=args.use_vllm)
    else:
        model = infer_data_job(
            model,
            work_dir=pred_root,
            model_name=model_name,
            support_dataset=support_dataset,
            query_dataset=query_dataset,
            verbose=args.verbose,
            api_nproc=args.api_nproc,
            ignore_failed=args.ignore,
            use_vllm=args.use_vllm,
            rag_method=args.rag_method,
            num_shots=shot,
            previous_query_data_cot=previous_query_data_cot)

    # Set the judge kwargs first before evaluation or dumping

    judge_kwargs = {
        'nproc': args.api_nproc,
        'verbose': args.verbose,
        'retry': args.retry if args.retry is not None else 3,
        **(json.loads(args.judge_args) if args.judge_args else {}),
    }

    if args.retry is not None:
        judge_kwargs['retry'] = args.retry
    if args.judge is not None:
        judge_kwargs['model'] = args.judge
    else:
        if query_dataset.TYPE in ['MCQ', 'Y/N', 'MCQ_MMMU_Pro'] or listinstr(
            ['moviechat1k'], query_dataset_name.lower()
        ):
            if listinstr(['WeMath'], query_dataset_name):
                judge_kwargs['model'] = 'gpt-4o-mini'
            else:
                judge_kwargs['model'] = 'chatgpt-0125'
        elif listinstr(['MMVet', 'LLaVABench', 'MMBench_Video'], query_dataset_name):
            judge_kwargs['model'] = 'gpt-4-turbo'
        elif listinstr(['MathVista', 'MathVerse', 'MathVision', 'DynaMath', 'VL-RewardBench', 'LogicVista', 'MOAT', 'MME_CoT'], query_dataset_name):  # noqa: E501
            judge_kwargs['model'] = 'gpt-4o-mini'
        elif listinstr(['MMLongBench', 'MMDU', 'DUDE', 'SLIDEVQA', 'MIA-Bench', 'WildVision', 'MMAlignBench'], query_dataset_name):  # noqa: E501
            judge_kwargs['model'] = 'gpt-4o'
        elif listinstr(['VDC'], query_dataset_name):
            judge_kwargs['model'] = 'llama31-8b'
        elif listinstr(['VideoMMLU_QA', 'VideoMMLU_CAP'], query_dataset_name):
            judge_kwargs['model'] = 'qwen-72b'

    if rank == 0:
        logger.info(judge_kwargs)

    if world_size > 1:
        dist.barrier()

    # Only Rank 0 handles the evaluation part
    if rank == 0:
        # Prepare Submission Files for MMMU_TEST AND MMT-Bench_ALL
        if query_dataset_name in ['MMMU_TEST']:
            result_json = MMMU_result_transfer(result_file)
            logger.info(f'Transfer MMMU_TEST result to json for official evaluation, '
                        f'json file saved in {result_json}')
            return False
        elif 'MMT-Bench_ALL' in query_dataset_name:
            submission_file = MMTBench_result_transfer(result_file, **judge_kwargs)
            logger.info(f'Extract options from prediction of MMT-Bench FULL split for official evaluation '
                        f'(https://eval.ai/web/challenges/challenge-page/2328/overview), '
                        f'submission file saved in {submission_file}')
            return False

        # Skip the evaluation part if only infer
        if args.mode == 'infer':
            logger.info('The evaluation is skipped since --mode is set to infer. ')
            return False

        # Skip the evaluation part if the dataset evaluation is not supported or annotations are missing
        if 'MLLMGuard_DS' in query_dataset_name:
            logger.info('The evaluation of MLLMGuard_DS is not supported yet. ')
            return False
        elif 'AesBench_TEST' == query_dataset_name:
            logger.info(f'The results are saved in {result_file}. '
                        f'Please send it to the AesBench Team via huangyipo@hotmail.com.')
            return False
        elif query_dataset_name in ['DocVQA_TEST', 'InfoVQA_TEST', 'Q-Bench1_TEST', 'A-Bench_TEST']:
            logger.info(f'{query_dataset_name} is a test split without ground-truth. '
                        'Thus only the inference part is supported for those datasets. ')
            return False
        elif query_dataset_name in [
            'MMBench_TEST_CN', 'MMBench_TEST_EN', 'MMBench', 'MMBench_CN',
            'MMBench_TEST_CN_V11', 'MMBench_TEST_EN_V11', 'MMBench_V11', 'MMBench_CN_V11'
        ] and not MMBenchOfficialServer(query_dataset_name):
            logger.error(
                f'Can not evaluate {query_dataset_name} on non-official servers, will skip the evaluation.')
            return False

        # Setup the proxy for the evaluation
        eval_proxy = os.environ.get('EVAL_PROXY', None)
        old_proxy = os.environ.get('HTTP_PROXY', '')
        if eval_proxy is not None:
            proxy_set(eval_proxy)

        # Perform the Evaluation
        eval_results = query_dataset.evaluate(result_file, **judge_kwargs)
        # Display Evaluation Results in Terminal
        if eval_results is not None:
            assert isinstance(eval_results, dict) or isinstance(eval_results, pd.DataFrame)
            if support_dataset is not None:
                logger.info(f'The evaluation of model {model_name} x support dataset {support_dataset_name} x query dataset {query_dataset_name} has finished! ')
            else:
                logger.info(f'The evaluation of model {model_name} x dataset {query_dataset_name} has finished! ')
            logger.info('Evaluation Results:')
            if isinstance(eval_results, dict):
                logger.info('\n' + json.dumps(eval_results, indent=4))
            elif isinstance(eval_results, pd.DataFrame):
                if len(eval_results) < len(eval_results.columns):
                    eval_results = eval_results.T
                logger.info('\n' + tabulate(eval_results))

        # Restore the proxy
        if eval_proxy is not None:
            proxy_set(old_proxy)

        # Create the symbolic links for the prediction files
        files = os.listdir(pred_root)
        if support_dataset is not None:
            files = [x for x in files if (f'{model_name}_{support_dataset_name}_{query_dataset_name}_{args.rag_method}_{shot}' in x or "status.json" in x)]
        else:
            files = [x for x in files if (f'{model_name}_{query_dataset_name}' in x or "status.json" in x)]
        for f in files:
            cwd = os.getcwd()
            file_addr = osp.join(cwd, pred_root, f)
            link_addr = osp.join(cwd, pred_root_meta, f)
            if osp.exists(link_addr) or osp.islink(link_addr):
                os.remove(link_addr)
            os.symlink(file_addr, link_addr)
    
    return True


def main():
    logger = get_logger('RUN')
    rank, world_size = get_rank_and_world_size()
    args = parse_args()
    use_config, cfg = False, None
    if args.config is not None:
        assert args.support_data is None, '--support_data should not be set when using --config'
        assert args.query_data is None, '--query_data should not be set when using --config'
        assert args.model is not None, '--model should be set when using --config'
        use_config, cfg = True, load(args.config)
        args.model = list(cfg['model'].keys())
        args.support_data = list(cfg['support_data'].keys())
        args.query_data = list(cfg['query_data'].keys())
    else:
        assert len(args.support_data), '--support_data should be a list of data files'
        assert len(args.query_data), '--query_data should be a list of data files'

    if rank == 0:
        if not args.reuse:
            logger.warning('--reuse is not set, will not reuse previous (before one day) temporary files')
        else:
            logger.warning('--reuse is set, will reuse the latest prediction & temporary pickle files')

    if 'MMEVAL_ROOT' in os.environ:
        args.work_dir = os.environ['MMEVAL_ROOT']

    if not use_config:
        for k, v in supported_VLM.items():
            if hasattr(v, 'keywords') and 'retry' in v.keywords and args.retry is not None:
                v.keywords['retry'] = args.retry
                supported_VLM[k] = v
            if hasattr(v, 'keywords') and 'verbose' in v.keywords and args.verbose is not None:
                v.keywords['verbose'] = args.verbose
                supported_VLM[k] = v

    if world_size > 1:
        local_rank = os.environ.get('LOCAL_RANK', 0)
        print("world_size: ", world_size, ", local_rank: ", local_rank)
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group(
            backend='nccl',
            timeout=datetime.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 3600))),
            device_id=torch.device(f"cuda:{local_rank}"),
            world_size=world_size,
            rank=rank,
        )
    
    for _, model_name in enumerate(args.model):
        model = None
        date, commit_id = timestr('day'), githash(digits=8)
        eval_id = f"T{date}_G{commit_id}"

        pred_root = osp.join(args.work_dir, model_name, eval_id)
        pred_root_meta = osp.join(args.work_dir, model_name)
        os.makedirs(pred_root_meta, exist_ok=True)

        prev_pred_roots = ls(osp.join(args.work_dir, model_name), mode='dir')
        if len(prev_pred_roots) and args.reuse:
            prev_pred_roots.sort()

        if not osp.exists(pred_root):
            os.makedirs(pred_root, exist_ok=True)

        if use_config:
            model = build_model_from_config(cfg['model'], model_name)

        for _, query_dataset_name in enumerate(args.query_data):
            for _, support_dataset_name in enumerate(args.support_data):
                if world_size > 1:
                    dist.barrier()
                
                correct = False
                if '_correct' in support_dataset_name:
                    correct = True
                    support_dataset_name = support_dataset_name.replace('_correct', '')

                try:
                    support_dataset = main_build_dataset(model_name, support_dataset_name, use_config, cfg, rank, world_size, logger)
                    if support_dataset is None:
                        logger.error(f'Support Dataset {support_dataset_name} is not valid, will be skipped. ')
                        continue
                    query_dataset = main_build_dataset(model_name, query_dataset_name, use_config, cfg, rank, world_size, logger)
                    if query_dataset is None:
                        logger.error(f'Query Dataset {query_dataset_name} is not valid, will be skipped. ')
                        continue

                    if args.icl_rationale:
                        support_result_file_base = f'{model_name}_{support_dataset_name}.xlsx'
                        # Handling Multi-Turn Dataset
                        if support_dataset.TYPE == 'MT':
                            support_result_file_base = support_result_file_base.replace('.xlsx', '.tsv')

                        support_result_file = osp.join(pred_root, support_result_file_base)
                        
                        if not osp.exists(support_result_file):
                            main_inference(
                                model,
                                model_name,
                                None,
                                support_dataset,
                                None,
                                support_dataset_name,
                                prev_pred_roots,
                                pred_root,
                                pred_root_meta,
                                support_result_file,
                                support_result_file_base,
                                args,
                                rank,
                                world_size,
                                logger,
                                0,  # shot
                                commit_id
                            )

                        support_dataset_name_original = support_dataset_name
                        support_dataset_name = f'{model_name}_{support_dataset_name}_rationale_all'
                        if rank == 0: 
                            # correction
                            original_data = load(osp.join(LMUDataRoot(), support_dataset_name_original.split('_QCME')[0] + '.tsv'))
                            possible_result_files = support_result_file.replace('.xlsx', '_openai_result.xlsx')
                            
                            if osp.exists(possible_result_files) and correct:
                                print("possible_result_files = ", possible_result_files)
                                updated_data = load(possible_result_files)
                            else:
                                updated_data = load(support_result_file)
                            
                            if 'image_path' in original_data:
                                original_data = original_data[~pd.isna(original_data['image_path'])]
                                original_data = original_data.sort_values(by='index')
                                updated_data = updated_data.sort_values(by='index')
                                assert updated_data['index'].tolist() == original_data['index'].tolist(), f"updated_data['index'] = {updated_data['index']}, original_data['index'] = {original_data['index']}"
                                # updated_data['index'] = original_data['index']
                                # updated_data['image_path'] = original_data['image_path']
                                if 'image_path' in updated_data:
                                    assert updated_data['image_path'].tolist() == original_data['image_path'].tolist(), f"updated_data['image_path'] = {updated_data['image_path']}, original_data['image_path'] = {original_data['image_path']}"
                                else:
                                    updated_data['image_path'] = original_data['image_path']

                            elif 'image' in original_data:
                                original_data = original_data[~pd.isna(original_data['image'])]
                                original_data = original_data.sort_values(by='index')
                                updated_data = updated_data.sort_values(by='index')
                                assert updated_data['index'].tolist() == original_data['index'].tolist(), f"updated_data['index'] = {updated_data['index']}, original_data['index'] = {original_data['index']}"
                                # updated_data['index'] = original_data['index']
                                # updated_data['image'] = original_data['image']
                                if 'image' in updated_data:
                                    assert updated_data['image'].tolist() == original_data['image'].tolist(), f"updated_data['image'] = {updated_data['image']}, original_data['image'] = {original_data['image']}"
                                else:
                                    updated_data['image'] = original_data['image']

                            else:
                                raise ValueError('No image_path or image found in the original data.')
                            
                        if world_size > 1:
                            dist.barrier()
                            
                        if correct:
                            # only keep the correct samples
                            support_dataset_name += '_correct'
                            if rank == 0:
                                print("Original support dataset size: ", len(updated_data))
                                if 'hit' in updated_data:
                                    # change the hit type to int
                                    updated_data['hit'] = updated_data['hit'].astype(int)
                                    updated_data = updated_data[updated_data['hit'] == 1]
                                elif 'match' in updated_data:
                                    # updated_data['match'] is a list of numbers
                                    # change match type from string to list, then get min(1, sum(match)/3)
                                    updated_data['match'] = updated_data['match'].apply(lambda x: eval(x))
                                    updated_data['match'] = updated_data['match'].apply(lambda x: min(1, sum(x) / 3))
                                    updated_data = updated_data[updated_data['match'] > 0.5]
                                else:
                                    raise ValueError('No hit or match found in the updated data.')
                                print("Corrected support dataset size: ", len(updated_data))

                        if world_size > 1:
                            dist.barrier()
                             
                        if rank == 0:
                            dump(updated_data, osp.join(LMUDataRoot(), support_dataset_name + '.tsv'))
                            
                        # important
                        if world_size > 1:
                            dist.barrier()
                        # change one value from the parent class of support_dataset
                        support_dataset.__class__.DATASET_URL[support_dataset_name] = osp.join(LMUDataRoot(), support_dataset_name + '.tsv')
                        support_dataset = main_build_dataset(model_name, support_dataset_name, use_config, cfg, rank, world_size, logger)
                    
                    if support_dataset is None:
                        logger.error(f'Support Dataset {support_dataset_name} is not valid, will be skipped. ')
                        continue

                    tags = [None]
                    if args.multi_step_icl:
                        if 'VLM-R1' in model_name:
                            tags = ['<think>', '<answer>']
                        elif 'LLaVA-CoT' in model_name:
                            tags = ['<SUMMARY>', '<CAPTION>', '<REASONING>', '<CONCLUSION>']
                    
                    # dictionary to store query cot results for each stage and each shot
                    query_cot_results = {}
                    final_support_dataset_name = support_dataset_name

                    for stage in range(len(tags)):
                        query_cot_results[stage] = {}
                        if len(tags) > 1:
                            tag = tags[stage]
                            start_tag = tags[0]
                            end_tag = tag.replace('<', '</')
                            support_dataset_name = final_support_dataset_name.replace('_all', f'_{tag}')
                            if rank == 0:
                                data_stage = updated_data.copy()
                                # only keep the rationale within the current and previous tag
                                data_stage['rationale'] = data_stage['rationale'].apply(
                                    lambda x: start_tag + x.split(start_tag)[-1].split(end_tag)[0] + end_tag
                                )
                                dump(data_stage, osp.join(LMUDataRoot(), support_dataset_name + '.tsv'))
                            
                            if world_size > 1:
                                dist.barrier()

                            # change one value from the parent class of support_dataset
                            support_dataset.__class__.DATASET_URL[support_dataset_name] = osp.join(LMUDataRoot(), support_dataset_name + '.tsv')
                            support_dataset = main_build_dataset(model_name, support_dataset_name, use_config, cfg, rank, world_size, logger)
                        
                        if stage < len(tags) - 1:
                            args.mode == 'infer'
                        else:
                            args.mode = 'all'

                        # previous_query_data_cot are from previous stages

                        for shot in args.num_shots:
                            result_file_base = f'{model_name}_{support_dataset_name}_{query_dataset_name}_{args.rag_method}_{shot}.xlsx'
                            # Handling Multi-Turn Dataset
                            if query_dataset.TYPE == 'MT':
                                result_file_base = result_file_base.replace('.xlsx', '.tsv')

                            result_file = osp.join(pred_root, result_file_base)

                            if stage == 0:
                                previous_query_data_cot = None
                            else:
                                previous_query_data_cot = [query_cot_results[prev_stage][shot] for prev_stage in range(stage)]

                            status = main_inference(
                                model,
                                model_name,
                                support_dataset,
                                query_dataset,
                                support_dataset_name,
                                query_dataset_name,
                                prev_pred_roots,
                                pred_root,
                                pred_root_meta,
                                result_file,
                                result_file_base,
                                args,
                                rank,
                                world_size,
                                logger,
                                shot,
                                commit_id,
                                previous_query_data_cot=previous_query_data_cot,
                            )

                            if world_size > 1:
                                dist.barrier()
                            
                            if len(tags) > 1:
                                query_cot_results[stage][shot] = load(result_file)['rationale'].apply(
                                        lambda x: tag + x.split(tag)[-1].split(end_tag)[0] + end_tag
                                ).tolist()

                            if status == False:
                                continue

                except Exception as e:
                    logger.exception(f'Model {model_name} x Support Dataset {support_dataset_name} x Query Dataset {query_dataset_name} combination failed: {e}, '
                                    'skipping this combination.')
                    continue

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    load_env()
    main()