"""CEC2022 DGDE 单次重复参数网格实验。

直接导入同目录 new_DGDE.py 和 all_function_new.py。
默认 F1-F12，100维、100代、seed=0；每函数17280组参数。
python run_DGDE_CEC2022_function.py --dry-run
python run_DGDE_CEC2022_function.py --functions 1 2 3 --limit 3 --generations 3 --output smoke_cec
python run_DGDE_CEC2022_function.py --workers 4

先由主进程串行准备实例，再启动worker，避免多个进程同时生成同一个文件。
保留已有实例；缺失实例由instance-seed确定性生成，独立于优化随机种子。
默认两阶段盒约束投影。结果分F01-F12保存，支持断点续跑和失败记录。
沿用当前函数库的块稀疏高维改编实例，不等同于官方原始CEC2022实例。
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import platform
import random
import sys
import time
import traceback
from new_DGDE import DGDE
from all_function_new import select_function
import new_DGDE
import all_function_new
import CEC2022_0923
import scipy
import estimated_gradient

import numpy as np

# 与 run_30_GD_EA.py 一样，可在此编辑默认实验设置。
Dimension = [100]
budget = 100
seed = 0
T_per = 1  # 兼容参数；新版 DGDE 每代重新抽样策略。
pho_high = 0.8
pho_low = 0.8
num_processes = min(4, max(1, (os.cpu_count() or 1)-1))

GRID = {
    'population_size': [10, 20, 30, 40],
    'niche_size': [3, 4, 5, 7],
    'fitness_p': [0.01, 0.05, 0.1, 0.3, 0.5, 0.7],
    'F': [0.3, 0.5, 0.7],
    'K_base': [0.3, 0.5, 0.7],
    'K_0': [1.1, 1.3, 1.5, 1.7],
    'learning_rate': [0.01, 0.001, 0.0001, 0.00001, 0.000001],
}
FUNS = list(range(1, 13))
FUNCTIONS = FUNS
BASELINE = dict(population_size=20, niche_size=5, fitness_p=0.5,
                F=0.5, K_base=0.5, K_0=1.5, learning_rate=0.001)
BENCHMARK_SOURCE = 'local all_function_new.py / CEC2022_0923.py; block-sparse high-dimensional adaptation'


_FUNCTION_CACHE = {}


def get_function(func_num, dimension, data_dir):
    key = (func_num, dimension, str(data_dir))
    if key not in _FUNCTION_CACHE:
        # 限制每个worker持有的高维实例，避免缓存全部30函数的矩阵。
        if len(_FUNCTION_CACHE) >= 2:
            _FUNCTION_CACHE.clear()
            all_function_new._CACHED.clear()
        _FUNCTION_CACHE[key] = select_function(func_num, dimension, data_dir=data_dir)
    return _FUNCTION_CACHE[key]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_instances(args):
    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    rng_state = np.random.get_state()
    try:
        for dimension in args.dimensions:
            for func_num in args.functions:
                local_seed = int(np.random.SeedSequence([args.instance_seed, func_num, dimension]).generate_state(1)[0])
                np.random.seed(local_seed)
                all_function_new._CACHED.clear()
                all_function_new.load_shift_and_rot(func_num, dimension, str(data_dir))
                filenames = [f'shift_data_{func_num}_D{dimension}.pkl', f'M_{func_num}_D{dimension}.pkl']
                if 6 <= func_num <= 8:
                    filenames.append(f'shuffle_data_{func_num}_D{dimension}.pkl')
                for filename in filenames:
                    hashes[filename] = file_hash(data_dir/filename)
    finally:
        np.random.set_state(rng_state)
        all_function_new._CACHED.clear()
    return hashes


def parameter_sets(mode):
    if mode == 'grid':
        for values in itertools.product(*GRID.values()):
            yield dict(zip(GRID, values))
    else:
        seen = set()
        for key, values in GRID.items():
            for value in values:
                config = dict(BASELINE, **{key: value})
                identity = tuple(config[k] for k in GRID)
                if identity not in seen:
                    seen.add(identity)
                    yield config


def atomic_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, path)


def task_path(root, task):
    identity = json.dumps(task, sort_keys=True, separators=(',', ':'))
    key = hashlib.sha256(identity.encode()).hexdigest()[:20]
    folder = Path(root) / ('F%02d' % task['function']) / (str(task['dimension'])+'D')
    return folder / ('DGDE_'+key+'.pkl')


def run_one(payload):
    root, task, retry = payload
    path = task_path(root, task)
    if path.exists():
        with path.open('rb') as stream:
            saved = pickle.load(stream)
        if saved['status'] == 'ok' or not retry:
            return summary_row(saved, path, True)
    np.random.seed(task['seed'])
    estimated_gradient.objective_calls = 0
    random.seed(task['seed'])
    estimated_gradient.objective_calls = 0
    params = dict(task['parameters'], dimension=task['dimension'], budget=task['generations'],
                  T_per=T_per, pho_high=pho_high, pho_low=pho_low, boundary_handling=task['boundary'],
                  store_population_history=False)
    model = DGDE(**params)
    result = dict(schema_version=2, algorithm='DGDE', function=task['function'],
                  run_id=0, seed=task['seed'], parameters=params,
                  benchmark='local-CEC2022-block-sparse-adaptation', status='failed', error=None)
    started = time.perf_counter()
    try:
        f, g, lb, ub = get_function(task['function'], task['dimension'], task['data_dir'])
        result['parameters'].update(x_min=lb, x_max=ub)
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            model.DGDE(lb, ub, f, g)
        result.update(status='ok', final_best_fitness=float(model.g_best.target.fitness),
                      # 保留旧脚本字段含义：best_solution 是标量适应度，不是坐标。
                      best_solution=float(model.g_best.target.fitness),
                      final_best_position=model.g_best.solution.copy(),
                      best_so_far_fitness=float(model.best_so_far.target.fitness),
                      best_so_far_position=model.best_so_far.solution.copy(),
                      final_population=np.array([a.solution for a in model.pop]),
                      final_fitness=np.array([a.target.fitness for a in model.pop]))
    except Exception:
        result['error'] = traceback.format_exc()
    result.update(run_time=time.perf_counter()-started, nfe=model.nfe, nge=model.nge,
                  gradient_internal_nfe=estimated_gradient.objective_calls,
                  total_objective_calls=model.nfe + estimated_gradient.objective_calls,
                  trial_projections=model.trial_projections, step_projections=model.step_projections,
                  completed_generations=max(0, len(getattr(model, 'history', []))-1),
                  best_fitness=list(model.best_history),
                  current_best_history=list(getattr(model, 'history', [])),
                  stop_reason=model.stop_reason if result['status'] == 'ok' else 'error')
    atomic_pickle(path, result)
    return summary_row(result, path, False)


def summary_row(result, path, cached):
    keys = ['function', 'seed', 'status', 'run_time', 'nfe', 'nge', 'completed_generations',
            'final_best_fitness', 'best_so_far_fitness', 'trial_projections', 'step_projections', 'gradient_internal_nfe', 'total_objective_calls']
    row = {key: result.get(key) for key in keys}
    row.update(result['parameters'])
    row.update(file=str(path), resumed=cached,
               error=(result.get('error') or '').strip().split('\n')[-1])
    return row


def make_tasks(args):
    # 按参数优先遍历；--limit 12 覆盖全部12函数的一组参数。
    for params in parameter_sets(args.search):
        for dimension in args.dimensions:
            for name in args.functions:
                yield dict(function=name, dimension=dimension, generations=args.generations,
                           seed=args.seed, boundary=args.boundary, parameters=params, data_dir=str(args.data_dir.resolve()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dimensions', type=int, nargs='+', default=Dimension)
    parser.add_argument('--generations', type=int, default=budget, help='代数 T，不是评估次数')
    parser.add_argument('--seed', type=int, default=seed, help='每组仅一次，所有组合使用同一seed')
    parser.add_argument('--workers', type=int, default=num_processes)
    parser.add_argument('--functions', type=int, nargs='+', choices=FUNCTIONS, default=FUNCTIONS)
    parser.add_argument('--search', choices=['grid', 'one-at-a-time'], default='grid')
    parser.add_argument('--boundary', choices=['none', 'clip'], default='clip')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'results_DGDE_CEC2022_tuning_projected')
    parser.add_argument('--limit', type=int, help='只执行前N个任务，用于试跑')
    parser.add_argument('--dry-run', action='store_true', help='只显示任务数量，不运行优化')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent/'pkl_data_sparsity')
    parser.add_argument('--instance-seed', type=int, default=42)
    args = parser.parse_args(argv)
    if (any(d not in (100, 1000, 10000) for d in args.dimensions) or args.generations < 1 or args.workers < 1
            or not 0 <= args.seed < 2**32 or not 0 <= args.instance_seed < 2**32 or (args.limit is not None and args.limit < 1)):
        parser.error('CEC2022库支持维度100/1000/10000；代数、进程数、limit需>=1；种子需在[0,2**32)')
    args.dimensions = list(dict.fromkeys(args.dimensions))
    args.functions = list(dict.fromkeys(args.functions))
    nparams = sum(1 for _ in parameter_sets(args.search))
    total = nparams*len(args.dimensions)*len(args.functions)
    total = min(total, args.limit) if args.limit else total
    print(f'Parameters={nparams}; tasks={total}; dimensions={args.dimensions}; T={args.generations}; seed={args.seed}', flush=True)
    print(f'Boundary={args.boundary}; output={args.output.resolve()}', flush=True)
    print('Per completed run: NFE=NP*(T+1), NGE=NP*T; failed runs are recorded separately.', flush=True)
    print('NFE counts direct calls; F4 finite-difference calls are also saved as gradient_internal_nfe and total_objective_calls.', flush=True)
    if args.dry_run:
        return 0
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    instance_hashes = prepare_instances(args)
    manifest = dict(schema_version=2, grid=GRID, baseline=BASELINE, search=args.search,
                    dimensions=args.dimensions, functions=args.functions, generations=args.generations,
                    seed=args.seed, boundary=args.boundary, rho_high=pho_high, rho_low=pho_low, T_per=T_per,
                    projection='trial-and-post-gradient' if args.boundary == 'clip' else 'none',
                    python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, benchmark_source=BENCHMARK_SOURCE,
                    instance_seed=args.instance_seed, data_dir=str(args.data_dir.resolve()),
                    instance_files=instance_hashes,
                    source_sha256={Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                   for p in (__file__, new_DGDE.__file__, all_function_new.__file__, CEC2022_0923.__file__, estimated_gradient.__file__)})
    manifest_path = root/'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Output contains a different experiment/code/environment. Choose a new --output directory.')
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    tasks = itertools.islice(make_tasks(args), total)
    payloads = ((str(root), task, args.retry_failed) for task in tasks)
    pool = None
    streams, writers = {}, {}
    failed = 0
    try:
        if args.workers == 1:
            results = map(run_one, payloads)
        else:
            pool = mp.get_context('spawn').Pool(args.workers)
            results = pool.imap_unordered(run_one, payloads, chunksize=1)
        for i, row in enumerate(results, 1):
            key = (row['function'], row['dimension'])
            if key not in streams:
                folder = Path(row['file']).parent
                streams[key] = (folder/'summary.csv').open('w', newline='', encoding='utf-8-sig')
                writers[key] = csv.DictWriter(streams[key], fieldnames=list(row))
                writers[key].writeheader()
            writers[key].writerow(row)
            streams[key].flush()
            failed += row['status'] != 'ok'
            if i == 1 or i % 100 == 0 or i == total:
                print(f'[{i}/{total}] failed={failed}; {row["function"]}; {row["status"]}', flush=True)
        if pool:
            pool.close()
            pool.join()
    finally:
        if pool:
            pool.terminate()
            pool.join()
        for stream in streams.values():
            stream.close()
    print(f'Finished: {total-failed} successful, {failed} failed. Results: {root}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    mp.freeze_support()
    sys.exit(main())
