import json
import numpy as np
import os

file_path = 'eval/40_modified_heuristic.json'
with open(file_path, 'r') as f:
    data = json.load(f)

results = data.get('results', [])

stats = {}

for run in results:
    algos = run.get('algorithms', {})
    for algo_name, algo_data in algos.items():
        if algo_name not in stats:
            stats[algo_name] = {
                'total': 0,
                'success': 0,
                'time': [],
                'path_length': [],
                'nodes': [],
                'iterations': []
            }
        
        stats[algo_name]['total'] += 1
        if algo_data.get('success', False):
            stats[algo_name]['success'] += 1
            stats[algo_name]['time'].append(algo_data.get('search_time', 0))
            stats[algo_name]['path_length'].append(algo_data.get('path_length', 0))
            stats[algo_name]['nodes'].append(algo_data.get('node_count', 0))
            stats[algo_name]['iterations'].append(algo_data.get('num_iterations', 0))

artifact_content = f"# BRRT Algorithms Evaluation Analysis\n\n"
artifact_content += f"**Total Test Runs:** {len(results)}\n\n"

artifact_content += "| Algorithm | Success Rate | Avg Time (s) | Avg Path Length | Avg Nodes | Avg Iterations |\n"
artifact_content += "|---|---|---|---|---|---|\n"

for algo_name, algo_stats in stats.items():
    total = algo_stats['total']
    success = algo_stats['success']
    success_rate = (success / total * 100) if total > 0 else 0
    
    avg_time = np.mean(algo_stats['time']) if algo_stats['time'] else 0
    avg_path = np.mean(algo_stats['path_length']) if algo_stats['path_length'] else 0
    avg_nodes = np.mean(algo_stats['nodes']) if algo_stats['nodes'] else 0
    avg_iters = np.mean(algo_stats['iterations']) if algo_stats['iterations'] else 0
    
    artifact_content += f"| **{algo_name}** | {success}/{total} ({success_rate:.1f}%) | {avg_time:.4f} | {avg_path:.2f} | {avg_nodes:.1f} | {avg_iters:.1f} |\n"

artifact_path = '/home/xuanloc/.gemini/antigravity-ide/brain/667c07e3-1a38-42c8-b04e-d2476a0c6e95/analysis_results.md'
with open(artifact_path, 'w') as f:
    f.write(artifact_content)

print(f"Artifact written to {artifact_path}")
