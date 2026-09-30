"""Build an offline, self-contained viewer from successful, real inference runs."""
import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def image_uri(path):
    return 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode('ascii')


def build_gallery(folder):
    folder = Path(folder)
    examples, all_metrics, runs = [], [], []
    for path in sorted(folder.glob('example_*/run.json')):
        run = json.loads(path.read_text())
        if run['status'] != 'success' or run['model_forward_calls'] <= 0:
            raise ValueError(f'Cannot display an unsuccessful inference: {path}')
        root = path.parent
        metrics = json.loads((root / 'metrics.json').read_text())
        examples.append({'id': run['example'], 'run': run, 'metrics': metrics,
                         'image': image_uri(root / 'input.png'),
                         'masks': {key: [image_uri(root / key / f'{i:02d}.png')
                                        for i in range(run['instances'])]
                                   for key in ['modal', 'amodal_pred', 'amodal_gt']}})
        all_metrics.extend(metrics)
        runs.append(run)
    if not examples:
        raise ValueError('No successful example runs found.')
    summary = {'generated_at': datetime.now(timezone.utc).isoformat(),
               'examples': len(runs), 'instances': len(all_metrics),
               'model_forward_calls': sum(r['model_forward_calls'] for r in runs),
               'mean_amodal_iou': sum(m['amodal_iou'] for m in all_metrics) / len(all_metrics),
               'mean_visible_baseline_iou': sum(m['visible_baseline_iou'] for m in all_metrics) / len(all_metrics),
               'added_pixels': sum(m['added_pixels'] for m in all_metrics),
               'lost_visible_pixels': sum(m['lost_visible_pixels'] for m in all_metrics),
               'device': sorted({r['device'] for r in runs}),
               'checkpoint_sha256': runs[0]['checkpoint_sha256'],
               'scope': 'Five bundled COCOA demo images at most; descriptive metrics, not a dataset benchmark.'}
    payload = json.dumps({'examples': examples, 'summary': summary}).replace('<', '\\u003c')
    template = (ROOT / 'web/viewer.html').read_text()
    (folder / 'index.html').write_text(template.replace('__RESULTS_JSON__', payload))
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    build_gallery(parser.parse_args().folder)
