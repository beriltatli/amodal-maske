"""Independently verify saved masks, checkpoint provenance, and reported metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def verify_results(folder, expected_examples=None):
    folder = Path(folder)
    manifests = sorted(folder.glob('example_*/run.json'))
    if expected_examples is not None:
        assert len(manifests) == expected_examples, 'Missing example results'
    assert manifests, 'No results found'
    instances, calls, ids, metrics_all = 0, 0, set(), []
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'weights/manifest.json').read_text())
    for path in manifests:
        report = json.loads(path.read_text())
        assert report['status'] == 'success'
        assert report['example'] not in ids
        ids.add(report['example'])
        assert report['model_forward_calls'] > 0
        assert report['ground_truth_used_for_inference'] is False
        assert report['checkpoint_sha256'] == manifest['sha256']
        assert report['checkpoint_load'] == 'strict=True; all keys matched'
        assert report['runtime']['architecture_changed'] is False
        metrics = json.loads((path.parent / 'metrics.json').read_text())
        with np.load(path.parent / 'masks.npz', allow_pickle=False) as data:
            shape = (report['instances'], *report['image_shape'][:2])
            for key in ['modal', 'amodal_pred', 'amodal_gt']:
                assert data[key].shape == shape
                assert np.isin(data[key], [0, 1]).all()
                for index, mask in enumerate(data[key]):
                    png = np.asarray(Image.open(path.parent / key / f'{index:02d}.png'))
                    assert np.array_equal(png, mask * 255), f'PNG/NPZ mismatch: {key}/{index}'
            order = data['order_matrix']
            assert order.shape == (shape[0], shape[0])
            assert np.array_equal(order, -order.T) and np.isin(order, [-1, 0, 1]).all()
            assert len(metrics) == shape[0]
            for index, item in enumerate(metrics):
                visible, prediction, target = [data[k][index].astype(bool)
                                              for k in ['modal', 'amodal_pred', 'amodal_gt']]
                for mask, key in [(prediction, 'amodal_iou'), (visible, 'visible_baseline_iou')]:
                    union = np.count_nonzero(mask | target)
                    score = np.count_nonzero(mask & target) / union if union else 1.0
                    assert abs(score - item[key]) < 1e-12
                assert item['added_pixels'] == np.count_nonzero(prediction & ~visible)
                assert item['lost_visible_pixels'] == np.count_nonzero(visible & ~prediction)
                assert item['hidden_gt_pixels'] == np.count_nonzero(target & ~visible)
            assert abs(report['mean_amodal_iou'] - np.mean([m['amodal_iou'] for m in metrics])) < 1e-12
        instances += report['instances']
        calls += report['model_forward_calls']
        metrics_all.extend(metrics)
    summary = json.loads((folder / 'summary.json').read_text())
    assert summary['instances'] == instances and summary['model_forward_calls'] == calls
    assert abs(summary['mean_amodal_iou'] - np.mean([m['amodal_iou'] for m in metrics_all])) < 1e-12
    assert (folder / 'index.html').exists()
    result = {'status': 'passed', 'examples': sorted(ids), 'instances': instances,
              'model_forward_calls': calls,
              'checks': ['checkpoint provenance', 'strict weight loading', 'nonzero model execution',
                         'binary masks', 'PNG/NPZ equality', 'occlusion matrix consistency',
                         'independently recomputed IoU and pixel counts']}
    (folder / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--expected-examples', type=int)
    args = parser.parse_args()
    verify_results(args.folder, args.expected_examples)
