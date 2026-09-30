"""Run the official COCOA PCNet-M inference algorithms on one bundled example.

Supports CPU and NVIDIA CUDA. The unchanged official UNet and FixModule are loaded
directly, avoiding training optimizers and unrelated PCNet-C/matting imports.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import time
from types import SimpleNamespace

from download_weights import verify


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(args, report):
    started = time.perf_counter()
    # No writes to global Matplotlib caches; compatible with the local sandbox.
    os.environ.setdefault('MPLCONFIGDIR', str(args.output / 'mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import pycocotools.mask as mask_utils
    import torch
    import yaml

    report.update(torch=torch.__version__, cuda_build=torch.version.cuda,
                  cuda_available=torch.cuda.is_available())
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Use --device cpu.')
    torch.set_num_threads(args.threads)
    report['device'] = device
    report['threads'] = args.threads
    report['gpu'] = torch.cuda.get_device_name(0) if device == 'cuda' else None
    report['packages'] = {name: importlib.metadata.version(name) for name in [
        'numpy', 'Pillow', 'matplotlib', 'pycocotools', 'PyYAML', 'scikit-image']}
    # Record portable versions, without local wheel paths or user-directory names.
    installed = json.loads(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'list', '--format=json'], text=True))
    report['pip_freeze'] = sorted(f"{p['name']}=={p['version']}" for p in installed)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sys.path.insert(0, str(args.runtime.resolve()))
    import inference as infer
    from models.backbone import unet2, FixModule
    from official_helpers import read_COCOA, expand_bbox

    config_path = args.repo / 'experiments/COCOA/pcnet_m/config.yaml'
    config = yaml.safe_load(config_path.read_text())
    params = config['model']
    if (params['backbone_arch'] != 'unet2' or params['use_rgb'] is not False
            or params['backbone_param'] != {'in_channels': 2, 'n_classes': 2}):
        raise RuntimeError('Beklenen resmî COCOA PCNet-M yapılandırması değil.')
    manifest = json.loads(args.manifest.read_text())
    report['checkpoint_sha256'] = verify(args.weights, manifest)
    # Explicit safe tensor loading; never fall back to a random model on failure.
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
        raise RuntimeError('Checkpoint state_dict içermiyor.')
    net = FixModule(unet2(**params['backbone_param']))
    state = checkpoint['state_dict']
    if set(state) != set(net.state_dict()):
        missing = sorted(set(net.state_dict()) - set(state))
        extra = sorted(set(state) - set(net.state_dict()))
        raise RuntimeError(f'Ağırlık anahtarları uyuşmuyor. Eksik: {missing}; fazla: {extra}')
    net.load_state_dict(state, strict=True)
    report['checkpoint_load'] = 'strict=True; all keys matched'
    report['checkpoint_step'] = int(checkpoint['step']) if 'step' in checkpoint else None
    del state, checkpoint
    for tensor in net.state_dict().values():
        if not torch.isfinite(tensor).all().item():
            raise RuntimeError('Ağırlıklarda sonlu olmayan değer bulundu.')
    net = net.to(device).eval()
    model = SimpleNamespace(model=net)

    image_path = args.repo / f'demos/demo_data/COCOA/{args.example}.jpg'
    annot_path = image_path.with_suffix('.json')
    image = np.asarray(Image.open(image_path).convert('RGB'))
    height, width = image.shape[:2]
    ann = json.loads(annot_path.read_text())
    modal, category, boxes, gt = [], [], [], []
    for region in ann['regions']:
        visible, box, cat = read_COCOA(region, height, width)
        if not visible.any():
            raise RuntimeError('Tamamen görünmez nesne bulundu; bu başlangıç örneği desteklemiyor.')
        modal.append(visible)
        boxes.append(box)
        category.append(cat)
        gt.append(mask_utils.decode(mask_utils.merge(
            mask_utils.frPyObjects([region['segmentation']], height, width))))
    modal, category, boxes, gt = map(np.asarray, (modal, category, boxes, gt))
    bboxes = expand_bbox(boxes, enlarge_ratio=3., single_ratio=1.5)
    calls = []

    def check_logits(module, inputs, output):
        if not torch.isfinite(output).all().item():
            raise RuntimeError('Tahmin logits değerleri NaN/Inf içeriyor.')
        calls.append(list(output.shape))

    hook = net.register_forward_hook(check_logits)
    with torch.inference_mode():
        order = infer.infer_order(
            model, image, modal, category, bboxes, use_rgb=False, th=0.1,
            dilate_kernel=0, input_size=256, min_input_size=16,
            interp='nearest', debug_info=False)
        predicted_patches = infer.infer_amodal(
            model, image, modal, category, bboxes, order, use_rgb=False, th=0.5,
            dilate_kernel=0, input_size=256, min_input_size=16,
            interp='linear', debug_info=False)
        predicted = infer.patch_to_fullimage(
            predicted_patches, bboxes, height, width, interp='linear')
    hook.remove()
    if not calls or predicted.shape != modal.shape or not np.isin(predicted, [0, 1]).all():
        raise RuntimeError('Gerçek model çağrısı / tahmin boyutu / ikili maske kontrolü başarısız.')

    # Per-instance sample diagnostics, never a full-dataset benchmark claim.
    metrics = []
    for index, (visible, prediction, target) in enumerate(zip(modal, predicted, gt)):
        visible, prediction, target = (m.astype(bool) for m in (visible, prediction, target))
        def iou(a, b):
            union = int((a | b).sum())
            return float((a & b).sum() / union) if union else 1.0
        hidden = target & ~visible
        added = prediction & ~visible
        metrics.append({'instance': index, 'visible_pixels': int(visible.sum()),
                        'predicted_pixels': int(prediction.sum()), 'hidden_gt_pixels': int(hidden.sum()),
                        'added_pixels': int(added.sum()),
                        'lost_visible_pixels': int((visible & ~prediction).sum()),
                        'amodal_iou': iou(prediction, target), 'visible_baseline_iou': iou(visible, target),
                        'hidden_iou': iou(added, hidden) if hidden.any() else None})
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    np.savez_compressed(args.output / 'masks.npz', modal=modal,
                        amodal_pred=predicted, amodal_gt=gt, order_matrix=order,
                        bboxes=bboxes, category=category)
    Image.fromarray(image).save(args.output / 'input.png')
    for label, masks in [('modal', modal), ('amodal_pred', predicted), ('amodal_gt', gt)]:
        folder = args.output / label
        folder.mkdir(exist_ok=True)
        for index, mask in enumerate(masks):
            Image.fromarray((mask * 255).astype(np.uint8)).save(folder / f'{index:02d}.png')

    colors = np.random.default_rng(0).uniform(0.2, 1., (len(modal), 3))
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    for ax, masks, title in zip(axes, [modal, predicted, gt],
                               ['Girdi: gorunur maskeler', 'PCNet-M: amodal tahmin', 'Referans: amodal GT']):
        ax.imshow(image)
        for index, mask in enumerate(masks):
            rgba = np.zeros((height, width, 4), dtype=np.float32)
            rgba[..., :3] = colors[index]
            rgba[..., 3] = mask * 0.45
            ax.imshow(rgba)
        ax.set_title(title)
        ax.axis('off')
    fig.savefig(args.output / 'comparison.png', dpi=150)
    plt.close(fig)
    # Select an occluded object for a legible binary-mask comparison. GT is only
    # used for this display selection, never passed to infer_order/infer_amodal.
    selected = int(((gt == 1) & (modal == 0)).sum(axis=(1, 2)).argmax())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, mask, title in zip(axes, [modal[selected], predicted[selected], gt[selected]],
                              ['Gorunur maske', 'PCNet-M tahmini', 'Referans GT']):
        ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Nesne {selected}: {title}')
        ax.axis('off')
    fig.savefig(args.output / 'object_comparison.png', dpi=150)
    plt.close(fig)
    report.update(status='success', example=args.example, instances=len(modal),
                  image_shape=list(image.shape), model_forward_calls=len(calls),
                  parameters=sum(p.numel() for p in net.parameters()),
                  input_image_sha256=file_hash(image_path), annotation_sha256=file_hash(annot_path),
                  runner_sha256=file_hash(__file__),
                  config_sha256=file_hash(config_path),
                  runtime=json.loads((args.runtime / 'runtime_manifest.json').read_text()),
                  thresholds={'order': 0.1, 'amodal': 0.5},
                  input_size=256, seed=0, ground_truth_used_for_inference=False,
                  prediction_postprocessing='Official patch_to_fullimage; no extra union with modal masks',
                  elapsed_seconds=round(time.perf_counter() - started, 3),
                  mean_amodal_iou=float(np.mean([m['amodal_iou'] for m in metrics])),
                  mean_visible_baseline_iou=float(np.mean([m['visible_baseline_iou'] for m in metrics])),
                  limitation='Bundled demonstration sample; not a held-out dataset benchmark.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path('third_party/deocclusion'))
    parser.add_argument('--runtime', type=Path, default=Path('.runtime/deocclusion'))
    parser.add_argument('--manifest', type=Path, default=Path('weights/manifest.json'))
    parser.add_argument('--weights', type=Path, default=Path('weights/COCOA_pcnet_m.pth.tar'))
    parser.add_argument('--example', type=int, choices=range(1, 6), default=4)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    # Prevent stale masks from looking like the result of a failed new run.
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError('Çıktı klasörü dolu. Yeni bir çıktı klasörü seçin.')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'started', 'python': sys.version, 'platform': platform.platform(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat()}
    try:
        run(args, report)
        print('GERÇEK TAHMİN TAMAMLANDI:', args.output)
    except Exception:
        report.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        (args.output / 'run.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
