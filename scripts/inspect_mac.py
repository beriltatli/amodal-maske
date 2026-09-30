"""Lightweight image/JSON inspection. Does not run a model or create predictions."""
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main():
    data = ROOT / 'third_party/deocclusion/demos/demo_data/COCOA'
    out = ROOT / 'outputs'
    out.mkdir(exist_ok=True)
    report = {'python': sys.version, 'platform': platform.platform(),
              'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
              'venv': sys.prefix, 'model_inference': False, 'examples': []}
    fig, axes = plt.subplots(1, 5, figsize=(16, 4), constrained_layout=True)
    for index, ax in enumerate(axes, 1):
        with Image.open(data / f'{index}.jpg') as source:
            rgb = np.asarray(source.convert('RGB'))
        annotation = json.loads((data / f'{index}.json').read_text())
        count = len(annotation['regions'])
        ax.imshow(rgb)
        ax.set_title(f'Ornek {index}: {count} nesne')
        ax.axis('off')
        report['examples'].append({'id': index, 'shape': list(rgb.shape), 'regions': count})
    fig.suptitle('Resmi girdi fotograflari — model tahmini DEGIL')
    fig.savefig(out / 'sample_preview.png', dpi=130)
    plt.close(fig)
    (ROOT / 'reports/mac-check.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print('Mac kontrolü tamamlandı. Girdi önizlemesi:', out / 'sample_preview.png')
    print('Bu dosya bir model tahmini değildir.')


if __name__ == '__main__':
    main()
