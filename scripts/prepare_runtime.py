"""Build a small inference-only view of pinned upstream code; never edit third_party.

Original architecture, forward pass and inference algorithms are preserved.
NumPy aliases and tensor device placement are adapted in inference.py. Exact official
read_COCOA and expand_bbox functions are extracted to avoid training/demo imports.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

COMMIT = 'ac543f9a54f4cb8baf0774bdb5f8638eadf8b57c'


def extract(source, name):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node) + '\n'


def prepare(repo, destination):
    repo, destination = Path(repo).resolve(), Path(destination).resolve()
    if destination == repo or repo in destination.parents or destination in repo.parents:
        raise ValueError('Çalışma kopyası resmî depodan ayrı olmalı.')
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True)
    if head != COMMIT or dirty:
        raise RuntimeError(f'Resmî depo commit/temizlik kontrolü başarısız: {head}\n{dirty}')
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}

    def write(relative, source):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)

    def read(relative):
        raw = (repo / relative).read_bytes()
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        return raw.decode()

    for file in ['models/backbone/unet/unet_model.py', 'models/backbone/unet/unet_parts.py',
                 'models/backbone/others.py', 'utils/data_utils.py', 'LICENSE']:
        write(file, read(file))
    write('models/__init__.py', '# Generated inference-only package.\n')
    write('models/backbone/__init__.py', 'from .unet import unet2\nfrom .others import FixModule\n')
    write('models/backbone/unet/__init__.py', 'from .unet_model import unet2\n')
    write('utils/__init__.py', 'from .data_utils import *\n')
    source = read('inference.py')
    counts = {}
    for old, new in [('int', 'int'), ('bool', 'bool'), ('float', 'float')]:
        source, counts[old] = re.subn(r'\bnp\.' + old + r'\b', new, source)
    device_calls = source.count('.cuda()')
    if device_calls != 8:
        raise RuntimeError('Unexpected upstream device calls; review adaptation.')
    source = source.replace('.cuda()', '.to(next(model.model.parameters()).device)')
    write('inference.py', '# Generated compatibility copy; see runtime_manifest.json.\n' + source)
    helper = '# Functions extracted unchanged from the pinned Apache-2.0 source.\n'
    helper += 'import numpy as np\nimport pycocotools.mask as maskUtils\nimport utils\n\n'
    helper += extract(read('datasets/reader.py'), 'read_COCOA') + '\n'
    helper += extract(read('demos/demo_utils.py'), 'expand_bbox')
    write('official_helpers.py', helper)
    report = {'commit': head, 'source_sha256': hashes, 'numpy_alias_replacements': counts,
              'scope': 'COCOA PCNet-M inference only; not a training environment',
              'architecture_changed': False, 'cuda_calls_preserved': False,
              'device_placement_replacements': device_calls,
              'device_policy': 'Input tensors follow the model device; no algorithm changes'}
    write('runtime_manifest.json', json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='third_party/deocclusion')
    parser.add_argument('--output', default='.runtime/deocclusion')
    args = parser.parse_args()
    print(json.dumps(prepare(args.repo, args.output), indent=2))
