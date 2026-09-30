"""Reproducible entry point: fetch pinned source, verify weights, run, build viewer."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid
import zipfile

from download_weights import download
from prepare_runtime import prepare
from build_gallery import build_gallery
from verify_results import verify_results

ROOT = Path(__file__).resolve().parents[1]


def setup_source():
    lock = json.loads((ROOT / 'third_party/deocclusion.lock.json').read_text())
    repo = ROOT / 'third_party/deocclusion'
    if not (repo / '.git').exists():
        if repo.exists() and any(repo.iterdir()):
            raise RuntimeError(f'Refusing to overwrite nonempty source directory: {repo}')
        repo.mkdir(parents=True, exist_ok=True)
        for command in [
            ['git', 'init', str(repo)],
            ['git', '-C', str(repo), 'remote', 'add', 'origin', lock['repository']],
            ['git', '-C', str(repo), 'fetch', '--depth', '1', 'origin', lock['commit']],
            ['git', '-C', str(repo), 'checkout', '--detach', lock['commit']],
        ]:
            subprocess.run(command, check=True)
    prepare(repo, ROOT / '.runtime/deocclusion')
    download(ROOT / 'weights/manifest.json', ROOT / 'weights/COCOA_pcnet_m.pth.tar')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', choices=['all', '1', '2', '3', '4', '5'], default='all')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    setup_source()
    output = args.output or ROOT / 'results' / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:6])
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory must be empty; existing runs are never overwritten.')
    output.mkdir(parents=True, exist_ok=True)
    examples = range(1, 6) if args.example == 'all' else [int(args.example)]
    for example in examples:
        print(f'Running example {example} on {args.device}...', flush=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts/run_pcnet_m.py'),
                        '--example', str(example), '--device', args.device,
                        '--output', str(output / f'example_{example}')], cwd=ROOT, check=True)
    build_gallery(output)
    verify_results(output, len(examples))
    archive = output.with_name(output.name + '.zip')
    if archive.exists():
        raise FileExistsError(f'Refusing to overwrite existing archive: {archive}')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob('*')):
            if path.is_file() and 'mpl-cache' not in path.parts:
                bundle.write(path, path.relative_to(output))
    print(f'Open the results viewer: {output / "index.html"}')
    print(f'Portable results bundle: {archive}')


if __name__ == '__main__':
    main()
