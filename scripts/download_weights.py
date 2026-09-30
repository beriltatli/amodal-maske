"""Download exactly the official COCOA PCNet-M checkpoint; reject HTML/partial files."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def verify(path, manifest):
    path = Path(path)
    if path.stat().st_size != manifest['bytes']:
        raise ValueError(f"Boyut yanlış: {path.stat().st_size}; beklenen {manifest['bytes']}")
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != manifest['sha256']:
        raise ValueError(f"SHA-256 uyuşmuyor: {digest}. Dosyayı model olarak yüklemeyin.")
    return digest


def download(manifest_path, destination):
    manifest = json.loads(Path(manifest_path).read_text())
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print('Mevcut ağırlık doğrulandı:', verify(destination, manifest))
        return
    print(f"Resmî COCOA PCNet-M: {manifest['bytes']:,} bayt; yalnızca maske tahmini.", flush=True)
    if manifest['bytes'] > 500_000_000:
        raise RuntimeError('500 MB sınırı: indirme başlatılmadı.')
    partial = destination.with_name(destination.name + '.part')
    subprocess.run([
        'curl', '--location', '--fail', '--show-error', '--connect-timeout', '30',
        '--max-time', '600', '--max-filesize', str(manifest['bytes']),
        '--output', str(partial), manifest['download_url'],
    ], check=True)
    digest = verify(partial, manifest)
    partial.replace(destination)
    print('İndirme ve SHA-256 doğrulandı:', digest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='weights/manifest.json')
    parser.add_argument('--output', default='weights/COCOA_pcnet_m.pth.tar')
    args = parser.parse_args()
    download(args.manifest, args.output)
