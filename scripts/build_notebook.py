"""Build a self-contained Colab notebook from the reviewable helper files."""
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build():
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))
    def code(text):
        cells.append(nbf.v4.new_code_cell(text.strip()))

    md('''# PCNet-M: resmî COCOA örneği

**Durum:** Resmî COCOA ağırlığı Mac'te tamamen indirildi ve SHA-256 kaydedildi.
Bu notebook'un CUDA çalıştırması henüz doğrulanmadı; kayıtlı tahmin çıktısı içermez.
Linux/Python 3.12 yardımcı paket çözümü ve wheel bağlantıları kontrol edildi.
Başarı ancak 6. hücre gerçek ağırlıklarla tahmin üretip `run.json` içinde `success` yazdığında doğrulanır.

1. Colab → **Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU** seçin.
2. Sürüm seçeneği varsa **2026.07** seçin (resmî listede Python 3.12.13 / PyTorch 2.11.0).
3. Aşağıdaki **1–8 numaralı kod hücrelerini sırayla** çalıştırın. Hata olursa durun ve **8. hücreyi** çalıştırarak tanı ZIP'ini indirin.

Ücretsiz GPU kullanın; ücretli kaynak oluşturulmaz. Eğitim ve tam veri kümesi indirme yoktur.
PCNet-M'ye verilen görünür maskeler resmî örnek anotasyonlarından gelir; bu bir RGB'den otomatik nesne bulma sistemi değildir.
COCOA modeli `use_rgb=False` kullanır: ağın girdisi hedef maske ve örten nesne maskesidir.
GT amodal maskeler yalnızca sonuç karşılaştırmasında kullanılır.

[Resmî depo](https://github.com/XiaohangZhan/deocclusion) ·
[Ağırlık klasörü](https://drive.google.com/drive/folders/1O89ItVWucCoL_VxIbLM1XLxr9JFfyj_Y) ·
[Colab sürüm listesi](https://research.google.com/colaboratory/runtime-version-faq.html)

EfficientSAM-Ti'ye geçiş yapılmaz; onun standart ağırlıkları amodal değildir ve ayrı amodal ince ayar gerektirir.''')
    md('''## 1 — Ortamı kontrol et

Bu hücre gerçek Python/PyTorch/CUDA sürümlerini kaydeder. Python 3.12 ve Linux x86_64 gereklidir.
PyTorch 2.11.0 dışında durur; uygun 2026.07 çalışma zamanını seçin. Eski PyTorch veya CUDA kurmaz.
Bu sürüm seçimi **çalıştırılarak doğrulanmış bir uyumluluk iddiası değildir**; sabitlenmiş başlangıç adayıdır.''')
    code('''from pathlib import Path
import os, sys, json, platform, subprocess, hashlib, shutil, uuid
from datetime import datetime, timezone

ROOT = Path('/content/pcnet-m-start')
ROOT.mkdir(parents=True, exist_ok=True)
LOGS = ROOT / 'logs'
LOGS.mkdir(exist_ok=True)
SESSION = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:6]
OUT = ROOT / 'outputs' / SESSION

def command(argv, name, env=None, cwd=None):
    print('Çalıştırılıyor:', ' '.join(map(str, argv)), flush=True)
    log = LOGS / (SESSION + '_' + name + '.log')
    with log.open('w') as stream:
        proc = subprocess.Popen(list(map(str, argv)), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=env, cwd=cwd)
        for line in proc.stdout:
            print(line, end='')
            stream.write(line)
        result = proc.wait()
    if result:
        raise RuntimeError(f'{name} başarısız (kod {result}). Log: {log}. 8. hücre ile tanı indirin.')

try:
    import torch
    environment = {'python': sys.version, 'platform': platform.platform(),
                   'torch': torch.__version__, 'cuda_build': torch.version.cuda,
                   'cuda_available': torch.cuda.is_available(),
                   'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
except Exception as exc:
    environment = {'python': sys.version, 'platform': platform.platform(), 'torch_error': repr(exc)}
(LOGS / (SESSION + '_environment.json')).write_text(json.dumps(environment, indent=2))
print(json.dumps(environment, indent=2))
assert sys.platform == 'linux' and platform.machine() == 'x86_64', 'Linux x86_64 Colab ortamı gerekiyor.'
assert sys.version_info[:2] == (3, 12), 'Bu kilit Python 3.12 için. Colab 2026.07 sürümünü seçin.'
assert environment.get('torch', '').split('+')[0] == '2.11.0', 'Colab 2026.07 / PyTorch 2.11.0 seçin; burada PyTorch indirmiyoruz.'
assert environment.get('cuda_available'), 'GPU yok. Çalışma zamanı türünden ücretsiz T4 GPU seçin.'
''')
    md('''## 2 — Sabit commit ve yardımcı dosyalar

Resmî depo `third_party/deocclusion` altında tutulur. Kopyası Mac'te yaklaşık 39 MB yer kapladı.
Notebook kendine yeterlidir; Mac'ten script veya veri yüklemeniz gerekmez. Aşağıdaki gömülü dosyalar
projedeki `scripts/` dosyalarından üretilmiştir. Kaynak kodu indirmesi tam veri kümesini içermez.''')
    files = ['scripts/prepare_runtime.py', 'scripts/download_weights.py', 'scripts/run_pcnet_m.py',
             'weights/manifest.json', 'third_party/deocclusion.lock.json', 'requirements-colab.lock.txt',
             'reports/colab-wheels.json']
    payload = {file: (ROOT / file).read_text() for file in files}
    code('BUNDLE = ' + repr(payload) + '''
for relative, content in BUNDLE.items():
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
lock = json.loads((ROOT / 'third_party/deocclusion.lock.json').read_text())
REPO = ROOT / 'third_party/deocclusion'
if not (REPO / '.git').exists():
    REPO.mkdir(parents=True, exist_ok=True)
    command(['git', 'init', REPO], 'git_init')
    command(['git', '-C', REPO, 'remote', 'add', 'origin', lock['repository']], 'git_remote')
    command(['git', '-C', REPO, 'fetch', '--depth', '1', 'origin', lock['commit']], 'git_fetch')
    command(['git', '-C', REPO, 'checkout', '--detach', lock['commit']], 'git_checkout')
actual = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
assert actual == lock['commit'], 'Commit uyuşmuyor; yeni Colab çalışma zamanı açın.'
assert not subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip()
print('Sabit resmî commit:', actual)
''')
    md('''## 3 — Hafif yardımcı paketleri ayrı klasöre kur

Colab'ın PyTorch/CUDA kurulumu korunur. Yardımcı paketler yalnızca bu oturumdaki `support/` klasörüne
kurulur ve sonraki Python alt süreçleri bu klasörü kullanır. Notebook çekirdeğine paket yüklenmez;
yeniden başlatma gerekmez. Her paketin sürümü ve wheel SHA-256 değeri kilitlidir.

Önce indirme boyutları gösterilir. 500 MB üzerindeki veya boyutu belirlenemeyen wheel indirilmez.
Bu adım paket kurulumunu sınar; GPU tahmini 6. hücrededir.''')
    code('''import urllib.request
SUPPORT = ROOT / 'support'
req = ROOT / 'requirements-colab.lock.txt'
req_hash = hashlib.sha256(req.read_bytes()).hexdigest()
marker = SUPPORT / '.complete'
if not marker.exists():
    if SUPPORT.exists() and any(SUPPORT.iterdir()):
        raise RuntimeError('Önceki kurulum yarım kalmış. Yeni Colab çalışma zamanı açıp tekrar çalıştırın.')
    items = json.loads((ROOT / 'reports/colab-wheels.json').read_text())
    total = 0
    for item in items:
        name = item['name']
        assert not name.lower().startswith(('torch', 'nvidia', 'triton')), name
        url = item['url']
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as response:
            size = int(response.headers.get('Content-Length', '0'))
        print(f'{name}: {size / 1e6:.2f} MB — veri okuma / maske işleme / çizim bağımlılığı')
        assert 0 < size <= 500_000_000, f'{name}: boyut belirsiz veya 500 MB sınırını aşıyor; indirme durduruldu.'
        total += size
    print(f'Toplam yardımcı paket indirmesi: {total / 1e6:.1f} MB')
    command([sys.executable, '-m', 'pip', 'install', '--target', SUPPORT,
             '--only-binary=:all:', '--require-hashes', '--no-deps',
             '--report', LOGS / (SESSION + '_install_report.json'), '-r', req], 'dependency_install')
    marker.write_text(req_hash)
assert marker.read_text() == req_hash, 'Paket kilidi değişmiş; yeni Colab çalışma zamanı açın.'
ENV = os.environ.copy()
ENV['PYTHONPATH'] = str(SUPPORT)
ENV['MPLCONFIGDIR'] = str(ROOT / 'mpl-cache')
check = "import sys,torch,numpy,cv2,yaml,skimage,pycocotools; from skimage.morphology import convex_hull; from PIL import Image; import matplotlib; print(sys.version); print('torch',torch.__version__,'numpy',numpy.__version__,'cv2',cv2.__version__,'CUDA',torch.cuda.is_available()); assert torch.cuda.is_available(); x=torch.ones(1,device='cuda'); print('CUDA tensor check:',x.cpu().tolist())"
command([sys.executable, '-c', check], 'imports_and_cuda_tensor', ENV)
''')
    md('''## 4 — Resmî ağırlığı indir ve doğrula

Yalnızca `COCOA_pcnet_m.pth.tar`: **26.870.559 bayt ≈ 26,9 MB**.
Bu dosyanın tamamı Mac'te 7 Eylül 2026 tarihinde indirildi.
Boyut ve SHA-256 eşleşmeden tahmine geçilmez. `.pth.tar` adına rağmen dosya bir PyTorch checkpoint'idir;
`tar` komutuyla açmayın. Drive erişimi başarısızsa HTML çıktısı ağırlık kabul edilmez.

SHA-256: `3740e8f709f824068dd36be46bd44ffd18105c4f606c099789c93f3ae4b860f7`.
Bu özet ilk resmî indirmeden yerelde hesaplandı; yayıncının sağladığı bir imza değildir.''')
    code('''WEIGHTS = ROOT / 'weights/COCOA_pcnet_m.pth.tar'
command([sys.executable, ROOT / 'scripts/download_weights.py',
         '--manifest', ROOT / 'weights/manifest.json', '--output', WEIGHTS], 'weights_download')
''')
    md('''## 5 — Uyumluluk kopyasını hazırla

Resmî UNet katmanları ve ileri geçiş kodu aynen alınır. Tahmin kopyasındaki eski `np.int`/`np.bool`
ifadeleri Python türleriyle değiştirilir. Girdi tensörleri modelin cihazına taşınır;
bu notebook CUDA cihazını kullanır, yerel demo ise CPU'yu da destekler.
`read_COCOA` ve `expand_bbox` resmî koddan aynen alınır.
YAML `safe_load` ile; ağırlık `weights_only=True`, `strict=True` ile yüklenir.
PCNet-C, matting, eğitim, optimizer ve eski `cvbase` bağımlılıkları bu tahmin akışına alınmaz.
Bu düzenlemeler tam resmî eğitim ortamına uyumluluk sağlamaz.''')
    code('''RUNTIME = ROOT / '.runtime/deocclusion'
command([sys.executable, ROOT / 'scripts/prepare_runtime.py',
         '--repo', REPO, '--output', RUNTIME], 'prepare_runtime')
''')
    md('''## 6 — Gerçek ağırlıklarla örnek tahmin

Varsayılan **örnek 4**, resmî COCOA demosunun seçimiyle aynıdır. İsterseniz sonraki çalıştırmada
`EXAMPLE` değerini 1–5 arasında değiştirin. Her çalıştırma ayrı klasör üretir.
Örtülme sırası eşiği 0,1; amodal eşiği 0,5; maske girdi boyutu 256×256'dır.
Ağ çağrıları ve sonlu çıktılar denetlenir. Model yüklenemezse işlem durur; yedek rastgele tahmin yoktur.''')
    code('''EXAMPLE = 4
OUT = ROOT / 'outputs' / (SESSION + '_' + uuid.uuid4().hex[:6])
command([sys.executable, ROOT / 'scripts/run_pcnet_m.py',
         '--repo', REPO, '--runtime', RUNTIME, '--manifest', ROOT / 'weights/manifest.json',
         '--weights', WEIGHTS, '--example', EXAMPLE, '--device', 'cuda', '--output', OUT], 'inference', ENV)
run_report = json.loads((OUT / 'run.json').read_text())
assert run_report['status'] == 'success'
print('Gerçek tahmin doğrulandı. Ağ çağrısı:', run_report['model_forward_calls'])
''')
    md('''## 7 — Maskeleri görselleştir

İlk görsel: görünür girdi maskeleri, PCNet-M amodal tahmini ve referans amodal maskeler.
İkinci görsel: örtülen bir nesnenin ikili maskeleri. GT yalnızca görsel seçim/karşılaştırma için kullanılır.
Tahmin referansla aynı olmak zorunda değildir. Tek örnek araştırma başarısını veya veri kümesi metriğini göstermez.''')
    code('''from IPython.display import display, Image as DisplayImage
assert (OUT / 'run.json').exists(), 'Önce 6. hücreyi çalıştırın.'
assert json.loads((OUT / 'run.json').read_text())['status'] == 'success', 'Tahmin başarısız; 8. hücreden tanı indirin.'
display(DisplayImage(filename=str(OUT / 'comparison.png')))
display(DisplayImage(filename=str(OUT / 'object_comparison.png')))
''')
    md('''## 8 — Sonuçları veya hata kayıtlarını indir

Hata aldıysanız da bu hücreyi çalıştırabilirsiniz (1. hücrenin başlamış olması gerekir).
Başarı varsa ZIP: `comparison.png`, `object_comparison.png`, `input.png`, nesne başına PNG maskeler,
`masks.npz`, `run.json`, kullanılan scriptler ve kurulum kayıtları içerir.
PNG maskeler 0/255; NPZ maskeler 0/1 değerindedir. Sıra matrisi: `-1` satır nesnesinin sütun nesnesinin
arkasında olduğunu gösterir. Hata varsa ZIP adı `pcnet_m_diagnostics_...` olur.
Ağırlıklar ve bütün depo ZIP'e eklenmez.''')
    code('''import zipfile
from google.colab import files
success = (OUT / 'run.json').exists() and json.loads((OUT / 'run.json').read_text()).get('status') == 'success'
kind = 'results' if success else 'diagnostics'
archive = ROOT / f'pcnet_m_{kind}_{SESSION}.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for folder in [LOGS, ROOT / 'scripts', OUT]:
        if folder.exists():
            for path in sorted(folder.rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts and 'mpl-cache' not in path.parts:
                    z.write(path, path.relative_to(ROOT))
    for relative in ['weights/manifest.json', 'third_party/deocclusion.lock.json',
                     'requirements-colab.lock.txt', 'reports/colab-wheels.json',
                     '.runtime/deocclusion/runtime_manifest.json']:
        path = ROOT / relative
        if path.exists():
            z.write(path, relative)
print('İndiriliyor:', archive.name)
files.download(str(archive))
''')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
        'colab': {'name': '01_pcnet_m_colab.ipynb', 'provenance': []},
        'accelerator': 'GPU'})
    nbf.validate(notebook)
    destination = ROOT / 'notebooks/01_pcnet_m_colab.ipynb'
    destination.parent.mkdir(exist_ok=True)
    nbf.write(notebook, destination)
    print(destination)


if __name__ == '__main__':
    build()
