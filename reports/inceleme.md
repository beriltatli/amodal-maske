> Historical setup notes from 2026-09-07. For the current implementation and verified results, see the root README. The original CUDA-only restriction described below has been superseded by explicit CPU/CUDA device support.

# PCNet-M kaynak ve kurulum incelemesi — 7 Eylül 2026

## Sonuç

PCNet-M'nin resmî kodu ve **COCOA hazır ağırlığı erişilebilir**. Ağırlık dosyasının tamamı
indirildi. Bu oturumda CUDA tahmini çalıştırılmadığından uçtan uca başarı iddiası yoktur.
Şu anda ağırlık erişim engeli bulunmuyor; kalan doğrulama Colab'da ağırlık yükleme ve ileri geçiştir.

Klasör başlangıçta boştu. Üst dizinlerde uygulanabilir `AGENTS.md` bulunmadı.
Resmî depo `third_party/deocclusion` altında, değiştirilmemiş Git checkout olarak duruyor.
Commit `ac543f9a54f4cb8baf0774bdb5f8638eadf8b57c` (2021-05-26).
Kaynak: [sabit commit](https://github.com/XiaohangZhan/deocclusion/tree/ac543f9a54f4cb8baf0774bdb5f8638eadf8b57c).

## README, lisans ve demolar

[Resmî README](https://github.com/XiaohangZhan/deocclusion/blob/ac543f9a54f4cb8baf0774bdb5f8638eadf8b57c/README.md)
Python 3.7, PyTorch ≥0.4.1, COCO API ve `requirements.txt` tarif ediyor.
Üst düzey [lisans Apache-2.0](https://github.com/XiaohangZhan/deocclusion/blob/ac543f9a54f4cb8baf0774bdb5f8638eadf8b57c/LICENSE);
yerel kopyası korundu ve üretilen çalışma kopyasına eklendi. Drive dosyalarında ayrı bir ağırlık lisansı
belgesi tespit edilmedi. Depo lisansı ile örnek görüntü/veri kümesi haklarının kapsamı aynı varsayılmadı.

- `demos/demo_cocoa.ipynb`: beş gömülü COCOA örneğiyle çalışabilir. Resmî varsayılan örnek 4;
  480×640 görüntü ve 10 nesne. Görünür maskeler anotasyondan okunur; amodal GT karşılaştırma içindir.
  Notebook'ta PCNet-M dışında PCNet-C, GCA matting, içerik tamamlama ve veri kümesi hücreleri de var.
  Başlangıç notebook'umuz yalnızca PCNet-M adımlarını kullanır.
- `demos/demo_kins.ipynb`: harici KINS/KITTI dosyaları ve ayrıca içerik/matting modelleri bekler.
  Küçük, verisi depoda olan başlangıç için seçilmedi.
- `demos/explore_manpillary.ipynb`: veri keşfi için yardımcı notebook; COCOA başlangıç akışı değil.

## Yapılandırmalar

COCOA/KINS/LVIS `pcnet_m/config.yaml` dosyalarının üçü de `PartialCompletionMask`,
`unet2`, `use_rgb=False`, iki giriş kanalı, iki çıktı sınıfı ve 256 maske girdi boyutu kullanıyor.
İki kanal hedef nesnenin görünür maskesi ve örten maskedir. RGB ağ girdisi değildir.
Farklı veri kümesi yolları ve eğitim çizelgeleri vardır (56.000 / 32.000 / 168.000 iterasyon);
bu çizelgeler **çalıştırılmadı**. PCNet-C yapılandırmaları ayrı `PConvUNet` içerik tamamlama modelleridir.

COCOA örnek tahmini resmî demo ile aynı kutu genişletmeyi (3,0 / 1,5), sıra eşiğini (0,1),
amodal eşiğini (0,5), sıralamada nearest ve tamamlamada linear yeniden boyutlandırmayı kullanır.
Tahmine ayrıca görünür maskeyle birleşim veya alan düzeltmesi eklenmedi.

## Gerçek ağırlık indirmesi

Kaynak: [README'nin bağladığı resmî Drive klasörü](https://drive.google.com/drive/folders/1O89ItVWucCoL_VxIbLM1XLxr9JFfyj_Y).

| Dosya | Drive dosya kimliği | Bu oturumdaki doğrulama |
|---|---|---|
| COCOA_pcnet_m.pth.tar | `12T_PKk5vB59bbLRil7dJrlAP6oJZipny` | **Tam indirme başarılı** |
| KINS_pcnet_m.pth.tar | `1yJ4RYCEnavuJkFbyvonIM4BnaxWhVulN` | Listelendi; indirilmedi |
| LVIS_pcnet_m.pth.tar | `1y8-mh5gC8VTvxlda36Tw45QSPF7BZ90n` | Listelendi; indirilmedi |

COCOA için HTTP 303 yönlendirmesinden sonra HTTP 200, `application/octet-stream`,
dosya adı ve `Content-Length: 26870559` alındı. `curl` çıkış kodu 0;
yerel dosya boyutu başlıkla aynı. Yaklaşık 26,9 MB (25,6 MiB).
SHA-256: `3740e8f709f824068dd36be46bd44ffd18105c4f606c099789c93f3ae4b860f7`.
Bu özet resmî indirmeden yerelde hesaplandı; bağımsız yayıncı özeti değildir.

PyTorch kurulmadan `pickletools` ile ilk beş pickle akışı incelendi; `state_dict`, `step`,
`module.inc...` anahtarları, FloatStorage/LongStorage görüldü. Pickle çalıştırılmadı.
Bu kontrol `torch.load` veya model eşleşmesi doğrulaması sayılmaz.
Dosya `.tar` uzantısına rağmen eski PyTorch serileştirmesidir; arşiv gibi açılmamalı.
Kanıtlar: `weights/manifest.json`, `reports/cocoa-download-headers.txt` ve yerel dosya.

## Uyumluluk yaklaşımı

[Resmî bağımlılıklar](https://github.com/XiaohangZhan/deocclusion/blob/ac543f9a54f4cb8baf0774bdb5f8638eadf8b57c/requirements.txt):
OpenCV, PyYAML, imutils, cvbase (iki kez), torchvision, tensorboardX, networkx,
scikit-image, `pillow==6.1`, tqdm. Bunların tamamını modern Colab'a körlemesine kurmuyoruz.

| Bulgu | Başlangıç akışındaki karşılığı |
|---|---|
| Eski Pillow pini / Python 3.7 tarifi | Python 3.12 için ayrı destek paketi kilidi; eski ortamı kurma iddiası yok |
| `np.int`, `np.bool` | Üretilen `inference.py` içinde 6 `int`, 2 `bool` değişimi; NumPy'ya global monkey patch yok |
| `yaml.load(f)` | Kendi yükleyicimizde `yaml.safe_load` |
| Orijinal `load_state` CUDA'ya yükler ve `strict=False` kullanır | Önce CPU'ya `weights_only=True`; tam anahtar eşleşmesi ve `strict=True`, ardından CUDA |
| Eğitim ve içerik modeli import zincirleri | Resmî `unet2` ve `FixModule` doğrudan yüklenir; optimizer kurulmaz |
| Veri okuyucuda cvbase import'u | `read_COCOA` fonksiyonu AST ile resmî kaynaktan aynen alınır; yalnızca örnek okuyucu kullanılır |
| Resmî kodda doğrudan `.cuda()` | Korunur; NVIDIA CUDA kontrolü yapılır, MPS'ye sessiz geçiş yok |

UNet katmanları, ileri geçiş, `read_COCOA`, `expand_bbox` ve tahmin algoritması korunur.
AST karşılaştırması, tahmin kodundaki tek algoritma kaynağı değişikliğinin eski tür takma adları
olduğunu yerelde kontrol etti. Kaynak SHA-256 değerleri `.runtime/deocclusion/runtime_manifest.json` içinde.
Bu dar uyumluluk katmanı **eğitim desteği değildir**.

Kaynaklar: [NumPy 1.24 kaldırılan takma adlar](https://numpy.org/doc/stable/release/1.24.0-notes.html),
[PyYAML belgeleri](https://pyyaml.org/wiki/PyYAMLDocumentation),
[PyTorch serileştirme belgeleri](https://docs.pytorch.org/docs/stable/notes/serialization.html).

## Colab sürümü ve doğrulama sınırı

7 Eylül 2026'da erişilen [resmî Colab sürüm listesi](https://research.google.com/colaboratory/runtime-version-faq.html)
`2026.07` için Ubuntu 22.04.5, Python 3.12.13, NumPy 2.0.2, PyTorch 2.11.0 listeliyor.
Bu **canlı bir kullanıcı Colab oturumunun ölçümü değildir**; notebook 1. hücre gerçek sürümleri ölçer.
Notebook bu adayı bekler ve farklı sürümde durur. Eski PyTorch/CUDA indirme komutu içermez.

`pip --dry-run --ignore-installed --only-binary=:all: --python-version 3.12
--platform manylinux2014_x86_64 --implementation cp --abi cp312` ile yardımcı paketlerin
bağımlılıkları çözüldü: 20 paket, Linux/Python 3.12 wheel adayları bulundu.
Sonuç `colab-dependency-resolution.json`, kesin URL/özetler `colab-wheels.json`, kurulum kilidi
`../requirements-colab.lock.txt`. Bu **paket çözümü** kontrolüdür; Linux ikili modül import'u,
CUDA uyumu veya GPU tahmini yerine geçmez. PyTorch Colab'dan gelir, bu kilidin parçası değildir.
Çalışma zamanı kalıcı değildir; tekrar üretim için başarılı koşunun `run.json`/paket listesini saklayın.

Mac'te Python 3.14.7, NumPy 2.5.3, Pillow 12.3.0, Matplotlib 3.11.1 ve nbformat 5.11.1
ayrı `.venv` içine kuruldu. Gerek kalmayan indirme paketleri kaldırıldı. `pip check` geçti.
Sistem Python'u, kullanıcı Git ayarları ve sistem paketleri değiştirilmedi.
İlk Git denemesi mevcut HTTPS→SSH yönlendirmesinde takıldı; yalnızca klon komutunda
`GIT_CONFIG_GLOBAL=/dev/null` kullanılarak HTTPS ile tamamlandı.

## Çalıştırılan / bekleyen

- **Geçti:** Git commit/temizlik; tam COCOA dosya indirmesi ve boyut/özet;
  hatalı indirme reddi; yardımcı paket metadata çözümü; notebook şeması/sözdizimi;
  uyumluluk kopyası kapsamı; beş resmî JPG/JSON okunması ve görsel önizleme; Mac paket kontrolü.
- **Çalıştırılmadı:** Colab kurulumu; gerçek `torch.load`; model anahtar/şekil eşleşmesi;
  CUDA ileri geçişi ve örnek maskeleri; Colab ZIP indirme arayüzü.
- **Kapsam dışında:** eğitim, tam veri kümeleri, ücretli GPU ve araştırma deneyleri.

PCNet-M erişilemez hale gelirse önce ağırlık/kurulum hatası raporlanacak.
[EfficientSAM-Ti](https://github.com/yformer/EfficientSAM) standart ağırlıkları amodal değildir;
ona geçiş amodal eğitim verisi, ince ayar ve ayrı başlangıç değerlendirmesi gerektirir.
Bu projede alternatif modele geçilmedi ve bu deneyler uygulanmadı.
