# Sentiment Classification + Aspect Extraction (SemEval-2014 Restaurants)

Yerel bir LLM (Qwen3 1.7B, Ollama) ile İngilizce restoran yorumlarından **yapılandırılmış JSON** üreten bir sistem. Model her yorum için iki iş yapar:

- **Sınıflandırma (classification):** Yorumun değerlendirilen şeye yönelik duygusu: `positive` veya `negative`
- **Bilgi çıkarma (information extraction):** Değerlendirilen şeyin kendisi (aspect), metinden birebir kopyalanarak

```
"The pizza was delicious."  →  {"sentiment": "positive", "aspect": "pizza"}
```

Nihai sistem JSON'a kısa bir gerekçe alanı da ekler (bkz. [Nihai sistem](#nihai-sistem)).

## Sonuçlar (test seti, 120 örnek)

| Sistem | Ayar | Ham JSON (`json.loads`) | Temizleme sonrası | Sentiment accuracy | Aspect exact match | Joint accuracy | Süre / örnek |
|---|---|---|---|---|---|---|---|
| v2 (başlangıç) | prompt-only | %100 | %100 | %94,2 | %90,0 | %84,2 | 0,72 sn |
| **v4 Reason (nihai)** | JSON şeması | **%100** | **%100** | **%99,2** | **%93,3** | **%92,5** | 1,40 sn |
| v4 Reason | prompt-only | %100 | %100 | %99,2 | %93,3 | %92,5 | 1,63 sn |
| v5 CoT (önce metin, sonra JSON) | prompt-only | **%0** | %100 | %99,2 | %92,5 | %91,7 | 1,89 sn |

*Joint accuracy:* sentiment ve aspect'in ikisinin birden doğru olduğu örneklerin oranı. Bütün deneylerin ayrıntılı sonuçları ve grafikleri [analysis.ipynb](analysis.ipynb) içindedir.

## Prompt-only ve şema kısıtlı çıktı karşılaştırması

Ödevin iki ayarı: **prompt-only** (JSON yalnızca prompt'ta istenir, Ollama `format` parametresi kullanılmaz) ve **şema kısıtlı** (`format` ile JSON şeması verilir).

| Sistem | Ayar | Ham JSON | Temizleme sonrası | Şema geçerliliği | Sentiment | Aspect exact |
|---|---|---|---|---|---|---|
| v2 (ayrıntılı prompt) | prompt-only | %100 | %100 | %100 | 0,942 | 0,900 |
| v2 | şema | %100 | %100 | %100 | 0,942 | 0,900 |
| v4 Reason | prompt-only | %100 | %100 | %100 | 0,992 | 0,933 |
| v4 Reason | şema | %100 | %100 | %100 | 0,992 | 0,933 |
| weak (3 satırlık prompt) | prompt-only | **%35** | %100 | %100 | 0,992 | 0,525 |
| weak | şema | %100 | %100 | %100 | 0,992 | 0,525 |

- **Ayrıntılı prompt'ta** (`v2`, `v4_reason`) iki ayarın 120 çıktısı da birebir aynı. Model zaten temiz JSON üretiyor, şema yalnızca format garantisi ekliyor.
- **Kısa prompt'ta** (`weak`) model çıktıyı çoğunlukla ` ```json ` ile sarıyor: 120 çıktının 78'i (%65) temizleme aşamasında kurtarıldı, 42'si doğrudan geçerliydi. Şema bu sorunu ortadan kaldırıyor.
- **Doğruluk:** Şema, görev doğruluğunu değiştirmiyor (weak'te de aynı). Şema yalnızca geçersiz token'ları yasaklar, model zaten aynı token'ları seçtiği için içerik değişmez.

## Veri seti ve gold etiketler

Kaynak: [tomaarsen/setfit-absa-semeval-restaurants](https://huggingface.co/datasets/tomaarsen/setfit-absa-semeval-restaurants) (SemEval-2014 Task 4, Pontiki ve ark., 2014). Veri setinin test bölümünde etiket bulunmadığı için bütün örnekler **train** bölümünden seçildi.

**Gold etiketler** (aspect ifadesi ve duygu) SemEval'in uzman anotasyonundan alındı, elle etiketleme yapılmadı. Elle yapılan iş **örnek seçimidir**.

Seçim adımları ([prepare_data.py](prepare_data.py)):

1. Aynı cümleye ait satırlar gruplandı (birden çok aspect içeren cümleler birden çok satırda tekrarlanıyor).
2. Yalnızca **tek aspect'li** ve etiketi **positive/negative** olan cümleler alındı (599 positive, 214 negative aday).
3. Adaylar elle incelendi. 170 cümle elendi: örtük duygu (61), birden çok hedef (45), karışık duygu (27), etiket uyumsuzluğu (14), yarım metin (9), ironi (8), anlamsız aspect (6). Her biri gerekçesiyle [data/excluded_samples.json](data/excluded_samples.json) dosyasında.
4. Sabit seed (42) ile karıştırılan havuzdan iki ayrı set oluşturuldu:

| Set | Dosya | Örnek | Kullanım |
|---|---|---|---|
| Test | `data/absa_120.json` | 60 positive + 60 negative | Deneyler ve nihai sonuç |
| Dev | `data/absa_dev_80.json` | 40 positive + 40 negative | İyileştirme yöntemlerinin seçimi (test setiyle ortak cümle yok) |

### Eşleştirme kuralı

Aspect karşılaştırması ([evaluate.py](evaluate.py)): metinler `str.lower()` ile küçültülür ve boşluklar tek boşluğa indirilir.

- **Exact match (ana metrik):** normalize edilmiş iki aspect birebir aynı.
- **Partial match:** biri diğerini içeriyor (örnek: `pizza` ve `mushroom pizza`).

Her cümlede tek gold aspect var ve model tek aspect üretiyor. Bu yüzden entity F1 (exact), aspect exact-match oranıyla aynı çıkar. Veri İngilizce olduğu için Türkçe'ye özgü `İ`/`I` dönüşümü gerekmedi.

## Klasör yapısı

```
.
├── prepare_data.py          Veri setini indirir, filtreler, test ve dev setlerini oluşturur
├── model.py                 Ollama API çağrıları (JSON şeması, düşünme modu, parametreler)
├── parser.py                JSON ayrıştırma: doğrudan → Markdown/<think> temizliği → { } arasını çıkarma
├── run.py                   Bir modeli veri setindeki bütün örneklerde çalıştırır, sonuçları kaydeder
├── run_pipeline.py          İki aşamalı yöntem (yalnızca dev seti deneyi için)
├── evaluate.py              Metrikler ve hata analizi
├── baseline_tfidf.py        Klasik temel çizgi: BoW / TF-IDF + lojistik regresyon (yalnızca sentiment)
├── learning_curve_tfidf.py  Klasik temel çizginin öğrenme eğrisi (10-272 etiketli örnek)
├── exp_prompt_length.py     Few-shot sayısı ve gecikme deneyi
├── plot_prompt_length.py    Bu deneyin grafiği
├── exp_tokenization.py      Tokenizer deneyi (Hugging Face tokenizer ve Ollama token sayısı)
├── analysis.ipynb           Ana deneylerin grafikleri ve yorumları
├── HANDOFF.md               Oturum notları ve kalan işler
├── data/                    Test seti, dev seti, elenen örnekler
├── modelfiles/              Ollama Modelfile'ları (prompt + parametreler)
└── results/
    ├── test/                Test seti sonuçları (120 örnek)
    ├── dev/                 Dev seti sonuçları (80 örnek)
    └── *.txt, *.json, *.png Baseline, öğrenme eğrisi, gecikme ve tokenizer deneylerinin çıktıları
```

## Kurulum

Gereksinimler: Python 3.12, [Ollama](https://ollama.com) (0.40.16 ile test edildi, Apple M4). Bulut API'si kullanılmaz, her şey yerelde çalışır.

```bash
# Python paketleri (proje kökündeki .venv kullanıldı)
pip install requests datasets pandas numpy matplotlib ipykernel scikit-learn transformers

# Temel modeller
ollama pull qwen3:1.7b-q4_K_M
ollama pull qwen3:1.7b-q8_0     # yalnızca Q8 deneyi için
ollama pull qwen3:4b            # yalnızca model boyutu deneyi için
ollama pull gemma3:4b           # yalnızca model boyutu deneyi için

# Nihai sistemin modeli (proje kökünden)
ollama create absa-model-v4-reason -f modelfiles/Modelfile_v4_reason
```

Diğer deneylerin modelleri aynı şekilde oluşturulur:

| Ollama modeli | Modelfile | Açıklama |
|---|---|---|
| `absa-model-v1` | `Modelfile_v1` | Ayrıntılı prompt, önce sentiment sonra aspect |
| `absa-model-v2` | `Modelfile_v2` | Ayrıntılı prompt, önce aspect sonra sentiment (ana sistem) |
| `absa-model-v2-q8` | `Modelfile_v2_q8` | v2, `qwen3:1.7b-q8_0` ile |
| `absa-model-weak` | `Modelfile_weak` | 3 satırlık kısa prompt (parser stres testi) |
| `absa-model-v3` | `Modelfile_v3` | Kısa prompt + aspect kuralları |
| `absa-model-v4-fewshot` | `Modelfile_v4_fewshot` | v2 + 5 örnek |
| `absa-model-v4-reason` | `Modelfile_v4_reason` | v2 + JSON içinde gerekçe alanı (**nihai**) |
| `absa-model-v5-cot` | `Modelfile_v5_cot` | v2 + önce serbest metin akıl yürütme, sonra JSON (şemasız) |
| `absa-stage-aspect` | `Modelfile_stage_aspect` | İki aşamalı yöntemin 1. aşaması |
| `absa-stage-sentiment` | `Modelfile_stage_sentiment` | İki aşamalı yöntemin 2. aşaması |
| `absa-model-v2-qwen3-4b`, `absa-model-v4-reason-qwen3-4b` | `Modelfile_v2_qwen3-4b`, `Modelfile_v4_reason_qwen3-4b` | v2 ve v4 Reason, `qwen3:4b` ile |
| `absa-model-v2-gemma3-4b`, `absa-model-v4-reason-gemma3-4b` | `Modelfile_v2_gemma3-4b`, `Modelfile_v4_reason_gemma3-4b` | v2 ve v4 Reason, `gemma3:4b` ile |

Modelfile'lar hazır bir modeli sistem prompt'u ve parametrelerle yapılandırır. **Hiçbir model eğitilmedi veya ince ayar yapılmadı.**

## Tekrarlanabilirlik

| Parametre | Değer |
|---|---|
| Model | `qwen3:1.7b-q4_K_M` (Q4_K_M nicemleme, `ollama show` ile doğrulanır) |
| Ollama sürümü | 0.40.16 |
| Donanım | Apple M4 |
| Temperature / seed | 0 / 42 |
| Context length (`num_ctx`) | 4096 (gecikme deneyinde 8192) |
| `num_predict` | 128 (çoğu sistem), 256 (v4 Reason), 384 (v5 CoT), 2048 (düşünme modu) |

Temperature 0'da aynı çalıştırma tekrarlandığında ve `--seed 7` ile çalıştırıldığında 120 örneğin hepsinde ham çıktı birebir aynı çıktı. Bu yalnızca bu makine ve Ollama sürümü için gözlemdir, genel bir garanti değildir.

## Çalıştırma

Bütün komutlar proje kökünden çalıştırılır. Ollama açık olmalıdır.

```bash
# 1. Veri setlerini oluştur (data/ klasöründe hazır olduğu için isteğe bağlı)
python prepare_data.py

# 2. Nihai sistemi test setinde çalıştır (şema kısıtlı) ve prompt-only karşılaştırması
python run.py absa-model-v4-reason results/test/results_v4_reason.json --schema --reason
python run.py absa-model-v4-reason results/test/results_v4_reason_noschema.json --reason

# 3. Değerlendir (birden çok sonuç dosyası yan yana karşılaştırılabilir)
python evaluate.py results/test/results_v2_schema.json results/test/results_v4_reason.json

# 4. Önce akıl yürütme, sonra JSON (şemasız)
python run.py absa-model-v5-cot results/test/results_v5_cot.json

# 5. Klasik temel çizgi ve öğrenme eğrisi
python baseline_tfidf.py
python learning_curve_tfidf.py

# 6. Few-shot sayısı ve gecikme deneyi, grafik
python exp_prompt_length.py
python plot_prompt_length.py

# 7. Tokenizer deneyi (ilk çalıştırmada Hugging Face'ten tokenizer indirir)
python exp_tokenization.py

# 8. Grafikler: analysis.ipynb dosyasını açıp "Run All"
#    (9. bölüm modeli canlı çağırır, Ollama açık olmalı)
```

`run.py` seçenekleri:

| Seçenek | Anlamı |
|---|---|
| `--data <dosya>` | Veri dosyası (varsayılan: `data/absa_120.json`) |
| `--schema` | Ollama'ya JSON şeması gönderir (constrained decoding) |
| `--reason` | Çıktıda `reason` alanı bekler. `--schema` ile birlikte gerekçe şemasını kullanır |
| `--think` | Qwen3 düşünme modunu açar (`--num-predict 2048` ile kullanılmalı) |
| `--temperature`, `--seed`, `--num-predict` | Modelfile'daki parametreleri bu çalıştırma için değiştirir |

Her çalıştırma ölçümden önce bir ısınma (warm-up) isteği gönderir, böylece modelin belleğe yüklenme süresi süre ölçümüne dahil olmaz.

## Nihai sistem

**v4 Reason + Qwen3 1.7B Q4 + temperature 0 + JSON şeması.** Model, sentiment'e karar vermeden önce tek cümlelik bir gerekçe yazar. JSON şeması alanların sırasını (aspect → reason → sentiment) zorunlu kılar:

```json
{
  "aspect": "meal",
  "reason": "The reviewer says they have never had a disappointing meal here.",
  "sentiment": "positive"
}
```

Gerekçe puanlanmaz. Değerlendirme yalnızca `sentiment` ve `aspect` alanlarını gold etiketlerle karşılaştırır.

Seçim süreci: dev setinde 4 iyileştirme yöntemi (Q8, few-shot, gerekçe alanı, iki aşamalı çağrı) karşılaştırıldı. Gerekçe alanı ile iki aşamalı yöntem joint accuracy'de berabere kaldı. Gerekçe alanı tek çağrı yaptığı ve yorumlanabilir olduğu için seçildi. Seçilen yöntem test setinde **yalnızca bir kez** çalıştırıldı (şemalı). Prompt-only karşılaştırması sonradan eklendi ve aynı çıktıları verdi.

## Yapılan deneyler

| Deney | Soru | Ana bulgu |
|---|---|---|
| v1 ile v2 | Görev sırası önemli mi? | Fark 1 örnek, etkisi yok |
| Kısa prompt (weak) | Parser işe yarıyor mu? | İlk deneme başarısı %35'e düştü (78 çıktı ` ```json ` ile sarılmıştı), parser %100'e çıkardı |
| JSON şeması | Format zorlanabilir mi? | %100 geçerli JSON, içerik ve doğruluk değişmiyor |
| Prompt-only ve şema, v4 Reason | Şema doğruluğu etkiler mi? | 120 çıktının hepsi aynı, fark yalnızca süre (1,63 sn ve 1,40 sn) |
| weak / v3 | Prompt ayrıntısı neyi etkiliyor? | Ayrıntılı kurallar aspect'i iyileştiriyor, sentiment'te negatif eğilim yaratıyor |
| Q4 ile Q8 | Nicemleme etkisi? | Test setinde sentiment +4 örnek (McNemar p = 0,22), dev setinde iyileşme yok. Q8: bellekte 1,5 GB yerine 2,1 GB (+%40), %43 daha yavaş |
| Temperature 0–1,0 | Rastgeleliğin etkisi? | Doğruluk farkı seed gürültüsü içinde (sentiment 0,931–0,942). JSON her durumda %100 |
| Temperature 0 tekrarlanabilirlik | Aynı prompt iki kez aynı sonucu verir mi? | Tekrarda ve farklı seed'de 120/120 ham çıktı birebir aynı |
| Düşünme modu | Akıl yürütme yardımcı mı? | Sentiment 0,958, aspect 0,842, 3/120 boş cevap, 19 kat daha yavaş |
| Gerekçe alanı ve v5 CoT | Önce akıl yürütme doğruluğu etkiler mi? | Sentiment 0,942 → 0,992, joint 0,842 → 0,925 (alan içinde) ve 0,917 (önce metin). v5 CoT'ta ham JSON %0, parser'ın `extracted` aşaması %100'e çıkardı. Sentiment kazancı sınırda anlamlı (McNemar p = 0,07) |
| Model boyutu | Daha büyük model? | qwen3:4b joint 0,925 (v2) ve 0,950 (Reason), 1,7 ve 3,0 sn. gemma3:4b joint 0,858, 4,2 ve 5,7 sn. Reason alanı 1,7B'de işe yarıyor, 4B'de kazanç az veya yok |
| Klasik temel çizgi | LLM, TF-IDF ve LR'a karşı? | TF-IDF + LR: 80 örnekle %67–72, 10 örnekle %56. LLM eğitimsiz %94–99. LR tahmini 0,008 ms/örnek |
| Öğrenme eğrisi | Klasik model kaç örnekle yaklaşır? | 272 örnekle %64, LLM'e yaklaşmıyor. Ek veri test setinden farklı dağılımda (elenen cümleler), bu yüzden saf "daha çok veri" etkisi değil |
| Few-shot ve gecikme | Prompt uzunluğu süreyi nasıl etkiler? | k = 0–40 (428–1703 token): soğuk prefill 1,4 sn → 5,9 sn (~3,5 ms/token, neredeyse doğrusal), decode 35 → 39 ms/token. Önbellekli (sıcak) istekler k'dan bağımsız ~0,7 sn. Grafik: `results/prompt_length_experiment.png` |
| Tokenizer | Türkçe neden daha çok token? | `kitaplarımızdan` → `kit \| ap \| larımız \| dan` (4 token), `books` tek token. 3 Türkçe/İngilizce çiftte oran 1,34. Hugging Face sayıları Ollama `prompt_eval_count` ile birebir aynı |
| Dev seti yöntemleri | En iyi iyileştirme? | Gerekçe alanı (sentiment %98,8) |

Hata analizleri (confusion matrix, olumsuzluk ifadeleri, yorum uzunluğu, aspect hata türleri, halüsinasyon, ironi) için [analysis.ipynb](analysis.ipynb).

## Sınırlamalar

- **Seçilmiş veri:** Yalnızca tek aspect'li, açık ifadeli cümleler kullanıldı. Karışık duygu ve çoğu ironi içeren cümleler elendi. Görev orijinal SemEval görevinden daha kolaydır. Elle eleme tek kişi tarafından yapıldı.
- **Küçük örneklemler:** Bazı analizler çok az örneğe dayanır (olumsuzluk 5 cümle, ironi 8 cümle, gecikme deneyinde 30 örnek ve her k için tek soğuk ölçüm, tokenizer deneyinde 3 cümle çifti). Bu sonuçlar gözlem niteliğindedir. Sentiment farklarının çoğu istatistiksel olarak sınırda veya anlamsızdır.
- **Test setinde keşif:** weak, v3, Q8, temperature, düşünme modu, v5 CoT ve model boyutu deneyleri test setinde yapıldı. Nihai yöntem dev setinde seçildi, ama yöntemin fikri test setinin hata analizinden doğdu.
- **Tek model ailesi (ana sistem):** Ana deneyler Qwen3 1.7B ile yapıldı. Yalnızca model boyutu deneyinde qwen3:4b ve gemma3:4b denendi.
- **Klasik temel çizgi:** Yalnızca sentiment'i yapar, aspect çıkaramaz. Öğrenme eğrisindeki ek eğitim verisi test setinden farklı dağılımdadır.
- **Gecikme yorumları:** Dikkat (attention) maliyeti ve KV cache açıklamaları model boyutlarından yapılan hesaplara dayanır. Prefill'in O(n²) bileşeni bu uzunluklarda ölçüm gürültüsünden ayırt edilemedi. KV cache'in fp16 olduğu varsayıldı, doğrulanmadı.
- **Düşünme modunda boş cevaplar:** `num_predict` bütçesinin tükenmesi tutarlı bir açıklama, ama `done_reason` kaydedilmediği için doğrulanmadı.
