# Sentiment Classification + Aspect Extraction (SemEval-2014 Restaurants)

Yerel bir LLM (Qwen3 1.7B, Ollama) ile İngilizce restoran yorumlarından **yapılandırılmış JSON** üreten bir sistem. Model her yorum için iki iş yapar:

- **Sınıflandırma (classification):** Yorumun değerlendirilen şeye yönelik duygusu: `positive` veya `negative`
- **Bilgi çıkarma (information extraction):** Değerlendirilen şeyin kendisi (aspect), metinden birebir kopyalanarak

```
"The pizza was delicious."  →  {"sentiment": "positive", "aspect": "pizza"}
```

Nihai sistem JSON'a kısa bir gerekçe alanı da ekler (bkz. [Nihai sistem](#nihai-sistem)).

## Sonuçlar (test seti, 120 örnek)

| Sistem | JSON ilk deneme | Sentiment accuracy | Aspect exact match | Joint accuracy | Süre / örnek |
|---|---|---|---|---|---|
| v2 (başlangıç) | %100 | %94,2 | %90,0 | %84,2 | 0,72 sn |
| **v4 Reason (nihai)** | **%100** | **%99,2** | **%93,3** | **%92,5** | 1,40 sn |

*Joint accuracy:* sentiment ve aspect'in ikisinin birden doğru olduğu örneklerin oranı. Bütün deneylerin ayrıntılı sonuçları ve grafikleri [analysis.ipynb](analysis.ipynb) içindedir.

## Veri seti

Kaynak: [tomaarsen/setfit-absa-semeval-restaurants](https://huggingface.co/datasets/tomaarsen/setfit-absa-semeval-restaurants) (SemEval-2014 Task 4, Pontiki ve ark., 2014). Veri setinin test bölümünde etiket bulunmadığı için bütün örnekler **train** bölümünden seçildi.

Seçim adımları ([prepare_data.py](prepare_data.py)):

1. Aynı cümleye ait satırlar gruplandı (birden çok aspect içeren cümleler birden çok satırda tekrarlanıyor).
2. Yalnızca **tek aspect'li** ve etiketi **positive/negative** olan cümleler alındı (599 positive, 214 negative aday).
3. Adaylar elle incelendi. Örtük duygu, karışık duygu, ironi, birden çok hedef, anlamsız aspect veya yarım metin içeren 170 cümle elendi. Her biri gerekçesiyle [data/excluded_samples.json](data/excluded_samples.json) dosyasında.
4. Sabit seed (42) ile karıştırılan havuzdan iki ayrı set oluşturuldu:

| Set | Dosya | Örnek | Kullanım |
|---|---|---|---|
| Test | `data/absa_120.json` | 60 positive + 60 negative | Deneyler ve nihai sonuç |
| Dev | `data/absa_dev_80.json` | 40 positive + 40 negative | İyileştirme yöntemlerinin seçimi (test setiyle ortak cümle yok) |

## Klasör yapısı

```
semeval_absa/
├── prepare_data.py     Veri setini indirir, filtreler, test ve dev setlerini oluşturur
├── model.py            Ollama API çağrıları (JSON şeması, düşünme modu, parametreler)
├── parser.py           JSON ayrıştırma: doğrudan → Markdown/<think> temizliği → { } arasını çıkarma
├── run.py              Bir modeli veri setindeki bütün örneklerde çalıştırır, sonuçları kaydeder
├── run_pipeline.py     İki aşamalı yöntem (yalnızca dev seti deneyi için)
├── evaluate.py         Metrikler ve hata analizi
├── analysis.ipynb      Bütün deneylerin grafikleri ve yorumları
├── data/               Test seti, dev seti, elenen örnekler
├── modelfiles/         Ollama Modelfile'ları (prompt + parametreler)
└── results/
    ├── test/           Test seti sonuçları (120 örnek)
    └── dev/            Dev seti sonuçları (80 örnek)
```

## Kurulum

Gereksinimler: Python 3.12, [Ollama](https://ollama.com) (0.40 ile test edildi). Bulut API'si kullanılmaz, her şey yerelde çalışır.

```bash
# Python paketleri (proje kökündeki .venv kullanıldı)
pip install requests datasets pandas numpy matplotlib ipykernel

# Temel modeller
ollama pull qwen3:1.7b-q4_K_M
ollama pull qwen3:1.7b-q8_0     # yalnızca Q8 deneyi için

# Nihai sistemin modeli
cd semeval_absa
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
| `absa-model-v4-reason` | `Modelfile_v4_reason` | v2 + gerekçe alanı (**nihai**) |
| `absa-stage-aspect` | `Modelfile_stage_aspect` | İki aşamalı yöntemin 1. aşaması |
| `absa-stage-sentiment` | `Modelfile_stage_sentiment` | İki aşamalı yöntemin 2. aşaması |

## Çalıştırma

Bütün komutlar `semeval_absa/` klasöründen çalıştırılır. Ollama açık olmalıdır.

```bash
# 1. Veri setlerini oluştur (data/ klasöründe hazır olduğu için isteğe bağlı)
python prepare_data.py

# 2. Nihai sistemi test setinde çalıştır
python run.py absa-model-v4-reason results/test/results_v4_reason.json --schema --reason

# 3. Değerlendir (birden çok sonuç dosyası yan yana karşılaştırılabilir)
python evaluate.py results/test/results_v2.json results/test/results_v4_reason.json

# 4. Grafikler: analysis.ipynb dosyasını açıp "Run All"
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

Seçim süreci: dev setinde 4 iyileştirme yöntemi (Q8, few-shot, gerekçe alanı, iki aşamalı çağrı) karşılaştırıldı. Gerekçe alanı ile iki aşamalı yöntem joint accuracy'de berabere kaldı. Gerekçe alanı tek çağrı yaptığı ve yorumlanabilir olduğu için seçildi. Seçilen yöntem test setinde **yalnızca bir kez** çalıştırıldı.

## Yapılan deneyler

| Deney | Soru | Ana bulgu |
|---|---|---|
| v1 ile v2 | Görev sırası önemli mi? | Fark 1 örnek, etkisi yok |
| Kısa prompt (weak) | Parser işe yarıyor mu? | İlk deneme başarısı %35'e düştü, parser %100'e çıkardı |
| JSON şeması | Format zorlanabilir mi? | %100 geçerli JSON, içerik ve hız değişmiyor |
| weak / v3 | Prompt ayrıntısı neyi etkiliyor? | Ayrıntılı kurallar aspect'i iyileştiriyor, sentiment'te negatif eğilim yaratıyor |
| Q4 ile Q8 | Nicemleme etkisi? | Test setinde sentiment +4 örnek, ama dev setinde iyileşme yok. Q8 %40 daha fazla bellek, %43 daha yavaş |
| Temperature 0–1,0 | Rastgeleliğin etkisi? | Neredeyse etkisiz. JSON her durumda %100 |
| Düşünme modu | Akıl yürütme yardımcı mı? | Sentiment ↑, aspect ↓, %2,5 boş cevap, 19 kat daha yavaş |
| Dev seti yöntemleri | En iyi iyileştirme? | Gerekçe alanı (sentiment %98,8) |

Hata analizleri (confusion matrix, olumsuzluk ifadeleri, yorum uzunluğu, aspect hata türleri, halüsinasyon, ironi) için [analysis.ipynb](analysis.ipynb).

## Sınırlamalar

- **Seçilmiş veri:** Yalnızca tek aspect'li, açık ifadeli cümleler kullanıldı. Görev orijinal SemEval görevinden daha kolaydır. Elle eleme tek kişi tarafından yapıldı.
- **Küçük örneklemler:** Bazı analizler çok az örneğe dayanır (olumsuzluk 5 cümle, ironi 6 cümle). Bu sonuçlar gözlem niteliğindedir.
- **Test setinde keşif:** weak, v3, Q8, temperature ve düşünme modu deneyleri test setinde yapıldı. Nihai yöntem dev setinde seçildi, ama yöntemin fikri test setinin hata analizinden doğdu.
- **Tek model ailesi:** Bütün deneyler Qwen3 1.7B ile yapıldı.
