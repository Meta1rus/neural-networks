# Тональность русскоязычных отзывов: RNN, LSTM, GRU

Финальный проект по дисциплине «Нейронные сети» (преподаватель — Селетков И. П.).
Подольский Филипп, магистратура ПМИ, ФМИ ПГНИУ.

Развитие второй части практики 10 «RNN, LSTM»: классификация тональности русскоязычных текстов
(NEUTRAL / POSITIVE / NEGATIVE) на наборе [`Megnis/ru_sentiment_dataset-50000`](https://huggingface.co/datasets/Megnis/ru_sentiment_dataset-50000).
В тетрадке: разведочный анализ, очистка от повторов, базовые модели (TF-IDF + логистическая регрессия,
случайный лес), сравнение RNN / LSTM / GRU / BiLSTM / BiGRU, размера сети, оптимизаторов и регуляризации,
перенос обучения (эмбеддинги и дообучение `cointegrated/rubert-tiny2`), итоговая оценка на тесте, разбор
ошибок, t-SNE скрытых состояний и сравнение с опубликованными на HuggingFace результатами.

## Состав

| Файл | Что это |
|---|---|
| `project_rusentiment_rnn.ipynb` | тетрадка проекта (все эксперименты, графики, выводы) — выполнена полностью |
| `project_lib.py` | загрузка и очистка данных, токенизация, батчи, модель, цикл обучения, журнал запусков |
| `requirements.txt` | нужные пакеты с версиями |
| `results/` | таблицы (`*.csv`, `*.json`), истории обучения (`runs/*.json`), журнал `progress.log`; веса моделей (`results/ckpt/`) в репозиторий не входят |
| `figures/` | все графики в PNG (24 шт.) |
| `report/project_report.pdf`, `report/project_report.docx` | отчёт по проекту (источник — `report/report.md`) |
| `presentation/presentation.pptx`, `presentation/presentation.pdf` | презентация к защите |

## Результаты

Итоговая модель — BiLSTM (h = 128, 1 слой) на подсловах `cointegrated/rubert-tiny2` с дообучаемыми предобученными
эмбеддингами, Adam `lr = 1e-3`, dropout 0,3. Выбрана по валидации среди 23 конфигураций.

| модель | macro-F1 на тесте (4 723 текста) |
|---|---|
| **итоговая RNN, 3 seed** | **0,751 ± 0,006** (accuracy 0,752 ± 0,005) |
| TF-IDF символьных n-грамм + логистическая регрессия | 0,747 |
| `rubert-tiny2`, дообученная целиком | 0,734 |
| BiLSTM по словам (этап A) | 0,722 |
| BiLSTM с обрезкой до 50 токенов, как в практике | 0,678 |
| простая RNN | 0,640 |

Подробности — в разделах 7–11 тетрадки и в отчёте.

## Установка

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Нужен интернет: набор данных и модель `cointegrated/rubert-tiny2` скачиваются с HuggingFace при первом запуске
(дальше берутся из кэша `~/.cache/huggingface`).

## Запуск

Целиком и неинтерактивно (рабочая папка — папка тетрадки):

```bash
MPLBACKEND=Agg PYTORCH_ENABLE_MPS_FALLBACK=1 \
  jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 project_rusentiment_rnn.ipynb
```

(или тем же способом через `nbclient`). Ход расчёта виден в `results/progress.log`:

```bash
tail -f results/progress.log
```

Переменные окружения:

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `NN_SMOKE` | `0` | `1` — проверка кода за пару минут: крошечная подвыборка, 1 эпоха, результаты в `smoke_out/` |
| `NN_DEVICE` | пусто | пусто — cuda → mps → cpu; `cpu` / `mps` / `cuda` — принудительно; `auto` — RNN на CPU, если замер показал, что CPU заметно быстрее MPS |
| `NN_RESUME` | `1` | уже посчитанные запуски не пересчитываются (продолжение прерванного расчёта) |
| `NN_RUN_TRANSFER` / `NN_RUN_BERT` | `1` | опыты с предобученными эмбеддингами / дообучение rubert-tiny2 |
| `NN_CHECK_PARENT` | `1` | проверка, что строки набора взяты из `MonoHime/ru_sentiment_dataset` (скачивает ≈160 МБ) |
| `NN_KEEP_ALL_CKPT` | `0` | `1` — не удалять в конце веса промежуточных моделей |

Если расчёт прервался, достаточно запустить тетрадку ещё раз: готовые запуски прочитаются из `results/runs/`.

## Проверка

* в конце `results/progress.log` — строка «Готово. Время выполнения тетрадки: … мин»;
* в `figures/` 24 графика, в `results/` — `final_test.csv`, `all_runs.csv`, `selection.json` и др.;
* итоговые метрики на тесте — `results/final_test.csv` (раздел 8 тетрадки).

## Время

Полный расчёт на Mac с Apple Silicon (MPS) занял 124 мин: 26 обучений рекуррентных сетей до 8 эпох с ранней
остановкой, базовые модели, дообучение rubert-tiny2. На другом компьютере время будет другим; строка «Устройство: …»
в начале `progress.log` сообщает время одного шага обучения BiLSTM (батч 128, длина 100) — на этом Mac 0,116 с.

## Презентация и отчёт

Готовые файлы — `presentation/presentation.pptx` и `.pdf`, `report/project_report.pdf` и `.docx`.
Отчёт можно пересобрать из `report/report.md` (нужен pandoc, для PDF ещё и xelatex):

```bash
cd report && pandoc report.md -o project_report.docx --resource-path=.:.. --toc
```

