# Тональность русскоязычных текстов рекуррентными сетями и 10 практик по нейросетям на PyTorch

Проект: сравнение RNN, LSTM, GRU и двунаправленных вариантов, размера сети, оптимизаторов, регуляризации и переноса
обучения с `rubert-tiny2` — 26 обучений, выбор модели по валидации и однократная оценка на тесте. Рядом — 10 практик:
от тензоров и оптимизаторов до CNN, transfer learning, VAE и трансформеров.

## Задача и данные

Определить тональность русскоязычного текста — нейтральный, позитивный или негативный — на наборе
[`Megnis/ru_sentiment_dataset-50000`](https://huggingface.co/datasets/Megnis/ru_sentiment_dataset-50000) (HuggingFace).
Данные очищены от повторов и разбиты на обучение, валидацию и тест без утечки; тест — 4 723 текста.

## Результат

**macro-F1 = 0,751 ± 0,006 на тесте (3 seed): на 2,9 п. п. выше BiLSTM из практики и на 7 п. п. выше той же модели
с обрезкой текстов до 50 токенов.**

| Модель | macro-F1 на тесте |
|---|---:|
| **Итоговая: BiLSTM на подсловах с дообучаемыми эмбеддингами `rubert-tiny2`** | **0,751 ± 0,006** |
| TF-IDF символьных n-грамм + логистическая регрессия | 0,747 |
| `rubert-tiny2`, дообученная целиком | 0,734 |
| BiLSTM по словам, как в практике, но без утечки | 0,722 |
| BiLSTM с обрезкой до 50 токенов, как в практике | 0,678 |
| Простая RNN | 0,640 |

* Итоговая модель выбрана по валидации среди 23 конфигураций, тест использован один раз. На тесте результат ниже, чем
  на валидации (0,751 против 0,769): оценка лучшей из 23 конфигураций на валидации оптимистична.
* Рекуррентная сеть вышла на уровень сильного бейзлайна TF-IDF и обошла `rubert-tiny2`, дообученную тем же рецептом.
  Для сравнения: опубликованная на HuggingFace `rubert-tiny2`, дообученная на исходном наборе, — 0,777 на своём тесте.
* В тетрадке также разбор ошибок, t-SNE скрытых состояний и проверка происхождения строк набора.
* [Тетрадка](project/project_rusentiment_rnn.ipynb), [код](project/project_lib.py),
  [отчёт](project/report/project_report.pdf), [презентация](project/presentation/presentation.pdf),
  [запуск проекта](project/README.md).

## Практики

| Практика | Тема |
|---|---|
| [1](practices/practice_01_tensors.ipynb) | тензоры, autograd, линейная регрессия градиентным спуском |
| [2](practices/practice_02_optimizers.ipynb) | SGD, Momentum, RMSprop, Adam на функциях Розенброка и Химмельблау; границы устойчивости по гессиану |
| [3](practices/practice_03_neuron_logic.ipynb) | один нейрон: логические функции, сравнение активаций |
| [4](practices/practice_04_mlp.ipynb) | MLP на Fashion-MNIST: инициализации и скорость обучения |
| [5](practices/practice_05_regularization.ipynb) | MLP на CIFAR-10: BatchNorm, Dropout, аугментации, ранняя остановка |
| [6](practices/practice_06_cnn.ipynb) | CNN на CIFAR-10: модификации архитектуры, GAP, depthwise separable свёртки |
| [8](practices/practice_08_transfer_learning.ipynb) | transfer learning: ResNet-18, fine-tuning и feature extractor на CIFAR-100 |
| [9](practices/practice_09_vae.ipynb) | вариационный автокодировщик, β-VAE, свёрточный VAE |
| [10](practices/practice_10_rnn_lstm.ipynb) | RNN, LSTM, GRU: adding problem и тональность текстов |
| [11](practices/practice_11_transformers.ipynb) | трансформеры: RuBERT через `Trainer`, генерация ruGPT-3 |

Главные ответы по каждой практике и изменения по сравнению с тетрадками преподавателя — в
[docs/assignments.md](docs/assignments.md).

## Стек

Python 3.13 · PyTorch (MPS / CUDA) · torchvision · Hugging Face Transformers и Datasets · scikit-learn · NumPy ·
pandas · matplotlib · seaborn · plotly

## Структура и запуск

```
├── practices/        # практики 1–6, 8–11
├── project/          # тетрадка, project_lib.py, results/, figures/, report/, presentation/
├── docs/assignments.md
└── requirements.txt
```

```bash
git clone https://github.com/Meta1rus/neural-networks.git && cd neural-networks
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Данные и предобученные модели (torchvision, HuggingFace, GitHub) тетрадки скачивают сами при первом запуске и кэшируют
в `data/`, в git они не хранятся. Тетрадки запускаются из своей папки. Все сохранены с результатами; полный прогон на Mac
с Apple Silicon: практики — от секунд до 2 ч каждая, проект — около 2 ч. `NN_DEVICE=cpu|mps|cuda` задаёт устройство явно.

---

Выполнено в рамках курса «Нейронные сети» магистратуры ПМИ ПГНИУ, 2026.
