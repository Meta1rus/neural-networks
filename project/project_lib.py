"""Вспомогательный модуль проекта «Тональность русскоязычных отзывов: RNN, LSTM, GRU».

Подольский Филипп, магистратура ПМИ, ФМИ ПГНИУ. Дисциплина «Нейронные сети».

Здесь лежит «механика», чтобы тетрадка читалась как отчёт об экспериментах:
загрузка и очистка данных, токенизация, батчи с упаковкой последовательностей,
модель RNN/LSTM/GRU, цикл обучения с ранней остановкой, журнал запусков с
возможностью продолжить прерванный расчёт, базовые модели sklearn и дообучение
трансформера-ориентира. Все решения, ради которых написан код, объясняются в тетрадке.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------------------------
# Метки и оформление графиков
# ---------------------------------------------------------------------------
# Кодировка меток — как в практике 10 и в карточке исходного набора MonoHime/ru_sentiment_dataset
LABEL_NAMES = ['NEUTRAL', 'POSITIVE', 'NEGATIVE']
LABEL_RU = ['нейтральный', 'позитивный', 'негативный']

# Категориальная палитра (фиксированный порядок цветов, проверенный на различимость при дальтонизме)
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
# Цвет закреплён за классом и не меняется от графика к графику (первые три слота палитры)
CLASS_COLORS = {0: '#2a78d6', 1: '#1baf7a', 2: '#eb6834'}
INK, INK2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
SEQ_CMAP_COLORS = ['#fcfcfb', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']


def setup_style():
    """Единый спокойный стиль графиков: светлый фон, тонкие линии, неяркая сетка."""
    import matplotlib as mpl
    mpl.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'axes.titlecolor': INK, 'text.color': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False,
        'lines.linewidth': 2.0, 'lines.markersize': 5,
        'axes.prop_cycle': mpl.cycler(color=PALETTE),
        'legend.frameon': False, 'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
        'figure.dpi': 100, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    })


def seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('seq_blue', SEQ_CMAP_COLORS)


# ---------------------------------------------------------------------------
# Воспроизводимость, устройство, журнал
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(override: str | None = None) -> torch.device:
    """cuda -> mps -> cpu; переменная окружения NN_DEVICE позволяет выбрать устройство вручную."""
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device('cuda')
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def check_rnn_on_device(device: torch.device, atol: float = 1e-3) -> dict:
    """Сверяет прямой и обратный проход упакованных RNN/LSTM/GRU на устройстве с CPU.

    Нужна для Apple MPS: если какая-то операция не поддерживается или считает неверно,
    тетрадка переходит на CPU, а не продолжает обучение на неправильных числах.
    """
    if device.type == 'cpu':
        return {'ok': True, 'note': 'CPU: сверка не нужна', 'checked': []}
    out = {'ok': True, 'checked': []}
    g = torch.Generator().manual_seed(0)
    x = torch.randn(6, 9, 8, generator=g)
    lengths = torch.tensor([9, 3, 7, 1, 5, 9])
    try:
        for cell in ['rnn', 'lstm', 'gru']:
            for bi in [False, True]:
                torch.manual_seed(0)
                base = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[cell](
                    8, 5, num_layers=2, batch_first=True, bidirectional=bi)
                res = {}
                for dev in [torch.device('cpu'), device]:
                    m = copy.deepcopy(base).to(dev)
                    xx = x.clone().to(dev).requires_grad_(True)
                    packed = pack_padded_sequence(xx, lengths, batch_first=True, enforce_sorted=False)
                    _, h = m(packed)
                    if isinstance(h, tuple):
                        h = h[0]
                    (h ** 2).sum().backward()
                    res[dev.type] = (h.detach().cpu(), xx.grad.detach().cpu())
                dh = (res['cpu'][0] - res[device.type][0]).abs().max().item()
                dg = (res['cpu'][1] - res[device.type][1]).abs().max().item()
                ok = dh < atol and dg < atol
                out['checked'].append({'cell': cell, 'bidirectional': bi, 'max_diff_h': dh, 'max_diff_grad': dg, 'ok': ok})
                out['ok'] = out['ok'] and ok
    except Exception as e:  # операция не реализована на устройстве и т. п.
        out['ok'] = False
        out['error'] = repr(e)[:500]
    return out


def time_train_steps(device, vocab_size=20000, batch=128, seq_len=100, steps=15, bidirectional=True):
    """Грубый замер: сколько секунд занимает один шаг обучения BiLSTM (для оценки времени и выбора устройства)."""
    set_seed(0)
    m = RNNClassifier(vocab_size, 128, 128, cell='lstm', bidirectional=bidirectional).to(device)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    x = torch.randint(2, vocab_size, (batch, seq_len))
    lens = torch.randint(seq_len // 2, seq_len + 1, (batch,))
    y = torch.randint(0, 3, (batch,))
    xd, yd = x.to(device), y.to(device)
    times = []
    for i in range(steps + 3):
        t0 = time.perf_counter()
        opt.zero_grad()
        loss = F.cross_entropy(m(xd, lens), yd)
        loss.backward()
        opt.step()
        loss.item()  # синхронизация с устройством
        if i >= 3:   # первые шаги — прогрев
            times.append(time.perf_counter() - t0)
    return float(np.median(times))


class ProgressLog:
    """Печатает сообщение и дописывает его с отметкой времени в файл (чтобы следить за долгим запуском)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()

    def __call__(self, msg: str):
        stamp = time.strftime('%H:%M:%S')
        el = (time.time() - self.t0) / 60
        line = f'[{stamp} | {el:6.1f} мин] {msg}'
        print(line, flush=True)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------
HF_DATASET = 'Megnis/ru_sentiment_dataset-50000'
HF_PARENT = 'MonoHime/ru_sentiment_dataset'


def load_main_dataset(smoke: bool, smoke_per_class: int = 300, seed: int = 42):
    """Загружает Megnis/ru_sentiment_dataset-50000 (один сплит train).

    Возвращает (DataFrame[src_id, text, label], описание источника).
    """
    from datasets import load_dataset
    ds = load_dataset(HF_DATASET, split='train')
    df = ds.to_pandas()
    need = {'Unnamed: 0', 'text', 'sentiment'}
    if not need.issubset(df.columns):
        raise ValueError(f'неожиданные колонки набора: {list(df.columns)}')
    df = df.rename(columns={'Unnamed: 0': 'src_id', 'sentiment': 'label'})
    source = f'HuggingFace: {HF_DATASET}'
    df['text'] = df['text'].fillna('').astype(str)
    df['label'] = df['label'].astype(int)
    if smoke:
        parts = [g.sample(n=min(smoke_per_class, len(g)), random_state=seed) for _, g in df.groupby('label')]
        df = pd.concat(parts)
    return df[['src_id', 'text', 'label']].reset_index(drop=True), source


def check_parent_dataset(df: pd.DataFrame, parent_name: str = HF_PARENT) -> dict:
    """Проверяет, что строки набора совпадают со строками исходного (более крупного) набора.

    Сопоставление по служебному индексу 'Unnamed: 0' (src_id), затем сравниваются текст и метка.
    """
    from datasets import load_dataset
    try:   # parquet-копия, которую HuggingFace строит для просмотрщика (≈160 МБ вместо CSV-файлов)
        par = load_dataset(parent_name, revision='refs/convert/parquet')
    except Exception:
        par = load_dataset(parent_name)
    frames = []
    for split in par:
        p = par[split].to_pandas().rename(columns={'Unnamed: 0': 'src_id', 'sentiment': 'label'})
        p['parent_split'] = split
        frames.append(p[['src_id', 'text', 'label', 'parent_split']])
    P = pd.concat(frames, ignore_index=True)
    stats = {'parent': parent_name, 'parent_rows': int(len(P)),
             'parent_rows_by_split': P['parent_split'].value_counts().to_dict(),
             'parent_src_id_unique': bool(P['src_id'].is_unique),
             'parent_class_counts': {LABEL_NAMES[k]: int(v) for k, v in P['label'].value_counts().sort_index().items()}}
    P1 = P.drop_duplicates('src_id')
    m = df.merge(P1, on='src_id', how='left', suffixes=('', '_parent'))
    found = m['text_parent'].notna()
    stats.update({
        'rows_checked': int(len(df)),
        'found_by_src_id': int(found.sum()),
        'same_text': int((m['text'] == m['text_parent']).sum()),
        'same_label': int((m['label'] == m['label_parent']).sum()),
        'found_in_parent_split': m.loc[found, 'parent_split'].value_counts().to_dict(),
    })
    # дополнительно — совпадение по самому тексту (если индекс где-то не совпал)
    stats['found_by_exact_text'] = int(df['text'].isin(set(P['text'])).sum())
    return stats


def normalize_text(s: str) -> str:
    s = s.lower().replace('ё', 'е')
    return re.sub(r'\s+', ' ', s).strip()


def deduplicate(df: pd.DataFrame):
    """Удаляет пустые тексты и повторы (после нормализации регистра/пробелов).

    Если одинаковый текст встречается с разными метками, вся группа удаляется:
    оставить любую из меток значило бы выбрать «правильный ответ» наугад.
    Возвращает (очищенный df, df удалённых строк-конфликтов, статистика).
    """
    d = df.copy()
    d['norm'] = d['text'].map(normalize_text)
    n_raw = len(d)
    empty = d['norm'].str.len() == 0
    d = d[~empty]
    g = d.groupby('norm')['label'].agg(['nunique', 'size'])
    conflict_keys = set(g.index[g['nunique'] > 1])
    conflicts = d[d['norm'].isin(conflict_keys)].copy()
    clean = d[~d['norm'].isin(conflict_keys)].drop_duplicates('norm', keep='first')
    stats = {
        'rows_raw': int(n_raw), 'rows_empty': int(empty.sum()),
        'dup_groups': int((g['size'] > 1).sum()),
        'extra_copies': int((g['size'] - 1).sum()),
        'conflict_groups': int(len(conflict_keys)), 'conflict_rows': int(len(conflicts)),
        'rows_clean': int(len(clean)),
        'dup_groups_by_label': {LABEL_NAMES[k]: int(v) for k, v in
                                d[d['norm'].isin(set(g.index[(g['size'] > 1) & (g['nunique'] == 1)]))]
                                .drop_duplicates('norm')['label'].value_counts().sort_index().items()},
    }
    return clean.drop(columns='norm').reset_index(drop=True), conflicts, stats


def split_data(df: pd.DataFrame, seed: int = 42):
    """Стратифицированное разбиение 80/10/10 — те же доли и seed, что в практике 10."""
    from sklearn.model_selection import train_test_split
    tr, tmp = train_test_split(df, test_size=0.2, random_state=seed, stratify=df['label'])
    va, te = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=tmp['label'])
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def data_fingerprint(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode('utf-8'))
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Токенизация и словарь
# ---------------------------------------------------------------------------
# слова (кириллица/латиница), числа, «!», «?» и смайлики из скобок — у последних есть эмоциональная окраска
TOKEN_RE = re.compile(r'[a-zа-я]+|\d+|[!?]|\)+|\(+')
PAD, UNK = '<pad>', '<unk>'


def tokenize(text: str) -> list[str]:
    toks = TOKEN_RE.findall(text.lower().replace('ё', 'е'))
    return [')' if t[0] == ')' else '(' if t[0] == '(' else t for t in toks]


def build_vocab(token_lists, min_freq: int = 2, max_size: int = 50000):
    cnt = Counter(t for toks in token_lists for t in toks)
    words = [w for w, c in cnt.most_common() if c >= min_freq][: max_size - 2]
    itos = [PAD, UNK] + words
    return {w: i for i, w in enumerate(itos)}, itos, cnt


def encode(tokens, stoi, max_len: int) -> list[int]:
    ids = [stoi.get(t, 1) for t in tokens[:max_len]]
    return ids or [1]   # пустой текст -> один <unk>: длина последовательности должна быть >= 1


def choose_max_len(lengths, quantile=0.95, step=16, cap=256) -> int:
    q = float(np.quantile(lengths, quantile))
    return int(min(cap, max(step, math.ceil(q / step) * step)))


def build_subword_bundle(model_name: str, tr_texts, va_texts, te_texts, quantile=0.95, cap=320,
                         load_embeddings=True):
    """Подсловная токенизация чужим (предобученным) токенизатором и матрица его входных эмбеддингов.

    Словарь — только токены, встретившиеся в обучающей части (остальные -> <unk>), поэтому
    и здесь «знание» о val/test в модель не попадает. Строки 0/1 — <pad>/<unk>.
    """
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(model_name)

    def enc(texts):
        return tok(list(texts), add_special_tokens=False, truncation=False)['input_ids']

    tr_ids, va_ids, te_ids = enc(tr_texts), enc(va_texts), enc(te_texts)
    lens = np.array([len(s) for s in tr_ids])
    max_len = choose_max_len(lens, quantile=quantile, cap=cap)
    used = sorted(Counter(i for s in tr_ids for i in s))
    remap = {o: k + 2 for k, o in enumerate(used)}

    def conv(seqs):
        return [[remap.get(i, 1) for i in s[:max_len]] or [1] for s in seqs]

    bundle = {'name': model_name, 'tokenizer': tok, 'remap': remap, 'max_len': max_len, 'train_lengths': lens,
              'train': conv(tr_ids), 'val': conv(va_ids), 'test': conv(te_ids),
              'vocab_size': len(used) + 2, 'pretrained': None, 'emb_dim': None,
              'unk_rate_val': float(np.mean([remap.get(i) is None for s in va_ids for i in s])) if va_ids else 0.0,
              'full_vocab_size': int(tok.vocab_size)}
    if load_embeddings:
        from transformers import AutoModel
        enc_model = AutoModel.from_pretrained(model_name)
        W = enc_model.get_input_embeddings().weight.detach().float().cpu()
        emb = torch.zeros(len(used) + 2, W.shape[1])
        if tok.unk_token_id is not None:
            emb[1] = W[tok.unk_token_id]
        emb[2:] = W[torch.tensor(used, dtype=torch.long)]
        bundle['pretrained'] = emb
        bundle['emb_dim'] = int(W.shape[1])
        bundle['pretrained_std'] = float(W.std())
        del enc_model
    return bundle


# ---------------------------------------------------------------------------
# Батчи: группировка по длине и динамический паддинг
# ---------------------------------------------------------------------------
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, labels):
        self.seqs, self.labels = seqs, list(map(int, labels))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], self.labels[i], i


class BucketBatchSampler(torch.utils.data.Sampler):
    """Перемешивает примеры, режет на «корзины» по batch*mult штук, внутри корзины сортирует по длине.

    Так в одном батче оказываются тексты похожей длины -> мало паддинга -> быстрее.
    Порядок зависит только от seed и номера эпохи, т. е. воспроизводим.
    """

    def __init__(self, lengths, batch_size, shuffle, seed=0, mult=50):
        self.lengths = np.asarray(lengths)
        self.bs, self.shuffle, self.seed, self.mult = batch_size, shuffle, seed, mult
        self.epoch = 0

    def _batches(self):
        n = len(self.lengths)
        if not self.shuffle:
            order = np.argsort(-self.lengths, kind='stable')
            return [order[i:i + self.bs] for i in range(0, n, self.bs)]
        rng = np.random.default_rng(self.seed * 1000 + self.epoch)
        idx = rng.permutation(n)
        batches = []
        chunk = self.bs * self.mult
        for s in range(0, n, chunk):
            part = idx[s:s + chunk]
            part = part[np.argsort(-self.lengths[part], kind='stable')]
            batches += [part[i:i + self.bs] for i in range(0, len(part), self.bs)]
        rng.shuffle(batches)
        return batches

    def __iter__(self):
        batches = self._batches()
        self.epoch += 1
        for b in batches:
            yield b.tolist()

    def __len__(self):
        return math.ceil(len(self.lengths) / self.bs)


def make_loader(seqs, labels, batch_size, shuffle, max_len, seed=0, pad_idx=0):
    lengths = [min(len(s), max_len) for s in seqs]
    sampler = BucketBatchSampler(lengths, batch_size, shuffle, seed)

    def collate(batch):
        ss, ys, idx = zip(*batch)
        ss = [s[:max_len] for s in ss]
        lens = torch.tensor([len(s) for s in ss], dtype=torch.long)
        x = torch.full((len(ss), int(lens.max())), pad_idx, dtype=torch.long)
        for i, s in enumerate(ss):
            x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return x, lens, torch.tensor(ys, dtype=torch.long), torch.tensor(idx, dtype=torch.long)

    return torch.utils.data.DataLoader(SeqDataset(seqs, labels), batch_sampler=sampler, collate_fn=collate)


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------
class RNNClassifier(nn.Module):
    """Эмбеддинги -> dropout -> RNN/LSTM/GRU (1..n слоёв, одно- или двунаправленная) -> h_T -> dropout -> Linear.

    Последовательности упаковываются (pack_padded_sequence), поэтому итоговое состояние h_T —
    это состояние после последнего НАСТОЯЩЕГО слова, а не после хвоста из <pad>.
    Инициализация — по рекомендациям лекции: ортогональные рекуррентные матрицы,
    смещение забывающего вентиля LSTM = 1.
    """

    def __init__(self, vocab_size, emb_dim, hidden_dim, n_classes=3, cell='lstm', n_layers=1,
                 bidirectional=False, dropout=0.3, pad_idx=0, pretrained=None, freeze_emb=False):
        super().__init__()
        self.cell, self.bidirectional = cell, bidirectional
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        if pretrained is not None:
            assert pretrained.shape == (vocab_size, emb_dim), (pretrained.shape, vocab_size, emb_dim)
            with torch.no_grad():
                self.embedding.weight.copy_(pretrained)
            self.embedding.weight.requires_grad_(not freeze_emb)
        rnn_cls = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[cell]
        kw = {'nonlinearity': 'tanh'} if cell == 'rnn' else {}
        self.rnn = rnn_cls(emb_dim, hidden_dim, num_layers=n_layers, batch_first=True,
                           bidirectional=bidirectional, dropout=dropout if n_layers > 1 else 0.0, **kw)
        self.emb_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), n_classes)
        self._init_rnn()

    def _init_rnn(self):
        H = self.rnn.hidden_size
        for name, p in self.rnn.named_parameters():
            if 'weight_hh' in name:
                for k in range(p.shape[0] // H):          # по блоку на каждый вентиль
                    nn.init.orthogonal_(p.data[k * H:(k + 1) * H])
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(p.data)
            elif 'bias' in name:
                p.data.zero_()
                if self.cell == 'lstm' and 'bias_ih' in name:
                    p.data[H:2 * H] = 1.0                 # вентили LSTM в PyTorch: i, f, g, o

    def encode(self, x, lengths):
        emb = self.emb_dropout(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        if isinstance(h, tuple):       # LSTM возвращает (h, c)
            h = h[0]
        return torch.cat([h[-2], h[-1]], dim=1) if self.bidirectional else h[-1]

    def forward(self, x, lengths):
        return self.fc(self.dropout(self.encode(x, lengths)))


def count_params(model, trainable_only=True):
    return int(sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only)))


# ---------------------------------------------------------------------------
# Обучение и оценка
# ---------------------------------------------------------------------------
def make_optimizer(params, cfg):
    name, lr, wd = cfg['optimizer'], cfg['lr'], cfg.get('weight_decay', 0.0)
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, weight_decay=wd)
    if name == 'sgd_nesterov':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    if name == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
    if name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(name)


@torch.no_grad()
def predict_logits(model, loader, device, encode=False):
    """Логиты (или скрытые представления при encode=True) в исходном порядке примеров."""
    model.eval()
    outs, ys, idxs = [], [], []
    for x, lens, y, idx in loader:
        x = x.to(device)
        o = model.encode(x, lens) if encode else model(x, lens)
        outs.append(o.float().cpu())
        ys.append(y)
        idxs.append(idx)
    out, y, idx = torch.cat(outs), torch.cat(ys), torch.cat(idxs)
    order = torch.argsort(idx)
    return out[order], y[order]


def metrics_from_logits(logits, y) -> dict:
    pred = logits.argmax(1).numpy()
    yt = y.numpy()
    return {'loss': float(F.cross_entropy(logits, y).item()), 'acc': float(accuracy_score(yt, pred)),
            'f1': float(f1_score(yt, pred, average='macro'))}


def train_model(model, train_loader, val_loader, cfg, device, log=print):
    """Обучение с ранней остановкой по macro-F1 на валидации.

    Возвращает историю по эпохам, след норм градиента (по шагам) и сведения о лучшей эпохе.
    Если patience = None — обучаем все эпохи; restore_best=False — оставляем веса последней эпохи.
    """
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(params, cfg)
    crit = nn.CrossEntropyLoss()
    clip = cfg.get('clip')
    patience = cfg.get('patience')
    hist = {k: [] for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1',
                            'epoch_time', 'grad_norm_mean', 'grad_norm_max']}
    grad_trace = []
    best = {'f1': -1.0, 'epoch': 0, 'state': None}
    bad, diverged = 0, False
    for epoch in range(1, cfg['max_epochs'] + 1):
        model.train()
        t0 = time.perf_counter()
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        n, norms = 0, []
        for x, lens, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x, lens)
            loss = crit(logits, y)
            loss.backward()
            # норма градиента ДО обрезки; при clip=None обрезки нет (порог = бесконечность)
            gn = torch.nn.utils.clip_grad_norm_(params, clip if clip else float('inf'))
            norms.append(gn.detach())
            opt.step()
            loss_sum += loss.detach() * len(y)
            correct += (logits.detach().argmax(1) == y).sum()
            n += len(y)
        train_loss = float((loss_sum / n).item())
        norms = torch.stack(norms).float().cpu().numpy()
        grad_trace += norms.tolist()
        if not math.isfinite(train_loss) or not np.all(np.isfinite(norms)):
            diverged = True
            log(f'    эпоха {epoch}: потери/градиенты стали бесконечными — обучение остановлено (расходимость)')
            hist['epoch_time'].append(time.perf_counter() - t0)
            break
        vl, vy = predict_logits(model, val_loader, device)
        vm = metrics_from_logits(vl, vy)
        dt = time.perf_counter() - t0
        for k, v in [('train_loss', train_loss), ('train_acc', float((correct / n).item())),
                     ('val_loss', vm['loss']), ('val_acc', vm['acc']), ('val_f1', vm['f1']),
                     ('epoch_time', dt), ('grad_norm_mean', float(np.mean(norms))),
                     ('grad_norm_max', float(np.max(norms)))]:
            hist[k].append(v)
        improved = vm['f1'] > best['f1']
        if improved:
            best = {'f1': vm['f1'], 'epoch': epoch,
                    'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
        log(f'    эпоха {epoch}: train_loss={train_loss:.4f}  val_loss={vm["loss"]:.4f}  '
            f'val_acc={vm["acc"]:.4f}  val_F1={vm["f1"]:.4f}  ({dt:.0f} с){"  *" if improved else ""}')
        if patience and bad >= patience:
            log(f'    ранняя остановка: {patience} эпох(и) без улучшения val macro-F1')
            break
    if cfg.get('restore_best', True) and best['state'] is not None:
        model.load_state_dict(best['state'])
    return hist, grad_trace, best, diverged


# ---------------------------------------------------------------------------
# Журнал экспериментов (с продолжением прерванного расчёта)
# ---------------------------------------------------------------------------
class Runner:
    """Запускает эксперимент по конфигурации, сохраняет историю (runs/*.json) и веса (ckpt/*.pt).

    При повторном выполнении тетрадки уже посчитанные запуски с той же конфигурацией и теми же
    данными не пересчитываются (если resume=True), а читаются с диска.
    """

    def __init__(self, out_dir, device, bundles, labels, data_tag, resume=True, log=print):
        self.dir = Path(out_dir)
        (self.dir / 'runs').mkdir(parents=True, exist_ok=True)
        (self.dir / 'ckpt').mkdir(parents=True, exist_ok=True)
        self.device, self.bundles, self.labels = device, bundles, labels
        self.data_tag, self.resume, self.log = data_tag, resume, log
        self.records = {}

    # --- построение модели и загрузчиков по конфигурации
    def build(self, cfg):
        b = self.bundles[cfg['tok']]
        pretrained = b.get('pretrained') if cfg.get('emb_init') == 'pretrained' else None
        return RNNClassifier(b['vocab_size'], cfg['emb_dim'], cfg['hidden_dim'], n_classes=3, cell=cfg['cell'],
                             n_layers=cfg['n_layers'], bidirectional=cfg['bidirectional'], dropout=cfg['dropout'],
                             pretrained=pretrained, freeze_emb=cfg.get('freeze_emb', False))

    def loader(self, cfg, split, shuffle=None, batch_size=None):
        b = self.bundles[cfg['tok']]
        shuffle = (split == 'train') if shuffle is None else shuffle
        bs = batch_size or (cfg['batch_size'] if split == 'train' else 512)
        return make_loader(b[split], self.labels[split], bs, shuffle, cfg['max_len'], seed=cfg['seed'])

    def _paths(self, name):
        return self.dir / 'runs' / f'{name}.json', self.dir / 'ckpt' / f'{name}.pt'

    def run(self, name, stage, cfg, note=''):
        jpath, cpath = self._paths(name)
        cfg = json.loads(json.dumps(to_jsonable(cfg)))
        if self.resume and jpath.exists():   # веса могут быть уже удалены — для чтения метрик они не нужны
            rec = json.load(open(jpath, encoding='utf-8'))
            if rec.get('cfg') == cfg and rec.get('data_tag') == self.data_tag:
                self.records[name] = rec
                self.log(f'[{name}] уже посчитан ранее — читаю результат с диска (val F1 = {rec["val_f1"]:.4f})')
                return rec
        self.log(f'[{name}] {note}')
        set_seed(cfg['seed'])
        model = self.build(cfg)
        tl, vl = self.loader(cfg, 'train'), self.loader(cfg, 'val')
        t0 = time.perf_counter()
        hist, trace, best, diverged = train_model(model, tl, vl, cfg, self.device, self.log)
        total = time.perf_counter() - t0
        if cfg.get('restore_best', True) and best['state'] is not None:
            sel = best['epoch'] - 1
        else:
            sel = len(hist['val_f1']) - 1
        ok = sel >= 0 and len(hist['val_f1']) > 0
        step = max(1, len(trace) // 2000)   # след градиента прореживаем до ~2000 точек
        rec = {
            'name': name, 'stage': stage, 'note': note, 'cfg': cfg, 'data_tag': self.data_tag,
            'device': str(self.device), 'n_params': count_params(model, True),
            'n_params_total': count_params(model, False),
            'epochs_run': len(hist['val_f1']), 'best_epoch': best['epoch'], 'selected_epoch': sel + 1 if ok else 0,
            'val_f1': hist['val_f1'][sel] if ok else float('nan'),
            'val_acc': hist['val_acc'][sel] if ok else float('nan'),
            'val_loss': hist['val_loss'][sel] if ok else float('nan'),
            'best_val_f1_any_epoch': max(hist['val_f1']) if ok else float('nan'),
            'time_total_s': total,
            'sec_per_epoch': float(np.mean(hist['epoch_time'])) if hist['epoch_time'] else float('nan'),
            'diverged': diverged, 'history': hist,
            'grad_trace': trace[::step], 'grad_trace_step': step, 'steps_per_epoch': len(tl),
        }
        torch.save(model.state_dict(), cpath)
        save_json(rec, jpath)
        self.records[name] = rec
        self.log(f'[{name}] готово: val F1 = {rec["val_f1"]:.4f}, val acc = {rec["val_acc"]:.4f}, '
                 f'эпох {rec["epochs_run"]} (выбрана {rec["selected_epoch"]}), {total / 60:.1f} мин')
        return rec

    def table(self, names=None):
        names = names or list(self.records)
        rows = []
        for n in names:
            r = self.records[n]
            c = r['cfg']
            rows.append({'run': n, 'stage': r['stage'], 'tok': c['tok'], 'cell': c['cell'],
                         'bi': c['bidirectional'], 'hidden': c['hidden_dim'], 'layers': c['n_layers'],
                         'emb': c['emb_dim'], 'emb_init': c.get('emb_init', 'random'),
                         'freeze_emb': c.get('freeze_emb', False), 'max_len': c['max_len'],
                         'optimizer': c['optimizer'], 'lr': c['lr'], 'wd': c.get('weight_decay', 0.0),
                         'dropout': c['dropout'], 'clip': c.get('clip'), 'seed': c['seed'],
                         'val_f1': r['val_f1'], 'val_acc': r['val_acc'], 'val_loss': r['val_loss'],
                         'epochs': r['epochs_run'], 'sel_epoch': r['selected_epoch'],
                         'sec_per_epoch': r['sec_per_epoch'], 'time_min': r['time_total_s'] / 60,
                         'params': r['n_params'], 'diverged': r['diverged']})
        return pd.DataFrame(rows)

    def load_model(self, name):
        rec = self.records[name]
        model = self.build(rec['cfg'])
        model.load_state_dict(torch.load(self._paths(name)[1], map_location='cpu'))
        return model.to(self.device).eval()

    def predict(self, name, split, encode=False, model=None):
        cfg = self.records[name]['cfg']
        model = model or self.load_model(name)
        return predict_logits(model, self.loader(cfg, split), self.device, encode=encode)

    def prune_checkpoints(self, keep):
        removed = 0
        for p in (self.dir / 'ckpt').glob('*.pt'):
            if p.stem not in keep:
                removed += p.stat().st_size
                p.unlink()
        return removed


def best_of(runner, names, key='val_f1'):
    """Имя запуска с наибольшим значением метрики на ВАЛИДАЦИИ."""
    # запуск, который разошёлся позже, участвует выбранной (до расхождения) эпохой — как при ранней остановке
    cand = [n for n in names if n in runner.records and math.isfinite(runner.records[n][key])]
    return max(cand, key=lambda n: runner.records[n][key])


# ---------------------------------------------------------------------------
# Базовые модели (sklearn)
# ---------------------------------------------------------------------------
def run_text_baselines(tr_texts, y_tr, va_texts, y_va, smoke=False, log=print, seed=42):
    """TF-IDF + логистическая регрессия (слова и символьные n-граммы), TF-IDF + случайный лес, константа.

    Векторизатор обучается только на train; параметр регуляризации C выбирается по macro-F1 на val.
    Возвращает (таблица всех вариантов, словарь лучших моделей {семейство: (векторизатор, классификатор)}).
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    rows, best = [], {}
    y_tr, y_va = np.asarray(y_tr), np.asarray(y_va)

    def record(family, params, clf, vec, Xva, fit_s):
        p = clf.predict(Xva)
        f1 = f1_score(y_va, p, average='macro')
        rows.append({'family': family, 'params': params, 'val_acc': accuracy_score(y_va, p), 'val_f1': f1,
                     'fit_s': fit_s, 'n_features': getattr(vec, 'vocabulary_', None) and len(vec.vocabulary_)})
        if family not in best or f1 > best[family][2]:
            best[family] = (vec, clf, f1, params)

    t0 = time.perf_counter()
    dummy = DummyClassifier(strategy='most_frequent').fit(np.zeros((len(y_tr), 1)), y_tr)
    record('dummy (самый частый класс)', '-', dummy, None, np.zeros((len(y_va), 1)), time.perf_counter() - t0)

    C_grid = [1.0] if smoke else [0.5, 2.0, 8.0]
    specs = [
        ('tfidf_word_lr', TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                          ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)),
        ('tfidf_char_lr', TfidfVectorizer(analyzer='char_wb', preprocessor=normalize_text, ngram_range=(2, 5),
                                          min_df=3, max_features=300000, sublinear_tf=True)),
    ]
    for family, vec in specs:
        t0 = time.perf_counter()
        Xtr = vec.fit_transform(tr_texts)
        Xva = vec.transform(va_texts)
        tv = time.perf_counter() - t0
        log(f'  {family}: признаков {Xtr.shape[1]}, векторизация {tv:.0f} с')
        for C in C_grid:
            t0 = time.perf_counter()
            clf = LogisticRegression(C=C, max_iter=3000, random_state=seed).fit(Xtr, y_tr)
            record(family, f'C={C}', clf, vec, Xva, time.perf_counter() - t0 + tv)
            log(f'    C={C}: val macro-F1 = {rows[-1]["val_f1"]:.4f}')
    # случайный лес: униграммы, ограниченный словарь (деревьям тяжело на сотнях тысяч разреженных признаков)
    t0 = time.perf_counter()
    vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False, min_df=3,
                          max_features=20000, sublinear_tf=True)
    Xtr = vec.fit_transform(tr_texts)
    Xva = vec.transform(va_texts)
    n_trees = 50 if smoke else 300
    rf = RandomForestClassifier(n_estimators=n_trees, n_jobs=-1, random_state=seed).fit(Xtr, y_tr)
    record('tfidf_rf', f'n_estimators={n_trees}', rf, vec, Xva, time.perf_counter() - t0)
    log(f'  tfidf_rf: val macro-F1 = {rows[-1]["val_f1"]:.4f} ({rows[-1]["fit_s"]:.0f} с)')
    return pd.DataFrame(rows), best


def baseline_predict_proba(model_tuple, texts):
    vec, clf = model_tuple[0], model_tuple[1]
    X = np.zeros((len(texts), 1)) if vec is None else vec.transform(texts)
    return clf.predict_proba(X)


# ---------------------------------------------------------------------------
# Трансформер-ориентир (дообучение предобученной модели)
# ---------------------------------------------------------------------------
def _hf_loader(ids, labels, batch_size, shuffle, pad_id, seed=0):
    lengths = [len(s) for s in ids]
    sampler = BucketBatchSampler(lengths, batch_size, shuffle, seed)

    def collate(batch):
        ss, ys, idx = zip(*batch)
        L = max(len(s) for s in ss)
        x = torch.full((len(ss), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(ss), L), dtype=torch.long)
        for i, s in enumerate(ss):
            x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            att[i, :len(s)] = 1
        return x, att, torch.tensor(ys, dtype=torch.long), torch.tensor(idx, dtype=torch.long)

    return torch.utils.data.DataLoader(SeqDataset(ids, labels), batch_sampler=sampler, collate_fn=collate)


@torch.no_grad()
def hf_predict_logits(model, loader, device):
    model.eval()
    outs, ys, idxs = [], [], []
    for x, att, y, idx in loader:
        out = model(input_ids=x.to(device), attention_mask=att.to(device)).logits
        outs.append(out.float().cpu())
        ys.append(y)
        idxs.append(idx)
    out, y, idx = torch.cat(outs), torch.cat(ys), torch.cat(idxs)
    order = torch.argsort(idx)
    return out[order], y[order]


class TransformerRef:
    """Дообучение предобученного трансформера (по умолчанию cointegrated/rubert-tiny2) на тех же train/val."""

    def __init__(self, model_name, texts, labels, cfg, device, out_dir, resume=True, log=print, data_tag=''):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        self.name, self.cfg, self.device, self.log = model_name, dict(cfg), device, log
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        self.max_len = int(min(cfg['max_len'], self.model.config.max_position_embeddings))
        self.cfg['max_len_effective'] = self.max_len
        self.labels = labels
        self.ids = {s: self.tok(list(t), truncation=True, max_length=self.max_len)['input_ids'] for s, t in texts.items()}
        self.trunc_share = {s: float(np.mean([len(i) >= self.max_len for i in v])) for s, v in self.ids.items()}
        self.dir = Path(out_dir)
        self.jpath, self.cpath = self.dir / 'runs' / 'F_rubert_tiny2.json', self.dir / 'ckpt' / 'F_rubert_tiny2.pt'
        self.resume, self.data_tag = resume, data_tag
        self.record = None

    def loader(self, split, shuffle=None):
        shuffle = (split == 'train') if shuffle is None else shuffle
        bs = self.cfg['batch_size'] if split == 'train' else 256
        return _hf_loader(self.ids[split], self.labels[split], bs, shuffle, self.tok.pad_token_id or 0,
                          seed=self.cfg['seed'])

    def fit(self):
        cfg = json.loads(json.dumps(to_jsonable({'model': self.name, **self.cfg})))
        if self.resume and self.jpath.exists() and self.cpath.exists():
            rec = json.load(open(self.jpath, encoding='utf-8'))
            if rec.get('cfg') == cfg and rec.get('data_tag') == self.data_tag:
                self.model.load_state_dict(torch.load(self.cpath, map_location='cpu'))
                self.model.to(self.device)
                self.record = rec
                self.log(f'[F_rubert_tiny2] уже посчитан ранее — читаю с диска (val F1 = {rec["val_f1"]:.4f})')
                return rec
        set_seed(self.cfg['seed'])
        model, dev = self.model.to(self.device), self.device
        tl, vl = self.loader('train'), self.loader('val')
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg['lr'], weight_decay=self.cfg.get('weight_decay', 0.01))
        total_steps = len(tl) * self.cfg['epochs']
        warm = int(self.cfg.get('warmup_ratio', 0.0) * total_steps)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total_steps - s) / max(1, total_steps - warm)))
        hist = {k: [] for k in ['train_loss', 'val_loss', 'val_acc', 'val_f1', 'epoch_time']}
        best = {'f1': -1.0, 'epoch': 0, 'state': None}
        t_all = time.perf_counter()
        self.log(f'[F_rubert_tiny2] дообучение {self.name}: max_len={self.max_len}, '
                 f'обучаемых параметров {count_params(model):,}')
        for epoch in range(1, self.cfg['epochs'] + 1):
            model.train()
            t0 = time.perf_counter()
            loss_sum, n = torch.zeros((), device=dev), 0
            for x, att, y, _ in tl:
                x, att, y = x.to(dev), att.to(dev), y.to(dev)
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(input_ids=x, attention_mask=att).logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                loss_sum += loss.detach() * len(y)
                n += len(y)
            vlog, vy = hf_predict_logits(model, vl, dev)
            vm = metrics_from_logits(vlog, vy)
            dt = time.perf_counter() - t0
            for k, v in [('train_loss', float((loss_sum / n).item())), ('val_loss', vm['loss']),
                         ('val_acc', vm['acc']), ('val_f1', vm['f1']), ('epoch_time', dt)]:
                hist[k].append(v)
            imp = vm['f1'] > best['f1']
            if imp:
                best = {'f1': vm['f1'], 'epoch': epoch,
                        'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            self.log(f'    эпоха {epoch}: train_loss={hist["train_loss"][-1]:.4f}  val_loss={vm["loss"]:.4f}  '
                     f'val_acc={vm["acc"]:.4f}  val_F1={vm["f1"]:.4f}  ({dt:.0f} с){"  *" if imp else ""}')
        model.load_state_dict(best['state'])
        sel = best['epoch'] - 1
        rec = {'name': 'F_rubert_tiny2', 'stage': 'F', 'cfg': cfg, 'data_tag': self.data_tag, 'device': str(dev),
               'n_params': count_params(model), 'epochs_run': len(hist['val_f1']), 'best_epoch': best['epoch'],
               'val_f1': hist['val_f1'][sel], 'val_acc': hist['val_acc'][sel], 'val_loss': hist['val_loss'][sel],
               'time_total_s': time.perf_counter() - t_all, 'sec_per_epoch': float(np.mean(hist['epoch_time'])),
               'truncated_share': self.trunc_share, 'history': hist}
        self.dir.joinpath('ckpt').mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.cpath)
        save_json(rec, self.jpath)
        self.record = rec
        return rec

    def predict(self, split):
        return hf_predict_logits(self.model, self.loader(split, shuffle=False), self.device)


# ---------------------------------------------------------------------------
# Графики, которые повторяются от этапа к этапу
# ---------------------------------------------------------------------------
def plot_histories(runner, names, labels=None, title='', path=None, colors=None):
    """Три панели по эпохам: потери на train, потери на val, macro-F1 на val (точка — выбранная эпоха)."""
    import matplotlib.pyplot as plt
    labels = labels or names
    colors = colors or PALETTE
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for i, (n, lab) in enumerate(zip(names, labels)):
        r = runner.records[n]
        h = r['history']
        ep = np.arange(1, len(h['val_f1']) + 1)
        c = colors[i % len(colors)]
        if len(ep) == 0:
            continue
        axes[0].plot(ep, h['train_loss'], '-o', color=c, label=lab, ms=4)
        axes[1].plot(ep, h['val_loss'], '-o', color=c, label=lab, ms=4)
        axes[2].plot(ep, h['val_f1'], '-o', color=c, label=lab, ms=4)
        if r['selected_epoch']:
            se = r['selected_epoch']
            axes[2].plot([se], [h['val_f1'][se - 1]], 'o', ms=10, mfc='none', mec=c, mew=2)
    for ax, t in zip(axes, ['Потери на обучающей выборке', 'Потери на валидации', 'macro-F1 на валидации']):
        ax.set_title(t)
        ax.set_xlabel('эпоха')
        ax.xaxis.get_major_locator().set_params(integer=True)
    axes[0].set_ylabel('кросс-энтропия')
    axes[2].set_ylabel('macro-F1')
    axes[2].legend(loc='lower right', fontsize=9)
    if title:
        fig.suptitle(title, y=1.02, fontsize=12, fontweight='bold')
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig
