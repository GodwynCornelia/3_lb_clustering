"""Собрать, выполнить и заполнить отчёт фактическими числами: python build_notebook.py."""
import json
import os
import sys
from pathlib import Path
import textwrap

import nbformat as nbf
from nbclient import NotebookClient
from report_analysis import ANALYSES

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
# Ограничение потоков ускоряет небольшие матрицы и снижает вариативность BLAS.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))


SPLITS = {
    'data': [('stats = X_raw.describe().T', 'data_load')],
    'features': [('X_selected = X_raw[features].copy()', 'correlations')],
    'outliers': [('X_clean = X_selected.copy()', 'outlier_detection')],
    'hopkins': [('fig, ax = plt.subplots(figsize=(8, 4))', 'hopkins_numbers')],
    'scaling': [('grid = sns.pairplot', 'scaling_numbers')],
    'selection': [
        ('fig, axes = plt.subplots(2, 2, figsize=(12, 8))', 'selection_table'),
        ('fig, axes = plt.subplots(1, 4, figsize=(17, 5))', 'selection_curves'),
        ('fig, ax = plt.subplots(figsize=(13, 5))', 'selection_profiles'),
    ],
    'parameters': [('covariance_rows = []', 'kmeans_parameters')],
}


def block(key, code):
    code = textwrap.dedent(code).strip()
    # Интерпретация присоединяется к своему расчёту, а не к концу отчёта.
    remainder = code
    parts = []
    for marker, subkey in SPLITS.get(key, []):
        before, separator, after = remainder.partition(marker)
        if not separator:
            raise ValueError(f'Не найден разделитель {key}: {marker}')
        parts.append((subkey, before.rstrip()))
        remainder = separator + after
    parts.append((key, remainder.rstrip()))
    for subkey, source in parts:
        source += '\n\n' + textwrap.dedent(ANALYSES[subkey]).strip()
        cells.append(nbf.v4.new_code_cell(source, metadata={'analysis_key': subkey}))
        md('{{analysis:' + subkey + '}}')


md('''
# Лабораторная работа №3. Кластеризация данных
### Отчёт по практическому заданию курса «Машинное обучение»

## 1. Введение
**Цель работы:** исследовать сегментацию смартфонов по аппаратным характеристикам,
сопоставить K-Means и EM (Gaussian Mixture), оценить качество и объяснить полученные группы.

**Постановка задачи:**

1. Описать данные, распределения, пропуски, повторные строки и выбросы.
2. Обосновать набор признаков и проверить избыточность и кластерную тенденцию.
3. Сравнить способы масштабирования и рассмотреть парные распределения.
4. Исследовать число групп по локтю, силуэту и дендрограмме.
5. Обучить два метода и сравнить внутренние, геометрические и внешние метрики.
6. Проверить чувствительность к параметрам и составить физические портреты центров.

**Датасет:** Mobile Price Classification, одно наблюдение — один смартфон.

При обучении без учителя алгоритм не получает правильные ценовые классы, в отличие
от классификации. Гипотеза: четыре группы характеристик могут частично восстановить
скрытые ценовые сегменты. Это проверяемая гипотеза, а не гарантированный результат.

План: описание данных → предобработка → выбор числа групп → обучение и метрики →
исследование параметров → визуализация и профили → заключение → источники.
Четыре кластера — заранее заданный исследовательский сценарий ТЗ; оптимальное k
оценивается независимо по внутренним метрикам. Метки не определяют признаки и параметры.

**Оформление:** структура пояснений адаптирована по [референсу лабораторной работы](https://github.com/AHTOH-CODER/ML_LAB_3/blob/main/lab_3.ipynb).
После каждого вычислительного блока разобраны его таблицы и графики: что измерено,
какие значения получены и как это влияет на следующий шаг исследования.
''')
block('setup', r'''
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import random
import warnings
warnings.filterwarnings('ignore')
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from IPython.display import display
from scipy.stats import shapiro, zscore
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, silhouette_samples,
    calinski_harabasz_score, davies_bouldin_score, pairwise_distances,
    adjusted_rand_score, rand_score, jaccard_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score, confusion_matrix)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
sns.set_theme(style='whitegrid', palette='Set2', font='DejaVu Sans')
plt.rcParams.update({'figure.dpi': 110, 'axes.titlesize': 11})
analysis = {}
FEATURE_LABELS = {
    'ram': 'ОЗУ, МБ', 'battery_power': 'Батарея, мАч',
    'px_height': 'Высота изображения, px', 'px_width': 'Ширина изображения, px',
    'int_memory': 'Встроенная память, ГБ', 'mobile_wt': 'Масса, г',
    'talk_time': 'Время разговора, ч', 'blue': 'Bluetooth', 'dual_sim': 'Две SIM-карты',
    'four_g': 'Поддержка 4G', 'three_g': 'Поддержка 3G',
    'touch_screen': 'Сенсорный экран', 'wifi': 'Wi-Fi',
    'clock_speed': 'Частота процессора, ГГц', 'm_dep': 'Толщина, см',
    'n_cores': 'Число ядер', 'fc': 'Фронтальная камера, Мп',
    'pc': 'Основная камера, Мп', 'sc_h': 'Высота экрана, см', 'sc_w': 'Ширина экрана, см',
}
''')
# В ноутбуке функции встроены: он не зависит от соседнего .py файла.
cells[-2].source += '\n\n' + (ROOT / 'clustering_utils.py').read_text(encoding='utf-8')

md('## 2. Описание датасета и дескриптивный анализ')
block('data', r'''
dataset_path = kagglehub.dataset_download('iabhishekofficial/mobile-price-classification')
csv_path = Path(dataset_path) / 'train.csv'
df = pd.read_csv(csv_path)
y_true = df['price_range'].copy()
X_raw = df.drop(columns=['price_range'])
assert X_raw.shape == (2000, 20), f'Неожиданная размерность: {X_raw.shape}'
assert 'price_range' not in X_raw
assert all(pd.api.types.is_numeric_dtype(t) for t in X_raw.dtypes)
assert np.isfinite(X_raw.to_numpy()).all(), 'Найдены пропуски/бесконечные значения'
X_raw.info()
display(X_raw.head())
stats = X_raw.describe().T
stats['skew'] = X_raw.skew()
stats['kurtosis'] = X_raw.kurtosis()
stats['variance'] = X_raw.var()
stats['CV'] = X_raw.std() / X_raw.mean().replace(0, np.nan)
display(stats.rename(index=FEATURE_LABELS))
missing = int(X_raw.isna().sum().sum())
duplicates = int(X_raw.duplicated().sum())
''')
block('distribution', r'''
binary = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']
continuous = [c for c in X_raw if c not in binary]
fig, axes = plt.subplots(4, 4, figsize=(16, 13))
normality = []
for ax, col in zip(axes.flat, continuous):
    sns.histplot(X_raw[col], kde=True, ax=ax, color=sns.color_palette('Set2')[0])
    ax.set(title=FEATURE_LABELS[col], xlabel=FEATURE_LABELS[col], ylabel='Число устройств')
    W, p = shapiro(X_raw[col])
    normality.append({'признак': col, 'W': W, 'p': p})
for ax in axes.flat[len(continuous):]:
    ax.set_visible(False)
plt.tight_layout(); plt.show()
normality = pd.DataFrame(normality).set_index('признак')
display(normality.rename(index=FEATURE_LABELS))
rejected = int((normality.p < .05).sum())
''')
block('features', r'''
features = ['ram', 'battery_power', 'px_height', 'px_width', 'int_memory', 'mobile_wt', 'talk_time']
corr = X_raw.corr()
fig, ax = plt.subplots(figsize=(13, 10))
sns.heatmap(corr.rename(index=FEATURE_LABELS, columns=FEATURE_LABELS), cmap='viridis', vmin=-1, vmax=1, ax=ax,
            cbar_kws={'label': 'Корреляция Пирсона'})
ax.set(title='Корреляция аппаратных характеристик', xlabel='Признак', ylabel='Признак')
plt.tight_layout(); plt.show()
X_selected = X_raw[features].copy()
selected_corr = X_selected.corr()
upper = np.abs(selected_corr.to_numpy()[np.triu_indices(len(features), 1)])
condition = np.linalg.cond(StandardScaler().fit_transform(X_selected))
display(stats.loc[features, ['variance', 'CV', 'skew']].rename(index=FEATURE_LABELS,
    columns={'variance': 'Дисперсия', 'skew': 'Асимметрия'}))
''')
block('outliers', r'''
q1, q3 = X_raw[continuous].quantile(.25), X_raw[continuous].quantile(.75)
iqr = q3 - q1
iqr_flags = (X_raw[continuous] < q1 - 1.5 * iqr) | (X_raw[continuous] > q3 + 1.5 * iqr)
z_flags = pd.DataFrame(np.abs(zscore(X_raw[continuous])) > 3,
                       index=X_raw.index, columns=continuous)
outliers = pd.DataFrame({'IQR': iqr_flags.sum(), '|Z|>3': z_flags.sum()})
display(outliers.rename(index=FEATURE_LABELS))
fig, axes = plt.subplots(4, 4, figsize=(15, 12))
for ax, col in zip(axes.flat, continuous):
    sns.boxplot(x=X_raw[col], ax=ax, color=sns.color_palette('Set2')[1])
    ax.set(title=FEATURE_LABELS[col], xlabel=FEATURE_LABELS[col], ylabel='Устройства')
for ax in axes.flat[len(continuous):]: ax.set_visible(False)
plt.tight_layout(); plt.show()
X_clean = X_selected.copy()
clipped = {}
for col in features:
    if (iqr_flags[col] | z_flags[col]).any():
        lo, hi = X_clean[col].quantile([.01, .99])
        old = X_clean[col].to_numpy().copy()
        X_clean[col] = np.clip(old, lo, hi)
        clipped[col] = int(np.count_nonzero(old != X_clean[col]))
assert X_clean.index.equals(y_true.index)
''')
block('hopkins', r'''
X_h = StandardScaler().fit_transform(X_clean)
hopkins_values = np.array([hopkins(X_h, random_state=RANDOM_STATE+i) for i in range(30)])
h_linear = hopkins(X_h, power=1)
h_raw = hopkins(StandardScaler().fit_transform(X_selected))
rng = np.random.default_rng(RANDOM_STATE)
null_values = []
for i in range(99):
    null_X = rng.uniform(X_h.min(axis=0), X_h.max(axis=0), X_h.shape)
    null_values.append(hopkins(null_X, random_state=RANDOM_STATE+i))
h_observed = hopkins(X_h)
empirical_p = (1 + np.count_nonzero(np.array(null_values) >= h_observed)) / 100
display(pd.Series({'H (степень d)': h_observed, 'H (степень 1)': h_linear,
                   'H до винзоризации': h_raw, 'H среднее 30 повторов': hopkins_values.mean(),
                   'H std': hopkins_values.std(ddof=1), 'uniform H среднее': np.mean(null_values),
                   'Монте-Карло p': empirical_p}))
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(null_values, ax=ax, color=sns.color_palette('Set2')[0], label='Равномерный контроль')
ax.axvline(h_observed, color=sns.color_palette('Set2')[1], label=f'Данные H={h_observed:.3f}')
ax.set(title='Кластерная тенденция: контроль Хопкинса', xlabel='H (степень d)', ylabel='Число симуляций')
ax.legend(); plt.tight_layout(); plt.show()
''')
md('## 3. Предобработка данных и визуальная оценка структуры')
block('scaling', r'''
scalers = {'StandardScaler': StandardScaler(), 'MinMaxScaler': MinMaxScaler(), 'RobustScaler': RobustScaler()}
scaled = {name: sc.fit_transform(X_clean) for name, sc in scalers.items()}
scale_results = []
for name, values in scaled.items():
    labels = KMeans(n_clusters=4, n_init=20, random_state=RANDOM_STATE).fit_predict(values)
    scale_results.append({'масштабирование': name, 'силуэт': silhouette_score(values, labels),
                          'CH': calinski_harabasz_score(values, labels), 'DB': davies_bouldin_score(values, labels)})
scale_results = pd.DataFrame(scale_results).set_index('масштабирование')
display(scale_results)
scaler = scalers['StandardScaler']
X = scaled['StandardScaler']
D = pairwise_distances(X)
np.fill_diagonal(D, 0)
assert np.max(np.abs(X.mean(axis=0))) < 1e-10
assert np.allclose(X.std(axis=0), 1)
grid = sns.pairplot(X_clean.rename(columns=FEATURE_LABELS), diag_kind='kde', plot_kws={'s': 7, 'alpha': .25, 'color': sns.color_palette('Set2')[0]})
grid.figure.suptitle('Парные распределения семи физических характеристик', y=1.02)
plt.show()
''')
md('## 4. Ход работы\n### 4.1. Определение числа кластеров')
block('selection', r'''
def internal(labels):
    return {'силуэт': silhouette_score(D, labels, metric='precomputed'),
            'CH': calinski_harabasz_score(X, labels), 'DB': davies_bouldin_score(X, labels)}
k_models, rows = {}, []
for k in range(2, 11):
    model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X)
    k_models[k] = model
    rows.append({'k': k, 'WCSS': model.inertia_, **internal(model.labels_)})
k_table = pd.DataFrame(rows).set_index('k')
display(k_table)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.flat, ['WCSS', 'силуэт', 'CH', 'DB']):
    ax.plot(k_table.index, k_table[col], 'o-', color=sns.color_palette('Set2')[0])
    ax.axvline(4, ls='--', color=sns.color_palette('Set2')[1], label='Сценарий k=4')
    ax.set(title=f'Выбор числа групп: {col}', xlabel='Число кластеров k', ylabel=col)
    ax.legend()
plt.tight_layout(); plt.show()
fig, axes = plt.subplots(1, 4, figsize=(17, 5))
for ax, k in zip(axes, [2, 3, 4, 5]):
    labels = k_models[k].labels_
    samples = silhouette_samples(D, labels, metric='precomputed')
    start = 0
    for cluster in range(k):
        vals = np.sort(samples[labels == cluster])
        ax.fill_betweenx(np.arange(start, start + len(vals)), 0, vals,
                         color=plt.cm.viridis(cluster / max(1, k-1)), label=f'Кластер {cluster}')
        start += len(vals) + 10
    ax.axvline(samples.mean(), color='black', ls='--', label='Средний силуэт')
    ax.set(title=f'Силуэтный профиль k={k}', xlabel='Коэффициент силуэта', ylabel='Упорядоченные устройства')
    ax.legend(fontsize=7)
plt.tight_layout(); plt.show()
fig, ax = plt.subplots(figsize=(13, 5))
tree = linkage(X, method='ward')
dendrogram(tree, truncate_mode='lastp', p=25, ax=ax, above_threshold_color=sns.color_palette('Set2')[0],
           link_color_func=lambda _: '#66c2a5')
cut = (tree[-3, 2] + tree[-4, 2]) / 2
ax.axhline(cut, ls='--', color='black', label=f'Отсечение для 4 групп: {cut:.2f}')
ax.set(title='Дендрограмма Уорда (последние 25 ветвей)', xlabel='Ветвь / число объединённых объектов', ylabel='Высота слияния Уорда')
ax.legend(); plt.tight_layout(); plt.show()
best_k = int(k_table['силуэт'].idxmax())
# Воспроизводимая эвристика локтя: максимальный отступ от хорды нормированной WCSS.
kn = (k_table.index.to_numpy() - 2) / 8
wn = (k_table.WCSS.to_numpy() - k_table.WCSS.iloc[-1]) / (k_table.WCSS.iloc[0] - k_table.WCSS.iloc[-1])
elbow_k = int(k_table.index[np.argmax(1 - kn - wn)])
''')
md('### 4.2. Обучение двух методов')
block('models', r'''
km = k_models[4]
gmm = GaussianMixture(n_components=4, covariance_type='full', n_init=5,
                      max_iter=500, reg_covar=1e-6, random_state=RANDOM_STATE).fit(X)
assert gmm.converged_, 'EM не сошёлся'
models = {'K-Means': km, 'GMM': gmm}
labels_by_model = {'K-Means': km.labels_, 'GMM': gmm.predict(X)}
centers = {'K-Means': km.cluster_centers_, 'GMM': gmm.means_}
probabilities = gmm.predict_proba(X)
display(pd.DataFrame({name: np.bincount(labels, minlength=4) for name, labels in labels_by_model.items()})
        .rename_axis('Номер кластера'))
''')
md('### 4.3. Внутренние, геометрические и внешние метрики')
block('metrics', r'''
metric_rows, geometry_rows, aligned_by_model, mappings = [], [], {}, {}
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for j, (name, labels) in enumerate(labels_by_model.items()):
    aligned, mapping = align_labels(y_true, labels)
    aligned_by_model[name], mappings[name] = aligned, mapping
    row = {'метод': name, **internal(labels), 'ARI': adjusted_rand_score(y_true, labels),
           'RI': rand_score(y_true, labels),
           'Jaccard macro': jaccard_score(y_true, aligned, average='macro', zero_division=0),
           'NMI': normalized_mutual_info_score(y_true, labels),
           'Homogeneity': homogeneity_score(y_true, labels),
           'Completeness': completeness_score(y_true, labels),
           'V-measure': v_measure_score(y_true, labels),
           'WCSS': km.inertia_ if name == 'K-Means' else np.nan,
           'BIC': gmm.bic(X) if name == 'GMM' else np.nan}
    metric_rows.append(row)
    distances = cdist(centers[name], centers[name])
    sns.heatmap(distances, annot=True, fmt='.2f', cmap='viridis', ax=axes[0, j],
                cbar_kws={'label': 'Евклидово расстояние'})
    axes[0, j].set(title=f'{name}: расстояния центров', xlabel='Номер кластера', ylabel='Номер кластера')
    cm = confusion_matrix(y_true, aligned, labels=[0, 1, 2, 3])
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', ax=axes[1, j],
                cbar_kws={'label': 'Число устройств'})
    axes[1, j].set(title=f'{name}: после венгерского сопоставления', xlabel='Сопоставленный кластер', ylabel='Эталонный ценовой класс')
    for c in range(4):
        d = np.linalg.norm(X[labels == c] - centers[name][c], axis=1)
        geometry_rows.append({'метод': name, 'кластер': c, 'размер': len(d),
                              'среднее расстояние до центра': d.mean(), 'радиус max': d.max(),
                              'радиус 95%': np.quantile(d, .95)})
plt.tight_layout(); plt.show()
metrics = pd.DataFrame(metric_rows).set_index('метод')
geometry = pd.DataFrame(geometry_rows).set_index(['метод', 'кластер'])
display(metrics); display(geometry)
''')
md('### 4.4. Влияние гиперпараметров')
block('parameters', r'''
parameter_rows = []
for k in range(2, 13):
    for init in ['k-means++', 'random']:
        for n_init in [1, 10, 20]:
            model = KMeans(n_clusters=k, init=init, n_init=n_init, random_state=RANDOM_STATE).fit(X)
            parameter_rows.append({'k': k, 'init': init, 'n_init': n_init,
                                   'WCSS': model.inertia_, **internal(model.labels_),
                                   'ARI': adjusted_rand_score(y_true, model.labels_)})
parameters = pd.DataFrame(parameter_rows)
display(parameters)
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, col in zip(axes, ['WCSS', 'силуэт', 'ARI']):
    for (init, n), part in parameters.groupby(['init', 'n_init']):
        ax.plot(part.k, part[col], marker='.', label=f'{init}, запусков {n}')
    ax.set(title=f'K-Means: зависимость {col}', xlabel='Число кластеров k', ylabel=col)
    ax.legend(fontsize=7)
plt.tight_layout(); plt.show()
covariance_rows = []
for cov in ['full', 'tied', 'diag', 'spherical']:
    model = GaussianMixture(n_components=4, covariance_type=cov, n_init=5,
                            max_iter=500, random_state=RANDOM_STATE).fit(X)
    assert model.converged_, f'EM не сошёлся: {cov}'
    lab = model.predict(X)
    covariance_rows.append({'covariance_type': cov, **internal(lab), 'BIC': model.bic(X),
                            'ARI': adjusted_rand_score(y_true, lab)})
covariance_table = pd.DataFrame(covariance_rows).set_index('covariance_type')
display(covariance_table)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ['силуэт', 'ARI', 'BIC']):
    sns.barplot(x=covariance_table.index, y=covariance_table[col], color=sns.color_palette('Set2')[0], ax=ax)
    ax.set(title=f'GMM: влияние ковариации на {col}', xlabel='Тип ковариации', ylabel=col)
plt.tight_layout(); plt.show()
sub = parameters[parameters.k == 4]
''')
block('dbscan', r'''
db_rows = []
for eps in np.arange(.5, 3.01, .5):
    for min_samples in [3, 5, 10, 20]:
        lab = DBSCAN(eps=float(eps), min_samples=min_samples).fit_predict(X)
        n_clusters = len(set(lab) - {-1})
        db_rows.append({'eps': eps, 'min_samples': min_samples, 'число кластеров': n_clusters,
                        'доля шума': np.mean(lab == -1)})
db_table = pd.DataFrame(db_rows)
display(db_table)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col, fmt in zip(axes, ['число кластеров', 'доля шума'], ['.0f', '.2f']):
    sns.heatmap(db_table.pivot(index='min_samples', columns='eps', values=col), annot=True,
                fmt=fmt, cmap='viridis', ax=ax, cbar_kws={'label': col})
    ax.set(title=f'DBSCAN: {col}', xlabel='Радиус ε', ylabel='Минимум соседей')
plt.tight_layout(); plt.show()
''')
md('## 5. Визуализация результатов и профилирование кластеров')
block('pca', r'''
pca = PCA(n_components=2, random_state=RANDOM_STATE)
projection = pca.fit_transform(X)
display(pd.DataFrame(pca.components_.T, index=features, columns=['Нагрузка PC1', 'Нагрузка PC2']))
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, name in zip(axes[:2], ['K-Means', 'GMM']):
    sc = ax.scatter(*projection.T, c=labels_by_model[name], cmap='viridis', vmin=0, vmax=3, s=9, alpha=.5)
    cp = pca.transform(centers[name])
    ax.scatter(*cp.T, marker='X', s=220, c=np.arange(4), cmap='viridis', edgecolor='black', label='Центры')
    ax.set(title=f'{name}: проекция кластеров', xlabel='Главная компонента 1', ylabel='Главная компонента 2')
    ax.legend(); fig.colorbar(sc, ax=ax, ticks=range(4), label='Номер кластера')
sc = axes[2].scatter(*projection.T, c=y_true, cmap='viridis', vmin=0, vmax=3, s=9, alpha=.5)
axes[2].set(title='Эталонные классы: только визуальный аудит', xlabel='Главная компонента 1', ylabel='Главная компонента 2')
fig.colorbar(sc, ax=axes[2], ticks=range(4), label='Ценовой класс')
plt.tight_layout(); plt.show()
''')
block('profiles', r'''
physical_profiles = {}
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
unit_names = ['ОЗУ, МБ', 'Батарея, мАч', 'Высота, px', 'Ширина, px', 'Память, ГБ', 'Масса, г', 'Разговор, ч']
for ax, name in zip(axes, ['K-Means', 'GMM']):
    physical = pd.DataFrame(scaler.inverse_transform(centers[name]), columns=features).rename_axis('кластер')
    physical_profiles[name] = physical
    display(physical.rename(columns=dict(zip(features, unit_names))))
    sns.heatmap(pd.DataFrame(centers[name], columns=unit_names), annot=True, fmt='.2f', cmap='viridis',
                ax=ax, cbar_kws={'label': 'Стандартизованное значение центра'})
    ax.set(title=f'{name}: физические профили центров', xlabel='Характеристика', ylabel='Номер кластера')
plt.tight_layout(); plt.show()
agreement = adjusted_rand_score(labels_by_model['K-Means'], labels_by_model['GMM'])
''')
md('### Сравнительный анализ методов')
block('comparison', r'''
display(metrics.round(4))
comparison = metrics[['силуэт', 'ARI', 'NMI', 'Jaccard macro']].rename(
    columns={'силуэт': 'Силуэт', 'Jaccard macro': 'Жаккар macro'})
comparison_plot = comparison.rename_axis('Метод').reset_index().melt(
    id_vars='Метод', var_name='Метрика', value_name='Значение')
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=comparison_plot, x='Метод', y='Значение', hue='Метрика',
            palette='Set2', ax=ax)
ax.set(title='Итоговое сравнение K-Means и GMM', xlabel='Метод', ylabel='Значение индекса')
ax.set_ylim(0, 1)
ax.legend(title='Метрика')
plt.tight_layout(); plt.show()
''')
md('## 6. Заключение\n{{analysis:conclusion}}')
md('''
## 7. Список источников
1. Mobile Price Classification / Abhishek Sharma // Kaggle : набор данных. URL: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification (дата обращения: 09.11.2026; загрузка через kagglehub).
2. Scikit-learn. Clustering : документация [Электронный ресурс]. URL: https://scikit-learn.org/stable/modules/clustering.html (справочный адрес; веб-страница при подготовке не просматривалась, использованы установленные API).
3. MacQueen J. Some Methods for Classification and Analysis of Multivariate Observations // Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability. 1967. Vol. 1. P. 281–297.
4. Dempster A. P., Laird N. M., Rubin D. B. Maximum Likelihood from Incomplete Data via the EM Algorithm // Journal of the Royal Statistical Society. Series B. 1977. Vol. 39, no. 1. P. 1–38.
5. Hopkins B., Skellam J. G. A New Method for Determining the Type of Distribution of Plant Individuals // Annals of Botany. 1954. Vol. 18, no. 2. P. 213–227.
6. Rousseeuw P. J. Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis // Journal of Computational and Applied Mathematics. 1987. Vol. 20. P. 53–65.
''')
md('## Приложение. Программное окружение и воспроизводимость')
block('environment', r'''
versions = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
            'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__, 'seaborn': sns.__version__}
display(pd.Series(versions, name='Версия'))
''')


def main():
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    # Временная ячейка передаёт вычисленные аналитические тексты генератору.
    notebook.cells.append(nbf.v4.new_code_cell(
        "from IPython.display import JSON\ndisplay(JSON(analysis))"))
    print('Executing notebook...', flush=True)
    client = NotebookClient(notebook, timeout=1800, kernel_name='python3',
                            resources={'metadata': {'path': str(ROOT)}},
                            on_cell_start=lambda cell, cell_index: print(
                                f'Cell {cell_index + 1}/{len(notebook.cells)}: {cell.cell_type}', flush=True))
    try:
        client.execute()
    except Exception:
        nbf.write(notebook, ROOT / 'Lab3_Clustering.failed.ipynb')
        raise
    payload = next(o['data']['application/json'] for o in notebook.cells[-1].outputs
                   if 'application/json' in o.get('data', {}))
    notebook.cells.pop()
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            for key, text in payload.items():
                cell.source = cell.source.replace('{{analysis:' + key + '}}', text)
            if '{{analysis:' in cell.source:
                raise RuntimeError('Не заполнен аналитический блок')
    nbf.validate(notebook)
    nbf.write(notebook, ROOT / 'Lab3_Clustering.ipynb')
    (ROOT / 'results.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved Lab3_Clustering.ipynb ({len(notebook.cells)} cells)', flush=True)


if __name__ == '__main__':
    main()
