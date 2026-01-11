---
title: Matplotlib プロット集
---

以下に、Matplotlib だけ（＋標準的な拡張モジュール mplot3d）で直接描画できる代表的なプロット（グラフ）タイプをなるべく網羅的にまとめました。用途や見た目の好みに合わせて使い分けてみてください。

---

## 1. 基本的な 2D プロット

| 種類            | 関数・メソッド                             | 説明                                      |
|:---------------|:-----------------------------------------|:-----------------------------------------|
| 折れ線グラフ     | `plt.plot()`                             | 連続データの推移や時系列データの可視化に最適 |
| 散布図           | `plt.scatter()`                          | 2 変量の分布・相関を点で表現                |
| 棒グラフ         | `plt.bar()` / `plt.barh()`               | カテゴリ別の比較                            |
| ヒストグラム     | `plt.hist()`                             | 分布（度数分布）                             |
| 箱ひげ図         | `plt.boxplot()`                          | データの分布概要（中央値、四分位範囲、外れ値） |
| バイオリンプロット | `plt.violinplot()`                       | 箱ひげ図＋カーネル密度推定による分布形状      |
| 円グラフ         | `plt.pie()`                              | 構成比（割合）の可視化                        |
| 面グラフ         | `plt.stackplot()` / `plt.fill_between()` | 複数系列の累積推移、領域の塗りつぶし          |
| ステッププロット | `plt.step()`                             | 階段状プロット（離散的変化を明示）            |
| ステムプロット   | `plt.stem()`                             | データ点＋垂直線（離散データの可視化）        |
| エラーバー       | `plt.errorbar()`                         | データ点の誤差範囲（上下）を線で描画         |
| ログプロット     | `plt.semilogx()` / `plt.semilogy()` / `plt.loglog()` | 軸を対数スケールで可視化                      |

---

## 2. カラーマップ／2D 密度系

| 種類               | 関数・メソッド                    | 説明                                     |
|:------------------|:--------------------------------|:----------------------------------------|
| 画像表示           | `plt.imshow()`                  | 2D 配列をグリッド状に色分け               |
| 等高線図           | `plt.contour()` / `plt.contourf()` | 等高線／塗りつぶし等高線                  |
| メッシュプロット   | `plt.pcolor()` / `plt.pcolormesh()` | グリッド上の塗りつぶし                     |
| ヒートマップ（Seaborn）※ | `sns.heatmap()`                 | Seaborn 経由でよりリッチなヒートマップ※   |
| 六角形ビンプロット | `plt.hexbin()`                  | 散布図の密度を六角形ビンで集計表示         |

※Seaborn は Matplotlib ベースのライブラリですが、公式にバンドルはされていません。より高度な統計プロットに便利です。

---

## 3. ベクトル／流れ場プロット

| 種類         | 関数・メソッド     | 説明                          |
|:------------|:-----------------|:-----------------------------|
| クワイバー   | `plt.quiver()`   | ベクトル場（矢印）の可視化     |
| ストリームプロット | `plt.streamplot()` | 流線（速度場などの流れのライン） |

---

## 4. イベント・カテゴリ系

| 種類       | 関数・メソッド        | 説明                                    |
|:----------|:--------------------|:---------------------------------------|
| イベントプロット | `plt.eventplot()`    | 縦線や点で時刻イベント（スパイク列）の可視化 |
| テーブル     | `plt.table()`         | 数値データをテーブル形式で埋め込み表示       |

---

## 5. 極座標プロット（Polar）

| 種類            | 関数・メソッド               | 説明                    |
|:---------------|:---------------------------|:-----------------------|
| 極座標線プロット | `ax = plt.subplot(projection='polar'); ax.plot()`      | 極座標上の折れ線グラフ |
| 極座標散布図     | `ax.scatter()`             | 極座標上の散布図        |
| 極座標棒グラフ   | `ax.bar()`                 | 極座標上の棒グラフ      |
| 極座標ヒストグラム | `ax.hist()`                | 極座標上のヒストグラム  |

---

## 6. 3D プロット（mpl_toolkits.mplot3d）

| 種類           | メソッド                          | 説明                                  |
|:--------------|:--------------------------------|:-------------------------------------|
| 3D 折れ線グラフ | `ax.plot()`                     | 3 次元空間での線グラフ                  |
| 3D 散布図      | `ax.scatter()`                  | 3 次元散布図                            |
| 3D ワイヤーフレーム | `ax.plot_wireframe()`           | 3D ワイヤーフレームサーフェス           |
| 3D サーフェス  | `ax.plot_surface()`             | カラーマップ付きサーフェス             |
| 3D 等高線      | `ax.contour()` / `ax.contourf()` | 3D 等高線プロット                      |
| 3D 棒グラフ     | `ax.bar3d()`                    | 3D 空間での棒グラフ                     |
| 3D ベクトル    | `ax.quiver()`                   | 3D ベクトル場（矢印）                  |

---

## 7. その他の便利プロット

| 種類                | 関数・メソッド              | 説明                                         |
|:-------------------|:--------------------------|:--------------------------------------------|
| スタックエリアチャート | `plt.stackplot()`          | 複数系列の面グラフ                            |
| ヒストグラム（累積）   | `plt.hist(cumulative=True)`| 累積度数／累積分布                            |
| 累積分布関数（CDF）    | `np.sort()`＋`plt.plot()`   | CDF を自前でプロット                          |
| 折れ線の塗り潰し       | `plt.fill_between()`       | ２線間の領域を色塗り潰し                      |
| ステップヒストグラム   | `plt.hist(histtype='step')` | 線のみのヒストグラム                          |
| 箱ひげ図の水平表示      | `plt.boxplot(vert=False)`  | 横倒しの箱ひげ図                              |
| エラーバンド           | `ax.fill_between(x, y-yerr, y+yerr, alpha=…)` | 誤差帯（バンド）を表現                         |
| プロット上への注釈      | `plt.annotate()`           | 任意の位置にテキストや矢印を付加                |
| サブプロットの並列描画  | `plt.subplot()` / `plt.subplots()` | 複数プロットを同じ図に配置                      |
| カラーバー             | `plt.colorbar()`           | カラーマップ付きプロットに対する凡例のような色バー |

---

## 8. 統計分析向けプロット

| 種類                 | 関数・メソッド                      | 説明                                         |
|:--------------------|:----------------------------------|:--------------------------------------------|
| 回帰直線付き散布図     | `np.polyfit()` + `plt.plot()`     | 散布図に回帰直線を重ねて傾向を確認            |
| 残差プロット           | `plt.subplots()` + `ax.scatter()` | 回帰モデルの残差の偏りを確認                 |
| 相関行列ヒートマップ   | `np.corrcoef()` + `plt.imshow()`  | 変数間の相関を色で可視化                      |

---

## 関数インデックス

### Matplotlib（pyplot）
- `plt.plot()` / `plt.scatter()` / `plt.bar()` / `plt.barh()` / `plt.hist()` / `plt.boxplot()` / `plt.violinplot()` / `plt.pie()`
- `plt.stackplot()` / `plt.fill_between()` / `plt.step()` / `plt.stem()` / `plt.errorbar()` / `plt.semilogx()` / `plt.semilogy()` / `plt.loglog()`
- `plt.imshow()` / `plt.contour()` / `plt.contourf()` / `plt.pcolor()` / `plt.pcolormesh()` / `plt.hexbin()`
- `plt.quiver()` / `plt.streamplot()` / `plt.eventplot()` / `plt.table()` / `plt.subplot()` / `plt.subplots()`
- `plt.colorbar()` / `plt.annotate()` / `plt.legend()` / `plt.savefig()` / `plt.show()` / `plt.tight_layout()`

### Matplotlib（Axes / 3D）
- `ax.plot()` / `ax.scatter()` / `ax.bar()` / `ax.hist()` / `ax.plot_surface()` / `ax.plot_wireframe()` / `ax.contour()` / `ax.contourf()` / `ax.bar3d()` / `ax.quiver()`

### NumPy
- `np.linspace()` / `np.random.seed()` / `np.random.rand()` / `np.random.randn()` / `np.random.normal()`
- `np.meshgrid()` / `np.mgrid` / `np.exp()` / `np.sin()` / `np.sqrt()` / `np.sort()` / `np.corrcoef()` / `np.polyfit()`

---

## サンプルスクリプト

以下に、代表的なプロットのサンプルコードを示します。各コードは独立して実行できます。

### 折れ線グラフ
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.sin(x)
# サイン波を折れ線で描画
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Line Plot')
plt.show()
```

![Line Plot](images/line_plot.png)


### 散布図
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.random.rand(50)
y = np.random.rand(50)
# x の値で色分けした散布図
plt.scatter(x, y, c=x, cmap='viridis', marker='o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
plt.colorbar()
plt.show()
```

![Scatter Plot](images/scatter_plot.png)


### 棒グラフ
```python
import matplotlib.pyplot as plt

categories = ['A', 'B', 'C', 'D']
values = [10, 24, 36, 18]
# カテゴリ別の棒グラフ
plt.bar(categories, values, color='skyblue')
plt.title('Bar Chart')
plt.show()
```

![Bar Chart](images/bar_chart.png)


### ヒストグラム
```python
import numpy as np
import matplotlib.pyplot as plt

data = np.random.randn(1000)
# 分布をヒストグラムで可視化
plt.hist(data, bins=30, color='gray', edgecolor='black')
plt.title('Histogram')
plt.show()
```

![Histogram](images/histogram.png)


### 箱ひげ図
```python
import numpy as np
import matplotlib.pyplot as plt

data = [np.random.randn(100) + i for i in range(4)]
# 複数系列の分布を箱ひげ図で比較
plt.boxplot(data)
plt.title('Box Plot')
plt.show()
```

![Box Plot](images/box_plot.png)


### 円グラフ
```python
import matplotlib.pyplot as plt

labels = ['Apple', 'Banana', 'Cherry', 'Date']
sizes = [30, 15, 45, 10]
# 割合の円グラフ
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Pie Chart')
plt.show()
```

![Pie Chart](images/pie_chart.png)


### imshow（画像表示）
```python
import numpy as np
import matplotlib.pyplot as plt

img = np.random.rand(10, 10)
# 2D 配列を画像として表示
plt.imshow(img, cmap='viridis')
plt.colorbar()
plt.title('Image Display')
plt.show()
```

![Image Display](images/image_display.png)


### 等高線図
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2))
# 等高線の塗りつぶし
plt.contourf(X, Y, Z, levels=20, cmap='coolwarm')
plt.colorbar()
plt.title('Filled Contour Plot')
plt.show()
```

![Filled Contour Plot](images/filled_contour_plot.png)


### クワイバー
```python
import numpy as np
import matplotlib.pyplot as plt

Y, X = np.mgrid[-3:3:100j, -3:3:100j]
U = -1 - X**2 + Y
V = 1 + X - Y**2
# ベクトル場の可視化
plt.quiver(X, Y, U, V)
plt.title('Quiver Plot')
plt.show()
```

![Quiver Plot](images/quiver_plot.png)


### 極座標プロット
```python
import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 2*np.pi, 100)
r = 1 + np.sin(4*theta)
# 極座標で描画
ax = plt.subplot(projection='polar')
ax.plot(theta, r)
ax.set_title('Polar Plot')
plt.show()
```

![Polar Plot](images/polar_plot.png)


### 3D サーフェスプロット
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X = np.linspace(-5, 5, 50)
Y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# 3D サーフェスを描画
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title('3D Surface Plot')
plt.show()
```

![3D Surface Plot](images/3d_surface_plot.png)

### 回帰直線付き散布図
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
x = np.linspace(0, 10, 50)
noise = np.random.normal(scale=1.2, size=x.size)
y = 1.8 * x + 2.5 + noise

slope, intercept = np.polyfit(x, y, 1)
y_fit = slope * x + intercept

# 回帰直線を重ねる
plt.scatter(x, y, color='tab:blue', alpha=0.75, label='Data')
plt.plot(x, y_fit, color='tab:orange', linewidth=2, label='Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regression Plot')
plt.legend()
plt.show()
```

![Regression Plot](images/regression_plot.png)


### 残差プロット
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)
x = np.linspace(0, 12, 60)
noise = np.random.normal(scale=1.5, size=x.size)
y = 2.2 * x - 1.0 + noise

slope, intercept = np.polyfit(x, y, 1)
y_fit = slope * x + intercept
residuals = y - y_fit

fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
# 上段: 元データと回帰直線
axes[0].scatter(x, y, color='tab:blue', alpha=0.75, label='Data')
axes[0].plot(x, y_fit, color='tab:orange', linewidth=2, label='Fit')
axes[0].set_ylabel('y')
axes[0].set_title('Residuals Plot')
axes[0].legend()

# 下段: 残差
axes[1].axhline(0, color='gray', linewidth=1)
axes[1].scatter(x, residuals, color='tab:green', alpha=0.75)
axes[1].set_xlabel('x')
axes[1].set_ylabel('Residuals')

plt.tight_layout()
plt.show()
```

![Residuals Plot](images/residuals_plot.png)


### 相関行列ヒートマップ
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)
data = np.random.randn(200, 4)
data[:, 1] = 0.6 * data[:, 0] + 0.4 * data[:, 1]
data[:, 2] = -0.5 * data[:, 0] + 0.3 * data[:, 2]

corr = np.corrcoef(data, rowvar=False)

# 相関行列をヒートマップ表示
plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.xticks(range(4), ['Var1', 'Var2', 'Var3', 'Var4'])
plt.yticks(range(4), ['Var1', 'Var2', 'Var3', 'Var4'])
plt.title('Correlation Heatmap')
plt.show()
```

![Correlation Heatmap](images/correlation_heatmap.png)

### まとめ

- **基本的な線・点・棒・面・円グラフ** から  
- **統計分布系（ヒストグラム・箱ひげ・バイオリン）**、  
- **密度マップ系（imshow・contour・hexbin）**、  
- **ベクトル／流体場（quiver・streamplot）**、  
- **極座標プロット**、  
- **3Dプロット** まで  
- **統計分析向けの可視化（回帰直線・残差・相関行列）** まで  

Matplotlib だけでこれだけ多彩な可視化が可能です。さらに Seaborn や pandas のラッパーを併用すると、より少ないコードで統計的可視化の表現力を高められます。まずは上記リストを参照しつつ、興味のあるプロットを試してみてください！
