# RecBole example on cookpad blog

## xDeepFM

### `run_recbole` を使った訓練方法。

```shell
python run.py --dataset_name example --config_file configs/basic.yaml
```

### `run_recbole` を使わない訓練方法。自分でモデルなどを変更したい場合にはこちらを用いる。

```shell
python run_manually.py
```

### `run_recbole` を使わない訓練方法。ハイパーパラメータの自動調整をしたい場合にはこちらを用いる。

```shell
python run_with_hparam_tuning.py
```

ハイパーパラメータの探索範囲は `configs/hyperparams/xdeepfm.hyper` のように設定する。
.hyper ファイルの記法は[こちら](https://recbole.io/docs/v1.0.0/user_guide/usage/parameter_tuning.html)を参考ください。

NOTE: `HyperTuning` を使うとなぜか最初のトライアルで生成されたログファイルにすべてのトライアルのログが記録される。その他のトライアルで生成されたログファイルには何も書き込まれず、空のファイルが生成されることに注意する。

## 状況 feature を加味したいとき

以下のように `.user`, `.item`, `.inter` に状況 feature を追加してコンフィグを追記する。

### .user と .inter にそれぞれ特徴量を追加する例

`dataset/example/example.user` に new_feature を追記する。

```
user_id:token	feature1:token	feature2:token	new_feture:token
1	286	130	987
2	491	3	876
3	342	32	765
```

`dataset/example/example.inter` に weather_id,, precipitation を追記する。

```
user_id:token	item_id:token	timestamp:float	weather_id:token	precipitation:float
1	1	1630461974	1	1
2	2	1630462246	2	2
3	2	1630462432	3	1
1	2	1630462532	1	5
1	3	1630462632	2	10
2	1	1630462732	3	3
2	2	1630462832	1	8
2	3	1630462932	2	7
3	1	1630463032	3	3
3	2	1630463132	1	6
3	3	1630463232	2	9
```

これに合わせて `configs.basic.yaml` の `load_col` を以下のように修正する。

```yaml
load_col:
     inter: [user_id, item_id, timestamp, weather_id, precipitation]
     user: [user_id, feature1, feature2, new_feature]
     item: [item_id, item_name, item_category_id]
```

## .inter に状況 feature を加えると KeyError が起こる原因調査

train のための forward の際には .user, .item, .inter の features が入力される。
しかし valid のための foward の際には .user の features のみが入力される。
このため、今回のケースで推論時に入力したいはずの .inter 内の features が入力されないということが起こっている。

[こちら](https://recbole.io/docs/user_guide/usage/running_new_dataset.html#convert-to-dataloader)によるとコンフィグファイルの eval_args を調整することで、そのあたりをコントロールできるかもしれない。

### TODO

- [x] eval_args の調整によって .inter の特徴量を .user と組み合わせて interaction を作成できないか調べる。

### 解決策

eval_args を以下のように変更することで解決した。

```diff
- mode: full
+ mode: uni001
```

eval_args 全コンフィグは以下である。

```yaml
eval_args:
    group_by: user  # user 単位でアイテムを集約して評価に使う。基本的にこれ以外使うことはない
    order: TO  # Temporal Order。時系列順で train, valid, test を分けてくれる
    split: {'RS': [0.8,0.1,0.1]}  # 80%, 10%, 10% で分けてくれる
    mode: uni001  # 1つのポジティブサンプルにつき、いくつのネガティブサンプルをどんな分布から何回行うか. uni001 は一様分布から1回行うの意. pop001 なども使える.
```
