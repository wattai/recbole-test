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

以下のように `.user` に状況 feature を追加してコンフィグを追記する。

### `.user`

```
user_id:token	feature1:token	feature2:token	new_feture:token
1	286	130	987
2	491	3	876
3	342	32	765
```

```yaml
load_col:
     inter: [user_id, item_id, timestamp]
     user: [user_id, feature1, feature2, new_feature]
     item: [item_id, item_name, item_category_id]
```
