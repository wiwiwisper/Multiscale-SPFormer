# SPFormer

这个仓库提供 `myplants` 数据集上的训练和测试入口。

## 环境

建议使用 Linux + CUDA + Python 3.8。

先安装 Python 依赖：

```bash
pip install -r requirements.txt
```

## 安装 `pointgroup_ops`

训练和测试依赖 `pointgroup_ops`，需要单独安装：

```bash
cd spformer/lib
pip install -v -e .
cd ../..
```

## 数据集目录

配置文件固定读取：

```bash
data/myplants
```

并且训练、验证、测试数据必须分别放在这三个子目录下：

```text
data/myplants/
├── train/
├── val/
└── test/
```

程序会直接扫描下面这些路径中的 `.pth` 文件：

```text
data/myplants/train/*.pth
data/myplants/val/*.pth
data/myplants/test/*.pth
```

## 单个样本格式

当前 `myplants` 数据集读取器支持 `.pth` 和 `.npz`，但配置里默认使用 `.pth`。

推荐每个 `.pth` 文件保存 5 个对象，顺序如下：

```python
(xyz, rgb, superpoint, semantic_label, instance_label)
```

其中：

- `xyz`: `float32`, 形状 `[N, 3]`
- `rgb`: `float32`, 形状 `[N, 3]`
- `superpoint`: `int`, 形状 `[N]`
- `semantic_label`: `int`, 形状 `[N]`
- `instance_label`: `int`, 形状 `[N]`

如果你的语义标签是 `1/2`，数据集类里会自动减到 `0/1`。

## 配置文件

默认配置文件：

```bash
configs/myplants.yaml
```

如果你要改数据路径、batch size、epoch 数等，直接修改这个文件。

## 训练

```bash
python tools/train.py configs/myplants.yaml --work_dir exps/myplants
```

训练输出会写到：

```text
exps/myplants/
```

主要包括：

- 日志文件 `*.log`
- TensorBoard 日志
- 检查点 `lastest.pth`
- 周期保存的 `epoch_XXXX.pth`
- 验证指标 `Evaluation_metrics.log`

## 继续训练

```bash
python tools/train.py configs/myplants.yaml --work_dir exps/myplants --resume exps/myplants/lastest.pth
```

## 测试

```bash
python tools/test.py configs/myplants.yaml exps/myplants/epoch_0512.pth --out exps/myplants/test_results
```

如果只想导出预测结果，不做评测：

```bash
python tools/test.py configs/myplants.yaml exps/myplants/epoch_0512.pth --out exps/myplants/test_results --no-eval
```

测试输出目录示例：

```text
exps/myplants/test_results/
├── Evaluation_metrics.log
├── pred_instance/
└── gt_instance/
```

## 常见检查

检查训练集是否被正确读取：

```bash
find data/myplants/train -name '*.pth' | head
find data/myplants/val -name '*.pth' | head
find data/myplants/test -name '*.pth' | head
```

检查扩展是否可导入：

```bash
python -c "import pointgroup_ops; print('ok')"
```

## 备注

- 如果 `pointgroup_ops` 没安装成功，训练和测试都无法正常运行。
- `work_dir` 可以改成你自己的输出目录。
