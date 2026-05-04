# SPFormer

## 1. 配环境

建议环境：

- Linux
- Python 3.8
- CUDA
- 已安装 PyTorch

安装依赖：

```bash
pip install -r requirements.txt
```

安装 `pointgroup_ops`：

```bash
cd spformer/lib
pip install -v -e .
cd ../..
```

数据集目录：

```text
data/myplants/
├── train/
├── val/
└── test/
```

程序会读取：

```text
data/myplants/train/*.pth
data/myplants/val/*.pth
data/myplants/test/*.pth
```

配置文件：

```bash
configs/myplants.yaml
```

## 2. 训练

```bash
python tools/train.py configs/myplants.yaml --work_dir exps/myplants
```

继续训练：

```bash
python tools/train.py configs/myplants.yaml --work_dir exps/myplants --resume exps/myplants/lastest.pth
```

## 3. 测试

```bash
python tools/test.py configs/myplants.yaml exps/myplants/epoch_0512.pth --out exps/myplants/test_results
```

只导出结果不评测：

```bash
python tools/test.py configs/myplants.yaml exps/myplants/epoch_0512.pth --out exps/myplants/test_results --no-eval
```
