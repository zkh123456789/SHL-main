# SHL - Pattern Recognition 2026
This repository is an official implementation of the paper "Learning semantic-spatial hierarchies representation for image super-resolution of remote sensing", Pattern Recognition, 2026. 


## :hammer:Environment
- Python 3.9
- PyTorch >=2.2

### Installation
``
pip install -r requirements.txt
python setup.py develop

```

### Training Commands

```
 python basicsr/train.py -opt options/train/test_SHL_x4.yml

```

### Testing Commands

```
python basicsr/test.py -opt options/test/test_SHL_x4.yml

```

## 🥰Acknowledgements

This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR) and  [CATANet](https://github.com/EquationWalker/CATANet) . Thanks for their good work.
