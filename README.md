# Universal Deep Image Compression via Content-Adaptive Optimization with Adapters
Official implementation of "Universal Deep Image Compression via Content-Adaptive Optimization with Adapters" presented at WACV 23.

## Environment
Recommend to use Miniconda.

```bash
# python=3.8, 3.9, or 3.10 should be fine.
conda create -n aof python=3.7
conda activate aof
# install 7z for [Lam+, ACM MM 20]
conda install -c bioconda p7zip
pip install -r requirements.txt
```

## Preparation
### Dataset
Prepare a dataset that consists of four domains: natural images, line drawings, comics, and vector arts.

```bash
# In 2022/08/31, four files have missing links.
# the four files: (`156526161`, `99117125`, `15642096`, `158633139`)
python scripts/download_dataset.py
```

### Pre-trained Weights
Prepare the weights of WACNN pre-trained on natural images.

```bash
bash scripts/download_pretrained_models.bash
```

## Usage
### Training of the Proposed Method
Apply the demo code (`main.py`) to a dataset by running a script.
Please specify `["vector", "natural", "line", "comic"]` as a dataset and `[1, 2, 3, 4, 5, 6]` as quality.

```bash
python run_main.py vector --out results/ours --quality 1
```

The refined latent representation is encoded and saved in `cache/`.
The adapter parameters are encoded and saved in `results/ours/wacnn/q1/<image file name>/weights.pt`.

### Evaluation of the Proposed Method
Perform evaluation using the compressed data obtained in training with the command below.
Please specify `--domain` from `["vector", "natural", "line", "comic"]`. The default is `"vector"`.

```bash
# Without adaptation
python decode.py --stage 0th # --domain vector
# Without adapters (= only refining the latent representation = [Yang+, NeurIPS 20])
python decode.py --stage 1st # --domain vector
# Ours
python decode.py --stage 2nd # --domain vector
```

You can finally obtain the results in the csv format in `results/ours/wacnn/q{1,2,3,4,5,6}/<dataset name>_{0th,1st,2nd}.csv`.

### Comparison with Other Adaptive Methods
Run other adaptive methods in our framework.

```bash
# [Lam+, ACM MM 20]
python run_main.py vector --out results/lam-mm20 --regex "'g_s\..*\.bias'" --n-dim 0 --width 0.0 --data_type float32 --quality 1
python decode.py --weight_root results/lam-mm20 --n-dim-2 0 --width 0.0 --regex "g_s\..*\.bias" --data_type float64+7z

# [Rozendaal+, ICLR 21]
python run_main.py vector --out results/rozendaal-iclr21 --regex "'.*'" --n-dim 0 --width 0.005 --alpha 1000 --sigma 0.05 --distrib spike-and-slab --lr 3e-5 --opt-enc --quality 1
python decode.py --weight_root results/rozendaal-iclr21 --rozendaal

# [Zou+, ISM 21]
python run_main.py vector --out results/zou-ism21 --n-dim -1 --groups 192 --width 0.0 --quality 1
python decode.py --weight_root results/zou-ism21 --n-dim-2 -1 --groups 192 --width 0.0
```

You can compare these methods with the baseline and proposed methods using the command below.

```bash
python plot_rdcurve.py
```

### Ablation Studies
(Optional) Perform ablation studies.

* Optimization only in terms of distortion
```bash
# Distortion opt.
python run_main.py vector --out results-abl/ours-dopt --width 0.0 --quality 1
python decode.py --weight_root results-abl/ours-dopt --width 0.0
```

* Optimization of other parameters

```bash
# Biases
python run_main.py vector --out results-abl/ours-bias --regex "'g_s\..*\.bias'" --n-dim 0 --lr_2 1e-5 --quality 1
python decode.py --weight_root results-abl/ours-bias --regex "'g_s\..*\.bias'" --n-dim-2 0
# OMPs
python run_main.py vector --out results-abl/ours-omp --distrib logistic --n-dim -1 --groups 192 --quality 1
python decode.py --weight_root results-abl/ours-omp --n-dim-2 -1 --groups 192
```

* Optimization Order

```bash
python run_main.py vector --out results-abl/ours-swap --swap --quality 1
python decode.py --weight_root results-abl/ours-swap --swap
```

* Another Base Network Architecture

```bash
# Cheng20
python run_main.py vector --out results/ours --model cheng2020-attn --regex "'g_s\.[8-8]\.adapter_1.*'" --quality 1
python decode.py --model cheng2020-attn --n-dim-1 2 --n-dim-2 0
```

## Contact
Feel free to contact me if there is any question: tsubota (a) hal.t.u-tokyo.ac.jp

## License
This code is licensed under MIT (if not specified in the code).

The code contains modified and copied open-source code.
Thus, I describe the original license of the code.
Please let me know if there is a license issue with code redistribution.
If so, I will remove the code and provide the instructions to reproduce the work.

As for the dataset, I do not have any rights and provide only URLs.
Please refer to the original datasets (Kodak and BAM) for the detailed license of the images.
When the link of some URLs is missing, I will redistribute the corresponding images if they are licensed under Creative Commons.
