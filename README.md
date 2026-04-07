# PolySLGen: Online Multimodal Speaking–Listening Reaction Generation in Polyadic Interaction

## Getting Started 

### 1. Environment Setup

```bash
conda env create --file environment.yml
conda activate polyslgen
pip install git+https://github.com/resemble-ai/monotonic_align.git 
```


### 2. Datasets
Download preprocessed dataset from [here](https://surfdrive.surf.nl/s/rAo9aL2PcDrpdGL). Alternatively, you can download it via:
```bash
cd src/prepare
sh download_data.sh
```

### 3. Checkpoints

* **Llama3-8B-Instruct**
Download from [here](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and place it under `./checkpoints/`.
Alternatively:
    ```python
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        local_dir="./checkpoints/Meta-Llama-3-8B-Instruct",
        local_dir_use_symlinks=False,
        token="your_hf_token_here"
    )
    ```

* **StyleTTS2**
    - Follow the instruction from [StyleTTS2](https://github.com/yl4579/StyleTTS2) to set up the working directory `./checkpoints/StyleTTS2/`.
    - Download the pretrained model from [here](https://huggingface.co/yl4579/StyleTTS2-LibriTTS/tree/main) and place it in `./checkpoints/StyleTTS2/StyleTTS2-LibriTTS/`.

    Alternatively:
    ```bash
    cd src/prepare
    sh download_StyleTTS2.sh
    ```

* **PolySLGen & Motion Evaluator**
Download the pre-trained [PolySLGen model](https://surfdrive.surf.nl/s/TRoJE6L52G4eFDy) and the [motion evaluator](https://surfdrive.surf.nl/s/fTq5T7nkAXEnAjG). 
Alternatively,
    ```bash
    cd src/prepare
    sh download_pretrained.sh
    ```

## Training
Update the dataset, checkpoint, and output paths in `run_train.py`, then run:
```bash
python run_train.py
```

## Evaluation
To evaluate a pre-trained model, set the following in `run_train.py`:
```python
eval_only = 1 # whether to only run evaluation or not
pretrained_model_dir = './pretrained_model/' # path to the pre-trained model folder
```

## Acknowledgements
This implementation builds upon [OneLLM](https://github.com/csuhan/OneLLM).

## Citation

