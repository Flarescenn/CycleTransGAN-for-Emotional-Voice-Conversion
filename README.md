# CycleTransGAN for Emotional Voice Conversion

This repository contains a **PyTorch implementation** of a CycleTransGAN for Emotional Voice Conversion (EVC). The model is designed to transform the emotions in a speech. (e.g., from a neutral to an angry tone) while preserving the original speaker's identity.

> This is a Pytorch implementation of CycleTransGAN inspired by [this repo.](https://github.com/CZ26/CycleTransGAN-EVC) Much credit to the original authors!

---

### Quick Rundown of the Features:


- **CycleGAN Architecture**  
  Trained on **unpaired data**. No need for parallel utterances of the same speaker in the two different emotions.

- **Gated CNN + Transformer Generator**  
  Transformer layers in the generator to capture both local and long-range dependencies.

- **Stable Training with WGAN-GP**  
  Uses **Wasserstein GAN with Gradient Penalty (WGAN-GP)** to stabilize training and mitigate mode collapse.

- **WORLD Vocoder for Preprocessing**  
  Uses [`pyworld`](https://github.com/mmorise/World) to decompose audio into:
  - F0 (fundamental frequency)
  - MCEPs (Mel-Cepstral Coefficients)
  - Aperiodicity  

##  Setup

Clone the repository and install the dependencies as mentioned in the `requirements.txt`

```bash
 git clone https://github.com/Flarescenn/CycleTransGAN-for-Emotional-Voice-Conversion.git
 cd CycleTransGAN-for-Emotional-Voice-Conversion
 pip install -r requirements.txt
```

Followed by the dataset preparations, for instance:
- Source audio files in `./data/neutral/`
- Target audio files in `./data/target_emotion/`

### Training
Training is done in multiple stages, starting with shorter audio frame lengths, increasing them with every stage. This "curriculum learning" approach helps stabilize training.

```bash
bash run.sh
```
The script automates the multi-stage training by calling `train.py` with different frame lengths (`--num_f`) as below:

```bash
python train.py --num_f 128 \
                --train_A_dir './data/neutral/' \
                --train_B_dir './data/target_emotion/' \
                --model_dir './model/' \
                --model_name 'model.ckpt' \
                --output_dir './validated/'
```
