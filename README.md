# EMO_Harmonizer
This is the official repository of **Emotion-Driven Melody Harmonization via Melodic Variation and Functional Representation**. 

* [demo page](https://yuer867.github.io/emo_harmonizer/)

## Environment
* **Python 3.8** and **CUDA 10.2** recommended
* Install dependencies (required)
```angular2html
pip install -r requirements.txt
```

* Install [fast transformer](https://github.com/idiap/fast-transformers) (required)
```
pip install --user pytorch-fast-transformers
```

* Install [midi2audio](https://github.com/bzamecnik/midi2audio) to synthesize generated MIDI to audio (optional)
```
pip install midi2audio
wget https://freepats.zenvoid.org/Piano/SalamanderGrandPiano/SalamanderGrandPiano-SF2-V3+20200602.tar.xz
tar -xzvf SalamanderGrandPiano-SF2-V3+20200602.tar.xz
```

## Quick Start

To make **emotion-driven melody harmonization** with our trained models, follow the steps as below.

### Best combination
**functional representation with rule-based key determination**
1. Download and unzip [EMOPIA events](https://drive.google.com/file/d/1IqisQe_bYAfUHZ__ioJrIrBkmn_8KLV_/view?usp=sharing) and the [checkpoint](https://drive.google.com/file/d/1oKJf3EYx4EnKARtUYfAes1AnBHNLo-U6/view?usp=sharing) (make sure you're in repository root directory).
2. Harmonize randomly selected melody sequences from EMOPIA dataset.
```angular2html
# output midi files
python3 inference.py \
        --configuration=config/emopia_finetune.yaml \
        --representation=functional \
        --key_determine=rule \
        --inference_params=emo_harmonizer_ckpt_functional/best_params.pt \
        --output_dir=generation/emopia_functional_rule

# output midi files and synthesized audio
python3 inference.py \
        --configuration=config/emopia_finetune.yaml \
        --representation=functional \
        --key_determine=rule \
        --inference_params=emo_harmonizer_ckpt_functional/best_params.pt \
        --output_dir=generation/emopia_functional_rule \
        --play_midi
```
### Other inference options

1. Download and unzip **all** [checkpoints](https://drive.google.com/file/d/1v5iaw_sf0HgEaeOntVIIerykm5BGGf8y/view?usp=sharing) (make sure you're in repository root directory).
2. Harmonize randomly selected melody sequences from EMOPIA dataset. Please modify the inference command with your preferred data `representation` and `key_determine` method, along with the specified `inference_params` (for more details, please refer to [inference.py](https://github.com/Yuer867/EMO_Harmonizer/blob/main/inference.py#L314)). 
Two more examples are provided below:
```angular2html
# REMI representation (transpose to C) with rule-based key determination
python3 inference.py \
        --configuration=config/emopia_finetune.yaml \
        --representation=transpose \
        --key_determine=rule \
        --inference_params=emo_harmonizer_ckpt/emopia_transpose/best_params.pt \
        --output_dir=generation/emopia_transpose_rule

# functional representation with model-based key determination
python3 inference.py \
        --configuration=config/emopia_finetune.yaml \
        --representation=functional \
        --key_determine=model \
        --inference_params=emo_harmonizer_ckpt/emopia_functional_model/best_params.pt \
        --output_dir=generation/emopia_functional_model
```

## Train the model by yourself
**Note**: Please consider the combination of **functional representation** with **rule-based key determination** as an example. 
Other options follow a similar format for inference.
1. Use the provided events ([EMOPIA](https://drive.google.com/file/d/1IqisQe_bYAfUHZ__ioJrIrBkmn_8KLV_/view?usp=sharing), [HookTheory](https://drive.google.com/file/d/1gBBRiX7UM0uUP57ofXerIdZgul37fmKC/view?usp=sharing)) directly or convert MIDI to events following the [steps](https://github.com/Yuer867/EMO_Harmonizer/tree/main/representations#readme).
2. Pre-train the model on `HookTheory`.
```angular2html
python3 train.py \
        --configuration=config/hooktheory_pretrain.yaml \
        --representation=functional \
        --key_determine=rule
```
3. Finetune the model on `EMOPIA`.
```angular2html
python3 train.py \
        --configuration=config/emopia_finetune.yaml \
        --representation=functional \
        --key_determine=rule
```

