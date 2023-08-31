# JoernTI

Joern Type Inference (JoernTI) is plugin for the Joern code analysis platform to recover unknown types from objects using neural type inference.  
You can find our main repository, including documentation how to use this type inference backend together with Joern, under https://github.com/joernio/joernti-codetidal5.

## Getting Started

The following commands should get you up and running:

```shell
pip install -r requirements.txt # Install dependencies
pip install .                   # Install JoernTI
```

From this point onwards, you can use JoernTI as an executable by calling `joernti` on the command
line. This executable gives you access to a way to leverage the type inference models from the CLI.

For experimenting with the ML model and the datasets used in `./experiments`, install the dependencies incl. CUDA and 
PyTorch 2.0 (GPU required):
```shell
cd ./experiments
./install_cuda_pytorch.sh
```

At this point you'll have the JoernTI system installed **without** model weights. To fetch the desired
model, see `./data/model_checkpoints`.

## Models

The default usage for LLM's uses the `llm` command. Given extracted usage slices, the model pipeline transforms these 
into numerical representations of word tokens which a multi-head transformer model then tries to learn and infer the 
data type.

The named models below simply wrap this command with the supported target language to infer types on and the model's 
weights at some checkpoint.

### CodeTIDAL5

`CodeTIDAL5`: Given extracted usage slices from JavaScript code, the CodeT5-based encode-decoder model infers types for
the target objects.

**Usage**:
The model can either be used for one-shot inference with
```
joernti codetidal5 --input-slice /path/to/extraced_slice
```
or as a continuously running endpoint with
```
joernti codetidal5 --run-as-server
```
In that case, connect to the server on port 1337 (default) via a TCP socket and send the raw json-encoded slice. The response will be of the following format:
`[{"target_identifier": "id", "type": "DataSet[]", "confidence": 0.6373817920684814}]`

You can find scripts and instructions how to generate a training dataset for type inference with a decoder model such as CodeT5 in `./training_dataset`.

## Structure

The following describes the structure of the project other than the models.

### `joernti`

The main Python package that holds the models and helper functions.

- `domain`: Where classes that are commonly used by each model belong.
- `util`: Some helper functions for manipulating data.

### `data`

Holds the datasets/trained models/corpora for the models to access.

## Obtaining Slices with Joern

This project operates on input obtained via Joern. Here is a short tutorial on 
obtaining these "usage slices":

```
joern-slice usages /path/to/project -o /path/to/slice/output/slices.json
```

The usage slice will be at `/path/to/slice/output/slices.json` which is what the models in this
directory both train and classify.

## Citation
If you use JoernTI / CodeTIDAL5 in your research or wish to refer to the baseline results, we kindly ask you to cite us:
```bibtex
@inproceedings{joernti2023,
  title={Learning Type Inference for Enhanced Dataflow Analysis},
  author={Seidel, Lukas and {Baker Effendi}, David and Pinho, Xavier and Rieck, Konrad and {van der Merwe}, Brink and Yamaguchi, Fabian},
  booktitle={28th European Symposium on
Research in Computer Security (ESORICS)},
  year={2023}
}
```

## Related Work

###### [Deep Learning Type Inference](https://vhellendoorn.github.io/fse2018-j2t.pdf), ACM ESEC/FSE '18

###### [Augmenting Decompiler Output with Learned Variable Names and Types](https://www.usenix.org/system/files/sec22-chen-qibin.pdf), USENIX '22

###### [LAMBDANET: PROBABILISTIC TYPE INFERENCE USING GRAPH NEURAL NETWORKS](https://openreview.net/pdf?id=Hkx6hANtwH), ICLR '20, [OpenReview](https://openreview.net/forum?id=Hkx6hANtwH)

###### [Probabilistic Type Inference by Optimising Logical and Natural Constraints](https://arxiv.org/pdf/2004.00348.pdf), arXiv preprint '20

###### [Advanced Graph-Based Deep Learning for Probabilistic Type Inference](https://arxiv.org/pdf/2009.05949.pdf), arXiv preprint '20

###### [Learning type annotation: is big data enough?](https://dl.acm.org/doi/abs/10.1145/3468264.3473135), ESEC/FSE `21

###### [FlexType: A Plug-and-Play Framework for Type Inference Models](https://dl.acm.org/doi/abs/10.1145/3551349.3559527), ASE '22

###### [Learning to Predict User-Defined Types](https://dl.acm.org/doi/10.1109/TSE.2022.3178945) ICSE`23

