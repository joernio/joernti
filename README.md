# JoernTI

This is the backend python package for the Type Inference plugin for the Joern code analysis platform (JoernTI).
This backend provides the interface to query and recover unknown types from objects using neural type inference.  

You can find our main repository, including documentation how to use this type inference backend together with Joern, under https://github.com/joernio/joernti-codetidal5.

## Getting Started

The following commands should get you up and running:

```shell
cd ./joernti
pip install -r requirements.txt # Install dependencies
pip install .                   # Install JoernTI
```

From this point onwards, you can use JoernTI as an executable by calling `joernti` on the command
line. This executable gives you access to a way to leverage the type inference models from the CLI.



Place your model weights (cf. [our CodeTIDAL5 repository](https://github.com/joernio/joernti-codetidal5)) in `./checkpoints` before starting the application.


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
```json
[{"target_identifier": "id", "type": "DataSet[]", "confidence": 0.6373817920684814}]`
```

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

