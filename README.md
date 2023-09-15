# JoernTI

This is the backend python package for the Type Inference plugin for the Joern code analysis platform (JoernTI).
This backend provides the interface to query and recover unknown types from objects using neural type inference.  

You can find our main repository, including d ocumentation on how to use this type inference backend together with Joern, under https://github.com/joernio/joernti-codetidal5.

## Getting Started

The following commands should get you up and running:

```shell
cd ./joernti
pip install -r requirements.txt # Install dependencies
pip install .                   # Install JoernTI
```

From this point onwards, you can use JoernTI as an executable by calling `joernti` on the command line. 
This executable gives you access to a way to leverage the type inference models from the CLI.

A full proof of concept for the integration with Joern can be found [here](https://github.com/joernio/joernti-codetidal5).

### Usage
The model can either be used as a queryable, continuously running endpoint with
```shell
joernti codetidal5 --run-as-server
```

In that case, connect to the server on port 1337 (default) via a TCP socket and send the raw JSON-encoded slice. The response will be of the following format:
```json
[{"target_identifier": "id", "type": "DataSet[]", "confidence": 0.6373817920684814}]`
```

### Model: CodeTIDAL5

If no local model (checkpoint) is specified, a remote snapshot of the
CodeTIDAL5 model will be fetched from our [Hugging Face repository](https://huggingface.co/joernio/codetidal5).

Given extracted usage slices from JavaScript code and the raw source code of the corresponding scope, e.g., function, the CodeT5-based encode-decoder model infers types for the target objects.

## Obtaining Slices with Joern

This project uses slices obtained via Joern as part of the type inference request. 
Here is a short example on obtaining these _usage slices_:

```shell
joern-slice usages /path/to/project -o /path/to/slice/output/slices.json
```

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

