import json
import socket
from typing import List, Optional, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from joernti.domain import InferenceSolution, TypeInferenceModel, ProgramUsageSlice, MethodUsageSlice


class LLMTypeInferenceModel(TypeInferenceModel):
    """
    A type inference model that looks at a variable and scope names and the procedure invocations made
    to recover a TypeHint given a UsageSlice.
    """

    def __init__(self,
                 checkpoint_path: str = None,
                 language: Optional[str] = None,
                 confidence_threshold: float = None,
                 ):
        """
        Instantiates an instance of a re-usable type inference querying engine.

        :param checkpoint_path: local path to a checkpoint file for the model,
            including model layout definitions and weights.
        :param language: Programming language of the slice to run inference on.
        """

        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path
        else:
            self.checkpoint_path = "joernio/codetidal5"

        self.confidence_threshold = confidence_threshold
        self.language = language

        print("[+] Initializing Large Language Model")
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-220m", add_prefix_space=True, use_fast=True)
        self.preamble = self.tokenizer(["Infer", "types", "for", language, ":"], is_split_into_words=True,
                                       truncation=False, add_special_tokens=False).input_ids

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint_path)

        print("[*] Initialization complete")

    def solve(self, problem: MethodUsageSlice, scope=None) -> List[InferenceSolution]:
        """
        Attempts to solve an InferenceProblem by invoking our ML model, finetuned on a diverse set of Usage Slices.
        :param problem: a specification of observations about a receiver and invoked procedures either as
            a raw json-encoded slice or as InferenceProblem.
        :param scope: the procedure scope of the problem.
        :return: a solution which describes the most likely type and potential alternatives. If there is no
            similar-enough type, the result will be empty.
        """

        solutions = []
        code_contents = problem.source

        targets = []
        # collect target objects and where they are defined from slice
        for obj in problem.slices:
            targets.append(obj.target_obj.name)

        # match objects in source code
        identifiers = [0] * len(targets)
        var_idx = {}
        for i, v in enumerate(targets):
            curr_cont_idx = 0

            while curr_cont_idx < len(code_contents):
                curr_cont_idx = code_contents.find(v, curr_cont_idx)
                if curr_cont_idx != -1:
                    if code_contents[curr_cont_idx + len(v)] in [" ", ":", ",", ";", ")"]:
                        var_idx[v] = curr_cont_idx
                        break
                    else:
                        curr_cont_idx += len(v)
                else:
                    break

            if i > 100:
                print("[!] Inferring more than 100 variables per method not supported. Please limit your slices.")
                break

        var_idx_sorted = sorted(var_idx.items(), key=lambda x: x[1])
        i = 0
        offset = 0

        # insert special tokens to guide model attention
        for v, idx in var_idx_sorted:
            curr_id = '<extra_id_{}>'.format(i)
            code_contents = code_contents[:idx + len(v) + offset] + ": {} ".format(curr_id) + code_contents[
                                                                                              idx + len(v) + offset:]

            identifiers[i] = v
            offset += len(curr_id) + 3
            i += 1

        encoded_input = self.tokenizer(code_contents, truncation=True, max_length=768, add_special_tokens=False)[
            'input_ids']

        # add CLS & EOS
        assembled_input = [1] + self.preamble + encoded_input + [2]

        print("[i] Running inference for method scope `{}`".format(scope))
        outputs = self.model.generate(torch.as_tensor(assembled_input).unsqueeze(0), max_new_tokens=64)
        outputs = np.where(outputs != -100, outputs, self.tokenizer.pad_token_id)
        answer = self.tokenizer.decode(outputs[0])

        # extract predictions
        if "No types inferred." not in answer:
            for pred in answer.split('\n')[:-1]:
                var_id, type_pred = pred.split(" ", 1)
                type_name = type_pred.replace(" ", "")
                var_id = int(var_id.split("id_")[-1].strip(">"))

                if var_id < len(identifiers):
                    obj_name = identifiers[var_id]
                    curr_sol = InferenceSolution(obj_name, type_name, float(1),
                                                 scope, [])

                    solutions.append(curr_sol)

        return solutions


def slice_to_solutions(ti_model: LLMTypeInferenceModel, program_slice: ProgramUsageSlice) -> List[Dict]:
    type_annotations = []
    for k in program_slice.object_slices.keys():
        solutions = ti_model.solve(program_slice.object_slices[k], scope=k)

        for sol in solutions:
            type_annotations.append({'target_identifier': sol.target_identifier,
                                     'type': sol.inferred_type,
                                     'confidence': sol.inferred_type_distance,
                                     'scope': sol.scope,
                                     'alternatives': []
                                     })
    return type_annotations


def main(input_slice: Optional[str] = None, checkpoint: Optional[str] = None, language: Optional[str] = None,
         server_mode: Optional[bool] = False, port: Optional[int] = 1337):
    """Entrypoint for ML-based inference task."""
    if input_slice is None and server_mode is False:
        print("[!] No input and not running in server mode! Bailing out...")
        exit(1)

    print("[+] Initializing ML Model...")
    ti_model = LLMTypeInferenceModel(checkpoint_path=checkpoint, language=language)

    if server_mode:
        print("[*] Running in server mode...")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                while True:
                    s.listen()
                    conn, addr = s.accept()
                    with conn:
                        print("[+] New Job incoming")
                        payload = b''
                        while True:
                            data = conn.recv(1024)
                            payload += data
                            # we expect the client to send a json-encoded program slice, terminated by '\r\r'
                            if not data or b'\r\r' in data:
                                break

                        program_slice = ProgramUsageSlice.from_json(json.loads(payload[:-2].decode()))
                        type_annotations = slice_to_solutions(ti_model, program_slice)
                        # send back json-encoded type annotations
                        conn.send(json.dumps(type_annotations).encode())
                        print(
                            "[*] Inference completed: Sent response with {} predictions".format(len(type_annotations)))

        except KeyboardInterrupt:
            print("[!] Exiting Type Inference Server...")
    else:
        with open(input_slice, 'r') as f:
            program_slice = ProgramUsageSlice.from_json(json.load(f))
            type_annotations = slice_to_solutions(ti_model, program_slice)
            print(type_annotations)
