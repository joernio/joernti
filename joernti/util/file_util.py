import json
import os
from typing import List

from joernti.domain import ProgramUsageSlice


def deserialize_slices(tgt_dir: str) -> List[ProgramUsageSlice]:
    """
    Deserializes all the object slices in the nested slice output directory.

    :param tgt_dir: the route directory of the slice extraction output.
    :return: a mapping between the method scopes and their object slices.
    """
    if not os.path.isdir(tgt_dir):
        raise RuntimeError("Given path is not a directory! '{}'", tgt_dir)
    slices = []
    for f in os.listdir(tgt_dir):
        if f.endswith(".json"):
            with open(tgt_dir + os.sep + f, 'r') as fh:
                slices.append(ProgramUsageSlice.from_json(json.load(fh)))
    return slices
