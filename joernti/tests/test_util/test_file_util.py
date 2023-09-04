from joernti.domain import DefComponent, CallEntry, ParamEntry
from joernti.util import file_util


def test_deserialize_object_slices(request):
    file = str(request.config.rootdir) + "/data/test_slice"
    slice_map = file_util.deserialize_slices(file)[0].object_slices
    k = "foo.js::program"
    assert "foo.js::program" in slice_map.keys()
    method_slice = slice_map[k]
    obj_slice = method_slice.slices[0]
    assert method_slice.source.startswith("const express = require('express')")
    assert DefComponent("app", "ANY") == obj_slice.target_obj
    assert DefComponent("express", "ANY") == obj_slice.defined_by
    assert obj_slice.invoked_calls == (
        CallEntry("get", [ParamEntry("__ecma.String"), ParamEntry("LAMBDA")], [ParamEntry("ANY")]),
        CallEntry("listen", [ParamEntry("__ecma.Number"), ParamEntry("LAMBDA")], [ParamEntry("ANY")]),
    )
    assert obj_slice.arg_to_calls == (
        (CallEntry("log", [ParamEntry("ANY")], [ParamEntry("ANY")]), 1),
        (CallEntry("debug", [ParamEntry("ANY")], [ParamEntry("ANY")]), 1),
    )
