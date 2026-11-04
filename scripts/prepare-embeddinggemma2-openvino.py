# /// script
# requires-python = ">=3.10"
# dependencies = ["onnx==1.23.2", "openvino==2026.4.1"]
# ///
"""Prepare the published EmbeddingGemma 2 Q8 export for the OpenVINO GPU EP."""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import onnx
from onnx import TensorProto, helper, numpy_helper
import openvino as ov


# These graph fingerprints identify the Q8 export whose operator semantics and
# GPU embeddings were checked. Reject graph revisions until validated as well.
GRAPHS = {
    "model_quantized": "d06edd601f851c633a2519304cbeb8dc6170d7ceb61b436625c17fb9b6e74953",
    "vision_encoder_quantized": "bb0de2df53a2448a32dc7908a187c168c8afd514d4d6f674f7f46024875fa4e3",
}


def explicit_rotary_positions(graph):
    # OpenVINO interprets the export's single zero as an index at every
    # position. ONNX Runtime interprets it as the start of the sequence.
    offset = "/model/constants/INT64/[0]"
    positions = "/model/positions/Range/output_0"
    nodes = list(graph.node)
    rotary = [n for n in nodes if n.domain == "com.microsoft" and n.op_type == "RotaryEmbedding"]
    index = next(i for i, n in enumerate(nodes) if n.op_type == "Range" and positions in n.output)
    zero = next(t for t in graph.initializer if t.name == offset)
    if (len(rotary) != 48 or any(n.input[1] != offset for n in rotary)
            or zero.data_type != TensorProto.INT64 or list(zero.dims) != [1]
            or numpy_helper.to_array(zero).tolist() != [0]):
        raise ValueError("Unexpected RotaryEmbedding offset in the Q8 graph")

    # The encoder processes one sequence per run. Make the full [1, S]
    # position tensor explicit using the graph's existing 0..S-1 sequence.
    ids = "dghs_rotary_position_ids"
    for node in rotary:
        node.input[1] = ids
    nodes[index + 1:index + 1] = [
        helper.make_node("Cast", [positions], ["dghs_rotary_range"], to=TensorProto.INT64),
        helper.make_node("Unsqueeze", ["dghs_rotary_range", offset], [ids]),
    ]
    del graph.node[:]
    graph.node.extend(nodes)


def convert(core, name, expected_hash, source, dest, work):
    filename = name + ".onnx"
    graph_bytes = (source / "onnx" / filename).read_bytes()
    if hashlib.sha256(graph_bytes).hexdigest() != expected_hash:
        raise ValueError(f"Unsupported Q8 graph revision: {filename}")
    model = onnx.load_from_string(graph_bytes)
    if name == "model_quantized":
        explicit_rotary_positions(model.graph)

    # HF snapshots use symlinks into a separate blobs directory. Put regular
    # files side by side for the ONNX frontend's external-data path checks.
    staged = work / filename
    weights_name = filename + "_data"
    weights = (source / "onnx" / weights_name).resolve(strict=True)
    staged_weights = work / weights_name
    try:
        os.link(weights, staged_weights)
    except OSError:
        shutil.copyfile(weights, staged_weights)
    onnx.save(model, staged)

    xml = dest / (name + ".openvino.xml")
    # serialize preserves the original precision (save_model defaults to FP16).
    ov.serialize(core.read_model(staged), xml)
    inputs, outputs = model.graph.input, model.graph.output
    context = helper.make_node(
        "EPContext", [v.name for v in inputs], [v.name for v in outputs],
        name=name, domain="com.microsoft", source="OpenVINOExecutionProvider",
        embed_mode=0, ep_cache_context=xml.name,
    )
    wrapper = helper.make_model(
        helper.make_graph([context], name, inputs, outputs),
        opset_imports=model.opset_import, ir_version=model.ir_version,
    )
    onnx.save(wrapper, xml.with_suffix(".onnx"))


def prepare(source, output):
    source = source.resolve(strict=True)
    output = output.parent.resolve(strict=True) / output.name
    user_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    hf_home = Path(os.environ.get("HF_HOME", user_cache / "huggingface"))
    hub = Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub")).resolve()
    protected = [source, hub]
    if source.parent.name == "snapshots" and source.parent.parent.name.startswith("models--"):
        protected.append(source.parents[2])  # snapshot from a non-default Hub cache
    if any(output.is_relative_to(path) for path in protected):
        raise ValueError("Choose an output directory outside the source export and Hub cache")
    if os.path.lexists(output):
        raise FileExistsError(f"Output already exists: {output}")

    with tempfile.TemporaryDirectory(prefix=".embeddinggemma2-", dir=output.parent) as tmp:
        tmp = Path(tmp)
        prepared = tmp / "prepared"
        dest = prepared / "onnx"
        dest.mkdir(parents=True)
        shutil.copyfile(source / "tokenizer.json", prepared / "tokenizer.json")
        core = ov.Core()
        for name, fingerprint in GRAPHS.items():
            convert(core, name, fingerprint, source, dest, tmp)
        if os.path.lexists(output):
            raise FileExistsError(f"Output already exists: {output}")
        prepared.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Q8 export root containing tokenizer.json and onnx/")
    parser.add_argument("output", type=Path, help="new model directory to create")
    args = parser.parse_args()
    try:
        prepare(args.source, args.output)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Conversion failed: {exc}\n")
    print(f"Prepared OpenVINO model in {args.output}")
