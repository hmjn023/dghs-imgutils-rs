#!/usr/bin/env python3
"""Build and run model contract tests in separate processes, retaining failures.

Example: python scripts/test-model-contracts.py --features openvino --backend openvino --device GPU --image /path/to/image.jpg
Each case is an opt-in real-model test. Network/cache access and a matching ORT
runtime are required. A signal, panic or timeout fails the overall command.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="")
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--device")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, default=Path("target/model-contract-results"))
    parser.add_argument("cases", nargs="*")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if args.image:
        args.image = args.image.resolve(strict=True)
    os.chdir(root)
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"DGHS_BACKEND": args.backend}
    if args.device:
        env["DGHS_ORT_DEVICE"] = args.device
    else:
        env.pop("DGHS_ORT_DEVICE", None)
    if args.image:
        env["DGHS_TEST_IMAGE"] = str(args.image)
    command = ["cargo", "test", "--no-default-features", "--test", "model_contracts", "--no-run", "--message-format=json"]
    if args.features:
        command += ["--features", args.features]
    build = subprocess.run(command, stdout=subprocess.PIPE, text=True, env=env)
    if build.returncode:
        return build.returncode
    artifacts = [json.loads(line) for line in build.stdout.splitlines() if line.startswith("{")]
    binaries = [a["executable"] for a in artifacts if a.get("reason") == "compiler-artifact" and a.get("executable") and a.get("target", {}).get("name") == "model_contracts"]
    if len(binaries) != 1:
        raise RuntimeError(f"expected one test binary, got {binaries}")
    binary = binaries[0]
    listing = subprocess.check_output([binary, "--list", "--ignored"], text=True, env=env)
    available = [line.removesuffix(": test") for line in listing.splitlines() if line.endswith(": test")]
    cases = args.cases or available
    if not cases or set(cases) - set(available):
        parser.error(f"choose cases from: {', '.join(available)}")
    results = []
    for case in cases:
        print(f"Running {case}", flush=True)
        log = args.output / f"{case}.log"
        try:
            with log.open("w") as stream:
                result = subprocess.run([binary, "--ignored", "--exact", case, "--nocapture"], env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout)
            code = result.returncode
            status = "passed" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            code, status = 124, "timeout"
        results.append({"case": case, "status": status, "returncode": code, "log": str(log)})
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"{case}: {status} ({code})", flush=True)
    return int(any(x["status"] != "passed" for x in results))


if __name__ == "__main__":
    sys.exit(main())
