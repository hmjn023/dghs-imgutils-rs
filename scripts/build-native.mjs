import { spawnSync } from "node:child_process";

const args = ["build", "--platform", "--dts", "index.d.ts"];

if (process.argv.includes("--release")) {
  args.push("--release");
}

// Provider bindings are dynamically loaded at runtime.  Including them in
// the native addon lets the library select a compatible host installation;
// it does not download or bundle CUDA/OpenVINO/driver libraries.
if (process.platform === "linux") {
  args.push("--features", "cuda,openvino");
} else if (process.platform === "win32") {
  args.push("--features", "openvino");
}

args.push("--cargo-flags=--lib");

const result = spawnSync("napi", args, {
  stdio: "inherit",
  // npm exposes the local `napi.cmd` shim through cmd.exe on Windows.
  shell: process.platform === "win32",
});
if (result.error) {
  console.error(`failed to start napi build: ${result.error}`);
  process.exit(1);
}
if (result.signal) {
  console.error(`napi build terminated by signal: ${result.signal}`);
  process.exit(1);
}
process.exit(result.status ?? 1);
