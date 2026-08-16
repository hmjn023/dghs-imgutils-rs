import { spawnSync } from "node:child_process";

const args = ["build", "--platform", "--dts", "index.d.ts"];

if (process.argv.includes("--release")) {
  args.push("--release");
}

// CUDA is dynamically loaded and does not require the CUDA SDK at link time.
// Include its binding in Linux N-API builds so the runtime can select it when
// a compatible NVIDIA installation is present. Other platforms retain the
// existing CPU-safe build unless they opt into a provider feature explicitly.
if (process.platform === "linux") {
  args.push("--features", "cuda");
}

args.push("--cargo-flags=--lib");

const result = spawnSync("napi", args, { stdio: "inherit" });
if (result.error) {
  console.error(`failed to start napi build: ${result.error}`);
  process.exit(1);
}
process.exit(result.status ?? 1);
