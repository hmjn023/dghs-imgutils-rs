"""
Python reference script for metadata + sd benchmark comparison.
"""
import sys
import json
from imgutils.metadata import read_geninfo_parameters, read_lsb_metadata, write_lsb_metadata
from imgutils.sd import parse_sdmeta_from_text
from imgutils.sd.model import read_metadata as read_safetensors_meta


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"success": False, "error": "Usage: python run_python_metadata_sd.py <command> <path>"}))
        sys.exit(1)

    command = sys.argv[1]
    path = sys.argv[2]

    try:
        if command == "geninfo":
            result = read_geninfo_parameters(path)
            print(json.dumps({"success": True, "geninfo": result}))

        elif command == "lsb_read":
            try:
                result = read_lsb_metadata(path)
                print(json.dumps({"success": True, "metadata": result}))
            except Exception:
                print(json.dumps({"success": True, "metadata": None}))

        elif command == "lsb_write":
            metadata_str = sys.argv[3] if len(sys.argv) > 3 else "{}"
            dst_path = sys.argv[4] if len(sys.argv) > 4 else "output_lsb.png"
            metadata = json.loads(metadata_str)
            _ = write_lsb_metadata(path, metadata)
            print(json.dumps({"success": True, "dst_path": dst_path}))

        elif command == "sd_parse":
            geninfo = read_geninfo_parameters(path)
            if geninfo:
                meta = parse_sdmeta_from_text(geninfo)
                result = {
                    "success": True,
                    "prompt": meta.prompt,
                    "neg_prompt": meta.neg_prompt,
                    "parameters": {k: str(v) for k, v in meta.parameters.items()}
                }
            else:
                result = {"success": True, "prompt": "", "neg_prompt": "", "parameters": {}}
            print(json.dumps(result))

        elif command == "safetensors":
            meta = read_safetensors_meta(path)
            print(json.dumps({"success": True, "metadata": meta}))

        else:
            print(json.dumps({"success": False, "error": f"Unknown command: {command}"}))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))


if __name__ == '__main__':
    main()
