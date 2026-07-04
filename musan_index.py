import json
from pathlib import Path
import soundfile as sf

def build_musan_index(musan_root: str, out_json: str = "musan_index.json"):
    musan_root = Path(musan_root)
    index = {"noise": [], "music": [], "speech": []}

    for category in index.keys():
        category_dir = musan_root / category
        if not category_dir.exists():
            continue
        for wav_path in category_dir.rglob("*.wav"):
            info = sf.info(str(wav_path))
            index[category].append({
                "path": str(wav_path),
                "frames": info.frames,
                "samplerate": info.samplerate,
            })

    with open(out_json, "w") as f:
        json.dump(index, f)
    return index

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--musan_root", type=str, required=True, help="Path to MUSAN root dir")
    parser.add_argument("--out_json", type=str, default="musan_index.json", help="Output JSON path")
    args = parser.parse_args()
    build_musan_index(args.musan_root, args.out_json)
