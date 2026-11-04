import argparse
import shutil
from pathlib import Path


def extract_bonafide_filenames(protocol_file: Path) -> set[str]:
    bonafide_ids = set()
    with open(protocol_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[4] == "bonafide":
                bonafide_ids.add(f"{parts[1]}.flac")
    return bonafide_ids


def process_subset(src_dir: Path, dst_dir: Path, protocol_file: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    bonafide_set = extract_bonafide_filenames(protocol_file)
    count = 0

    for file_path in src_dir.glob("*.flac"):
        if file_path.name in bonafide_set:
            # Move the file to avoid duplicating disk usage
            shutil.move(str(file_path), str(dst_dir / file_path.name))
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter bonafide files from ASVspoof 2019 LA and clean raw dir.")
    parser.add_argument("--source-dir", type=str, default="data/raw/LA")
    parser.add_argument("--target-dir", type=str, default="data/raw/asvspoof-bonafide")
    args = parser.parse_args()

    src_root = Path(args.source_dir)
    dst_root = Path(args.target_dir)

    mappings = [
        ("train", "ASVspoof2019_LA_train/flac", "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"),
        ("val", "ASVspoof2019_LA_dev/flac", "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"),
        ("test", "ASVspoof2019_LA_eval/flac", "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"),
    ]

    print("Starting extraction of bonafide recordings...")
    for target_subset, rel_flac_dir, rel_proto in mappings:
        flac_path = src_root / rel_flac_dir
        proto_path = src_root / rel_proto
        target_path = dst_root / target_subset

        if not flac_path.exists() or not proto_path.exists():
            raise FileNotFoundError(f"Missing required directory or protocol: {flac_path} / {proto_path}")

        extracted_count = process_subset(flac_path, target_path, proto_path)
        print(f"[{target_subset}] Extracted {extracted_count} bonafide files into {target_path}")


if __name__ == "__main__":
    main()
