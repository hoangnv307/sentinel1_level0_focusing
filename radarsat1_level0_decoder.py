"""Giải mã RADARSAT-1 Level-0 CEOS và tùy chọn focus bằng CSA."""

import argparse
from pathlib import Path

import numpy as np

from radarsat1_processing import decode, focus, read_metadata, to_uint8


def _default_raw() -> Path:
    files = list(Path("data/radarsat-1").glob("**/*.raw"))
    if len(files) != 1:
        raise SystemExit("Hãy truyền đường dẫn file .raw (không tìm thấy đúng một file mặc định)")
    return files[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", nargs="?", type=Path)
    parser.add_argument("--first-line", type=int, default=0)
    parser.add_argument("--lines", type=int, default=1536)
    parser.add_argument("--first-sample", type=int, default=0)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--agc", action="store_true", help="bù suy hao AGC như mã MATLAB")
    parser.add_argument("--focus", action="store_true", help="chạy thêm Chirp Scaling Algorithm")
    parser.add_argument(
        "--output", type=Path, default=Path("data/radarsat-1/generated-output/radarsat1_decoded.npy")
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("data/radarsat-1/generated-output/radarsat1_csa.png"),
    )
    args = parser.parse_args()

    raw = args.raw or _default_raw()
    metadata = read_metadata(raw)
    data = decode(
        raw,
        first_line=args.first_line,
        line_count=args.lines,
        first_sample=args.first_sample,
        sample_count=args.samples,
        apply_agc=args.agc,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, data)
    print(f"Đã giải mã {data.shape[0]} x {data.shape[1]} mẫu ({metadata.beam}) -> {args.output}")

    if args.focus:
        import matplotlib.pyplot as plt

        image = to_uint8(focus(data, metadata, first_sample=args.first_sample))
        args.image.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(args.image, image, cmap="gray", vmin=0, vmax=255)
        print(f"Đã focus CSA -> {args.image}")


if __name__ == "__main__":
    main()
