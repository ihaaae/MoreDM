"""Compatibility metric only; the invalid SD3 ablation driver is archived.

See ARCHIVE.md. Keep noise_proxy for existing result readers, not generation.
The original driver is in Archive/2026-10-invalid-evidence/lib/sd3_ablation.py.
"""


def noise_proxy(path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        small = np.asarray(image.convert("RGB").resize((224, 224), Image.BICUBIC)).astype(np.float32) / 255
    gray = small.mean(2)
    lap = (np.roll(gray, 1, 0) + np.roll(gray, -1, 0) + np.roll(gray, 1, 1)
           + np.roll(gray, -1, 1) - 4 * gray)[1:-1, 1:-1]
    return float(lap.var() / gray.std() ** 2)


if __name__ == "__main__":
    raise SystemExit("Archived SD3 ablation entry point; see ARCHIVE.md. No experiment was run.")
