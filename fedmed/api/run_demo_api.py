from flask import Flask, jsonify
from flask_cors import CORS

# Re-use repository stubs
from fedmed.model.unet3d import UNet3D
from fedmed.fl.client import LocalTrainer
from fedmed.encryption.mock import MockSecureAggregator

# Image generation
import numpy as np
from pathlib import Path
from PIL import Image

app = Flask(__name__)
CORS(app)


def _ensure_assets_and_write_mask(mask_array: np.ndarray, out_name: str = 'sample_mask.png') -> str:
    """Save mask_array (2D, values 0..1 or 0..255) into fedmed/docs/assets and return a web path."""
    repo_root = Path(__file__).resolve().parents[2]
    assets_dir = repo_root / 'fedmed' / 'docs' / 'assets'
    assets_dir.mkdir(parents=True, exist_ok=True)
    out_path = assets_dir / out_name

    # Normalize to 0..255 uint8
    arr = mask_array
    if arr.dtype != np.uint8:
        arr = (255 * (arr.astype(np.float32) - arr.min()) / max((arr.max() - arr.min()), 1e-6)).astype(np.uint8)

    img = Image.fromarray(arr)
    if img.mode != 'L':
        img = img.convert('L')
    img.save(out_path)

    # Return the path relative to the docs root (served by static server)
    return f"/assets/{out_name}"


@app.route('/run_demo')
def run_demo():
    model_stub = UNet3D()
    trainers = [LocalTrainer(model_stub) for _ in range(3)]
    updates = [t.get_weights() + (i * 0.1) for i, t in enumerate(trainers)]

    agg = MockSecureAggregator(rng_seed=1)
    masked = []
    masks = []
    for u in updates:
        m_u, m = agg.mask_update(u)
        masked.append(m_u)
        masks.append(m)
    averaged = agg.aggregate(masked, masks)

    # Generate a synthetic mask image (simple blob) and save under docs/assets
    try:
        size = (256, 256)
        # create a radial gaussian blob as a mask
        x = np.linspace(-1, 1, size[1])
        y = np.linspace(-1, 1, size[0])
        xv, yv = np.meshgrid(x, y)
        blob = np.exp(-((xv**2 + yv**2) * 6.0))
        # add a secondary offset blob
        blob += 0.6 * np.exp(-(((xv - 0.4)**2 + (yv + 0.3)**2) * 20.0))
        blob = blob / blob.max()
        mask_url = _ensure_assets_and_write_mask((blob * 255).astype(np.uint8))
    except Exception:
        mask_url = None

    # Prepare JSON-serializable output
    response = {
        'model_summary': model_stub.summary(),
        'local_updates': [u.tolist() for u in updates],
        'aggregated': averaged.tolist(),
    }
    if mask_url:
        response['mask_url'] = mask_url

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
