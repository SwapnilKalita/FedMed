from flask import Flask, jsonify
from flask_cors import CORS

# Re-use repository stubs
from fedmed.model.unet3d import UNet3D
from fedmed.fl.client import LocalTrainer
from fedmed.encryption.mock import MockSecureAggregator

app = Flask(__name__)
CORS(app)

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

    # Prepare JSON-serializable output
    return jsonify({
        'model_summary': model_stub.summary(),
        'local_updates': [u.tolist() for u in updates],
        'aggregated': averaged.tolist(),
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
