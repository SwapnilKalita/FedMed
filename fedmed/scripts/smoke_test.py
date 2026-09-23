"""Lightweight smoke test used by CI.

This script imports the repo stubs and runs a few basic operations to verify
the package can be imported and the demo wiring works. It intentionally uses
only lightweight dependencies (numpy).
"""
from fedmed.model.unet3d import UNet3D
from fedmed.encryption.mock import MockSecureAggregator
from fedmed.fl.client import LocalTrainer


def main():
    print('Running smoke test...')
    net = UNet3D()
    print(net.summary())
    trainers = [LocalTrainer(net) for _ in range(2)]
    updates = [t.get_weights() for t in trainers]
    print('Local updates:', updates)
    agg = MockSecureAggregator(rng_seed=0)
    masked = []
    masks = []
    for u in updates:
        m_u, m = agg.mask_update(u)
        masked.append(m_u)
        masks.append(m)
    averaged = agg.aggregate(masked, masks)
    print('Averaged:', averaged)


if __name__ == '__main__':
    main()
