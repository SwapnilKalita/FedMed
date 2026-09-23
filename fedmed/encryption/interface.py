"""Encryption interface and TenSEAL stub.

Defines a small, testable EncryptionInterface and a TenSEAL stub that
implements the same interface. The stub performs identity operations so
it can be used in local tests and demonstrations without native builds.
"""
from typing import Any


class EncryptionInterface:
    """Abstract encryption interface used by FedMed components.

    Implementations must provide:
      - encrypt(ndarray) -> encrypted object
      - decrypt(encrypted object) -> ndarray
      - add(encrypted_a, encrypted_b) -> encrypted sum
    """

    def encrypt(self, payload: Any) -> Any:
        raise NotImplementedError

    def decrypt(self, encrypted: Any) -> Any:
        raise NotImplementedError

    def add(self, a: Any, b: Any) -> Any:
        raise NotImplementedError


class TenSEALStub(EncryptionInterface):
    """A lightweight stub implementation that simulates TenSEAL behavior.

    The stub does NOT provide real encryption—it simply stores the raw
    payload and implements add/decrypt as identity-like operations so the
    higher-level aggregation and wiring can be exercised without native builds.
    """

    def __init__(self):
        # In a real implementation this would create TenSEAL contexts
        self._meta = {"stub": True}

    def encrypt(self, payload: Any) -> Any:
        # Simulated ciphertext wrapper
        return {"_tenseal_stub_cipher": True, "payload": payload}

    def decrypt(self, encrypted: Any) -> Any:
        if isinstance(encrypted, dict) and encrypted.get("_tenseal_stub_cipher"):
            return encrypted.get("payload")
        raise ValueError("Object is not a TenSEAL stub ciphertext")

    def add(self, a: Any, b: Any) -> Any:
        # Perform elementwise addition for numpy arrays and scalars
        pa = self.decrypt(a) if isinstance(a, dict) and a.get("_tenseal_stub_cipher") else a
        pb = self.decrypt(b) if isinstance(b, dict) and b.get("_tenseal_stub_cipher") else b
        try:
            import numpy as _np
            return self.encrypt(_np.add(pa, pb))
        except Exception:
            # Fallback to Python addition
            return self.encrypt(pa + pb)
