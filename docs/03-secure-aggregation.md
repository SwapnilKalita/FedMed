# Secure Aggregation

## What the PDF specifies vs. what this sandbox can build

The original plan calls for TenSEAL (CKKS homomorphic encryption) so the
coordinator can sum encrypted weights without ever decrypting an individual
hospital's contribution. TenSEAL needs a compiled C++ extension that this
environment cannot install (no network access to fetch it).

`server/secure_aggregation.py` implements the same *security guarantee* --
the coordinator never learns an individual hospital's update, only the sum --
using pairwise additive masking, the core idea behind Google's production
Secure Aggregation protocol (Bonawitz et al., 2017), which is what Flower's
`SecAgg`/`SecAgg+` mods layer on top of FedAvg in real deployments.

## How the masking works

Every pair of hospitals agrees on one shared random value. One hospital adds
it to its contribution, the other subtracts it. Summed across every hospital,
all the pairwise masks cancel out exactly, leaving only the true total. Any
single masked value on its own is indistinguishable from noise --
`tests/test_secure_aggregation.py::test_individual_masked_values_hide_the_true_contribution`
checks this directly.

`tests/test_secure_aggregation.py::test_matches_plaintext_fedavg_result` proves
the masked-and-summed result is numerically identical to plain FedAvg, so
swapping this module for real TenSEAL-based homomorphic encryption later
would not require changing `server/coordinator.py` at all -- only the
aggregation function's internals.

## Limitation to be upfront about

This simplified version assumes every hospital that starts a round finishes
it (no dropout *during* mask generation, only *between* rounds, which
`offline_rounds` in the coordinator already handles). Production secure
aggregation protocols include a recovery step for clients that disappear
mid-round; that is out of scope here.
