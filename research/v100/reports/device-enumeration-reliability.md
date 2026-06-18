# Device Enumeration Reliability

## Public Finding

Some V100-class systems can show intermittent device-enumeration problems after
boot. The public lesson is simple: local AI reliability depends on hardware
discovery and readiness checks, not only on whether a model file can load.

PLI Labs treats detailed chassis symptoms, firmware settings, recovery steps,
and board-level notes as private operational material. Those details are useful
for internal maintenance but are not necessary for public research readers.

## Public Guidance

For any owned-hardware legal AI deployment:

- verify accelerator discovery after boot;
- confirm runtime visibility before starting model services;
- keep recovery procedures in an internal runbook;
- track degraded-capacity states separately from model failures;
- report public lessons by hardware class, not by machine-specific symptoms.

## Boundary

This public note intentionally omits exact topology, affected slot behavior,
firmware settings, recovery commands, internal hostnames, and local paths.

## Links

- https://proprietarylegal.com
- https://proprietarylegal.ai
