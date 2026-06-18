# Official Specs Summary

## Public Hardware Facts

The Intel Arc Pro B70 public materials describe a professional GPU intended for
AI and content-creation workloads.

Relevant public facts for inference planning:

- high-memory professional GPU class;
- Intel XMX AI engines;
- Level Zero / XPU software path;
- PCIe accelerator form factor.

## Why These Facts Matter

The B70 class is interesting for local AI because it expands owned-hardware
capacity outside the CUDA path. That difference is the point of the research:
the hardware has real local-serving potential, but software-stack maturity
decides whether the system is merely interesting or production-useful.

## Public Planning Rule

Spec sheets are not benchmarks. They help plan the candidate lane, but the
public claim must come from measured model serving on a validated stack.
