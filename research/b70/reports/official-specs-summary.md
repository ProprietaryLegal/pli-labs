# Official Specs Summary

## Public Hardware Facts

The Intel Arc Pro B70 public materials describe a professional GPU intended for
AI and content-creation workloads.

Relevant public facts for inference planning:

- 32 GB dedicated VRAM per card.
- 32 Xe2 cores.
- 256 XMX AI engines.
- PCIe Gen 5 x16 native support.
- Intel branded-card power point of 230 W, within a broader listed range.

## Why These Facts Matter

The 32 GB VRAM figure is the main reason the card is interesting for local AI.
Four cards provide a 128 GB capacity island, enough to make 100B-plus
quantized MoE serving practical when the backend can distribute the model.

The XMX and Level Zero / XPU path make B70 a different engineering problem from
CUDA serving. That difference is the point of the research: the hardware has
real capacity, but software-stack maturity decides whether the system is merely
interesting or production-useful.

## Public Planning Rule

Spec sheets are not benchmarks. They help plan the candidate lane, but the
public claim must come from measured model serving on the exact stack.
