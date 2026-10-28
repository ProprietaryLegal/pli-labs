# DeepSeek-V4.1-Flash on 2× DGX Spark with x86 RTX 3090 offload

<!-- pli:update:2026-10-28:start -->

**Update 2026-10-28:** the DSV41 code was republished with test results for every published tree, a production build record and a full model card for the served weights: [2026-10-28-tests-build-model-card.md](2026-10-28-tests-build-model-card.md).

<!-- pli:update:2026-10-28:end -->

<!-- pli:update:2026-10-27:start -->

**Update 2026-10-27:** the full source is public at [ProprietaryLegal/dsv41-offload](https://github.com/ProprietaryLegal/dsv41-offload), with focused repos for the [measurement harness](https://github.com/ProprietaryLegal/dsv41-window-harness) and [sampled drafting](https://github.com/ProprietaryLegal/dsv41-sampled-drafting). Status and lessons: [2026-10-27-status-and-code.md](2026-10-27-status-and-code.md).

<!-- pli:update:2026-10-27:end -->

This page indexes the independently usable PLI Labs offload components for two DGX Spark / GB10 nodes and an x86 host. The standalone repositories are now canonical; the folders here are a snapshot for browsing earlier integrated material.

<!-- pli:blurb:start -->

> Published by [PLI Labs](https://proprietarylegal.ai/labs.html), the open research arm of [Proprietary Legal Intelligence](https://proprietarylegal.ai). PLI builds AI systems around a legal organization's own work product, combining retrieval, staged workflows, validation, and attorney review. Visit [proprietarylegal.ai](https://proprietarylegal.ai) to explore our work and discuss what your organization needs.

<!-- pli:blurb:end -->

Status as of 2026-10-23: remote Engram, one-card remote drafting and the async process handoff are deployed. The NVLink worker is opt-in, measured and reviewed, with promotion pending. Native fidelity still fails its strict gate. Additional cache and scheduler features remain separately gated.

| component | standalone GitHub | Hugging Face | hub snapshot folder | status |
|---|---|---|---|---|
| Async host-flag handoff | [source](https://github.com/ProprietaryLegal/dsv41-async-hostflag-handoff) | [entry](https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff) | [dsv41-async-hostflag-handoff/](dsv41-async-hostflag-handoff/) | Production; one-card async |
| Remote RTX 3090 drafter | [source](https://github.com/ProprietaryLegal/dsv41-remote-dspark-drafter-3090) | [entry](https://huggingface.co/ProprietaryLegal/dsv41-remote-dspark-drafter-3090) | [dsv41-remote-dspark-drafter-3090/](dsv41-remote-dspark-drafter-3090/) | Production; capacity mode |
| NVLink two-card drafter | [source](https://github.com/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink) | [entry](https://huggingface.co/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink) | [dsv41-dspark-drafter-tp2-nvlink/](dsv41-dspark-drafter-tp2-nvlink/) | Opt-in; promotion pending |
| Remote Engram | [source](https://github.com/ProprietaryLegal/dsv41-remote-engram-rdma) | [entry](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma) | [dsv41-remote-engram-rdma/](dsv41-remote-engram-rdma/) | Production; exact fallback |
| ofl RDMA/TCP transport | [source](https://github.com/ProprietaryLegal/ofl-rdma-transport) | [entry](https://huggingface.co/ProprietaryLegal/ofl-rdma-transport) | [ofl-rdma-transport/](ofl-rdma-transport/) | Production |
| Native drafter on SM86 | [source](https://github.com/ProprietaryLegal/dsv41-native-1017-drafter-sm86) | [entry](https://huggingface.co/ProprietaryLegal/dsv41-native-1017-drafter-sm86) | [dsv41-native-1017-drafter-sm86/](dsv41-native-1017-drafter-sm86/) | Merged opt-in only; fidelity gate fails; not serving |
| Long-context benchmark tools | [source](https://github.com/ProprietaryLegal/dsv41-longctx-bench-tools) | [entry](https://huggingface.co/ProprietaryLegal/dsv41-longctx-bench-tools) | [dsv41-longctx-bench-tools/](dsv41-longctx-bench-tools/) | Measured tooling |
| Exact acceptance ceiling | [source](https://github.com/ProprietaryLegal/dsv41-exact-acceptance-ceiling) | [entry](https://huggingface.co/ProprietaryLegal/dsv41-exact-acceptance-ceiling) | No hub snapshot; use standalone source | Research-only method and aggregates |
| Full stack | [source](https://github.com/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) | [entry](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) | [overview](README.md) | Production overview; standalone code index |
| Offload findings | [source](https://github.com/ProprietaryLegal/dsv41-spark-offload-findings) | [entry](https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings) | [dsv41-spark-offload-findings/](dsv41-spark-offload-findings/) | Findings and qualification limits |

Start with the [full-stack GitHub overview](https://github.com/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) or its [HF entry](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) for architecture, historical baseline comparisons, current configuration, reproduction and rollback. Each standalone component supplies its own code map, limitations, attribution and test instructions. The acceptance-ceiling repository contains the method and aggregate results; private captures and weights are excluded.

Work in progress: KV page packing validated on GB10 at 327,680 (production promotion pending); 524,288 serves correctly but is held for a prefill regression under investigation; 1M untested on GB10; prefix storage is cleared for gated tests but its host-RAM tier is not integrated; the prefill admission fix is CPU-tested and awaits independent review and GB10 validation; native arithmetic corrections have not cleared fidelity. Drafter self-distillation and further upstream adoption have no qualified serving gain. These are status notes, not performance results.

Upstream forks: [Spark image and offload code](https://github.com/ProprietaryLegal/deepseek-v41-1017-dgx-spark) · [two-node launcher](https://github.com/ProprietaryLegal/dgx-spark-deepseek-v41).

[Hugging Face collection](https://huggingface.co/collections/ProprietaryLegal/deepseek-v41-flash-on-2x-dgx-spark-x86-offload-6adab604a517b426ded3916e) · [PLI Labs research](https://proprietarylegal.ai/research/index.html).

<!-- pli:footer:start -->

---

## About PLI Labs

“Your proprietary work product is your proprietary intelligence.” PLI Labs is the open research arm of Proprietary Legal Intelligence. We publish legal model prunes, serving recipes for owned hardware, and open-source tools so readers can inspect our methods and test our claims before hiring us.

Founded by a practicing trial lawyer, PLI builds AI systems around a legal organization's own work product: retrieval, staged workflows, validation, and attorney review. The approach grew from systems running on live matters in a litigation practice. Our consulting work supports deployment through approved frontier APIs, private cloud, or open-weight models on hardware you own. Explore the research or contact us about your organization's work.

[Website](https://proprietarylegal.ai) · [PLI Labs](https://proprietarylegal.ai/labs.html) · [Research](https://proprietarylegal.ai/research/index.html) · [GitHub](https://github.com/ProprietaryLegal) · [Hugging Face](https://huggingface.co/ProprietaryLegal) · [Contact](https://proprietarylegal.ai/contact.html)

Proprietary Legal Intelligence LLC is a technology consultancy, not a law firm, and does not provide legal services or legal advice.

<!-- pli:footer:end -->
