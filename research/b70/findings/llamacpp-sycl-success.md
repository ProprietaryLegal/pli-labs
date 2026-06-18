# GGUF SYCL Serving Success

## Result

The B70-class research achieved the base goal: a large MoE GGUF served through
a SYCL layer-split lane and produced usable long-context legal completions.

The public result is a capacity and reliability finding. It shows that
B70-class owned hardware can support a serious local legal AI lane when the
runtime, model format, and workload are validated together.

## What Is Public

- The successful lane used a GGUF-compatible serving stack.
- The workload included legal drafting and review prompts.
- The result passed readiness and completion checks.
- The lane is useful for private local evaluation.

## What Is Private

Service ports, model cache paths, internal profile names, failed settings, and
failure runbooks remain internal. Curated successful launch settings and
measured throughput are published separately in
[successful-launch-profile.md](successful-launch-profile.md).

## Limitation

Layer-split serving is a reliability and capacity path. It is not the same
claim as a tensor-parallel speed record. XPU tensor-parallel work remains a
separate research lane that requires its own stability gates.

## Public Release Conclusion

B70-class local hardware can be a practical legal AI lane when evaluated on
real legal workloads and reported with a clear privacy boundary.
