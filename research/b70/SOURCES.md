# Sources

## Local Evidence Classes

The public B70 package is summarized from sanitized local evidence generated
during B70-class bring-up and benchmark work:

- official-spec summaries;
- offline stack staging notes for kernel, driver, telemetry, container, and
  model-format readiness;
- GGUF-compatible SYCL serve findings;
- Intel vLLM / llm-scaler XPU canary findings;
- OpenVINO GenAI HETERO findings;
- legal-context benchmark artifacts;
- restore and readiness notes.

The raw local materials are not published because they include operational
details that do not belong in a public repository.

## Public Technical References

- Intel Arc Pro B70 product and datasheet materials.
- Intel llm-scaler repository and vLLM-XPU documentation:
  https://github.com/intel/llm-scaler
- Intel xpu-smi tooling:
  https://github.com/intel/xpumanager
- llama.cpp Intel server images and SYCL backend:
  https://github.com/ggml-org/llama.cpp
- vLLM XPU documentation:
  https://docs.vllm.ai/
- OpenVINO GenAI:
  https://github.com/openvinotoolkit/openvino.genai
- Intel B70 community setup research:
  https://github.com/Hal9000AIML/arc-pro-b70-inference-setup-ubuntu-server
- Independent Arc Pro B70 benchmark research:
  https://github.com/PMZFX/intel-arc-pro-b70-benchmarks

## Public-Release Rule

Public source links are safe to publish. Raw local paths, hostnames, private
service URLs, model-cache paths, logs, environment files, credentials, and
client-related materials are not.
