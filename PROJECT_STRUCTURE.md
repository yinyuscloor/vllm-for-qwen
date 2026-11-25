# vllm-for-qwen 项目代码结构

> 本仓库是 [vLLM](https://github.com/vllm-project/vllm) 的 fork：一个高吞吐、低显存占用的大模型推理与服务引擎。
> 在上游 vLLM 基础上，本 fork 增加了针对 Kimi / Qwen / DeepSeek / MiniMax 等模型的定制实现与融合 CUDA kernel（如 `kimi_k3`、`muse_spark`、`qwen4_exp`、`inkling2` 等）。
>
> 技术栈：**Python（主体）+ C++/CUDA（kernel，经 CMake 构建）+ Rust（实验性前端）**。
> 命令行入口：`vllm`（见 `pyproject.toml` 中 `vllm.entrypoints.cli.main:main`）。

---

## 1. 顶层目录总览

| 路径 | 职责 |
| --- | --- |
| `vllm/` | 核心 Python 包（引擎、模型、服务入口、kernel 绑定），唯一被打包的目录 |
| `csrc/` | C++/CUDA 原生算子源码，编译为 `_C` 扩展供 Python 调用 |
| `rust/` | 实验性 Rust 原生前端（HTTP/gRPC 服务、chat 渲染、解析器） |
| `tests/` | pytest 测试套件 |
| `benchmarks/` | 独立 benchmark 脚本（latency / throughput / serving / kernel） |
| `docs/` | mkdocs 文档源（`mkdocs.yaml` 为配置） |
| `examples/` | 各场景示例脚本（离线推理、服务部署、多模态等） |
| `docker/` | 各平台 Dockerfile（CUDA / CPU / ROCm / ppc64le / EFA 等） |
| `tools/` | 开发与构建辅助脚本（kernel autotune、DeepGEMM、CI 工具等） |
| `cmake/` | CMake 辅助模块（CPU/ROCm 扩展、外部项目、补丁） |
| `requirements/` | 按平台拆分的依赖声明（`cuda.txt`、`cpu.txt`、`test/` 等） |
| `recipes/` | 模型部署 recipe（模型 + 硬件 的推荐配置） |
| `.buildkite/` | Buildkite CI 配置与脚本 |
| `.github/` | GitHub 工作流与模板 |
| `pyproject.toml` | 包元数据、构建依赖、ruff/mypy/pytest 等工具配置 |
| `CMakeLists.txt` | 原生扩展构建入口 |
| `setup.py` | setuptools 安装逻辑（驱动 CMake 构建） |
| `AGENTS.md` | AI 辅助贡献指南（开发工作流、测试、风格约定） |
| `rust-toolchain.toml` | Rust 工具链版本固定 |

---

## 2. 核心 Python 包 `vllm/`

### 2.1 引擎与调度

| 路径 | 职责 |
| --- | --- |
| `vllm/engine/` | 引擎对外接口层：`llm_engine.py`（同步）、`async_llm_engine.py`（异步）、`arg_utils.py`（`EngineArgs` 与 CLI 参数解析） |
| `vllm/v1/` | V1 架构主体（当前默认执行路径），见 2.2 |
| `vllm/entrypoints/llm.py` | 离线推理的 `LLM` 类（用户侧高层 API） |
| `vllm/sequence.py` | 序列（Sequence）数据结构 |
| `vllm/outputs.py` / `vllm/pooling_params.py` / `vllm/sampling_params.py` | 请求/输出参数与结果数据结构 |
| `vllm/logits_process.py` / `vllm/logprobs.py` | logits 处理与 logprobs 计算 |

### 2.2 V1 架构 `vllm/v1/`

| 路径 | 职责 |
| --- | --- |
| `v1/engine/` | V1 引擎核心：`core.py`（EngineCore 主循环）、`async_llm.py`、`llm_engine.py`、`detokenizer.py`、`output_processor.py`、`coordinator.py`（D/P 协调）、`parallel_sampling.py` |
| `v1/core/` | 调度与 KV cache 管理：`sched/`（调度器）、`block_pool.py`、`kv_cache_manager.py`、`kv_cache_coordinator.py`、prefix caching 相关 |
| `v1/executor/` | 执行器抽象：`uniproc_executor.py`、`multiproc_executor.py`、`ray_executor_v2.py` |
| `v1/worker/` | Worker 与 model runner：`gpu/`、`cpu/`、`gpu_worker.py`、`lora_model_runner_mixin.py` |
| `v1/attention/` | Attention 后端注册与选择（`backends/`：FlashAttention、FlashInfer、MLA 等；`selector.py`） |
| `v1/sample/` | 采样器实现（`ops/`：top-p/top-k/temperature 等算子） |
| `v1/spec_decode/` | 投机解码（n-gram、EAGLE、MTP 等，`dynamic/`） |
| `v1/structured_output/` | 结构化输出后端（xgrammar、outlines、guidance） |
| `v1/kv_offload/`、`v1/kv_offload/../simple_kv_offload`（`v1/simple_kv_offload/`） | KV cache 分层卸载（CPU/文件系统/tiering） |
| `v1/kv_hints/`、`v1/kv_cache_interface.py`、`v1/kv_cache_layout.py` | KV cache 抽象与布局 |
| `v1/metrics/` | 指标采集（Prometheus 等） |
| `v1/fault_tolerance/` | 容错 |
| `v1/pool/` | pooling 类任务（embedding 等） |
| `v1/hisparse/`、`v1/watermarking/` | 稀疏注意力与水印 |

### 2.3 服务入口 `vllm/entrypoints/`

| 路径 | 职责 |
| --- | --- |
| `entrypoints/cli/` | `vllm` 命令行：`serve.py`、`run_batch.py`、`benchmark/` 等 |
| `entrypoints/openai/` | OpenAI 兼容 API：`api_server.py`、`chat_completion/`、`completion/`、`responses/` |
| `entrypoints/anthropic/`、`entrypoints/cohere/` | Anthropic / Cohere 兼容 API |
| `entrypoints/serve/` | 服务中间件与周边：`middleware/`、`lora/`、`profile/`、`elastic_ep/`、`fault_tolerance/`、`instrumentator/` |
| `entrypoints/launcher(s)` | 服务器启动器 |
| `entrypoints/speech_to_text/`、`entrypoints/pooling/`、`entrypoints/generate/`、`entrypoints/rl/`、`entrypoints/mcp/`、`entrypoints/scale_out/` | 语音转写、pooling、生成、RL、MCP、横向扩展等入口 |
| `entrypoints/grpc_server.py` | gRPC 服务 |

### 2.4 模型执行 `vllm/model_executor/` 与 `vllm/models/`

| 路径 | 职责 |
| --- | --- |
| `model_executor/layers/` | 通用算子层：`linear.py`、`layernorm.py`、`activation.py`、`fused_moe/`、`attention/`、`mamba/`、`rotary_embedding` 等 |
| `model_executor/model_loader/` | 权重加载：`default_loader.py`、`tensorizer_loader.py`、`sharded_state_loader.py`、`runai_streamer_loader.py` 等 |
| `model_executor/models/` | （上游约定的模型实现挂载点） |
| `model_executor/custom_op.py` | CustomOp 基类（按平台分发 CUDA/ROCm/CPU 实现） |
| `model_executor/determinism/`、`model_executor/warmup/`、`model_executor/offloader/` | 确定性执行、预热、权重卸载 |
| `vllm/models/` | 模型实现注册目录。上游模型之外含本 fork 定制模型：`kimi_k3/`、`muse_spark/`、`qwen4_exp/`、`inkling2/`、`deepseek_v4/`、`deepseek_v41/`、`minimax_m3/`、`glm5next/`、`hy_v4/`、`step5/` 等 |

### 2.5 编译与 kernel 接入

| 路径 | 职责 |
| --- | --- |
| `vllm/compilation/` | torch.compile 集成：`backends.py`、`cuda_graph.py`、`piecewise_backend.py`、`passes/`（融合 pass）、`caching.py` |
| `vllm/kernels/` | Python 侧 kernel 入口：`vllm_c.py`（`_C` 扩展绑定）、`triton/`、`helion/` |
| `vllm/vllm_flash_attn/` | 内置 flash-attention |
| `vllm/triton_utils/`、`vllm/cute_utils/`、`vllm/tilelang_utils/` | Triton / CUTLASS CuTe / TileLang 的工具封装 |
| `vllm/_custom_ops.py`、`_cpu_ops.py`、`_xpu_ops.py`、`_aiter_ops.py` | 原生算子的 Python 包装（按平台） |

### 2.6 分布式与平台

| 路径 | 职责 |
| --- | --- |
| `vllm/distributed/` | 并行通信：`parallel_state.py`、`device_communicators/`、`kv_transfer/`（KV 传输/P2P）、`ec_transfer/`、`eplb/`（专家负载均衡）、`elastic_ep/`、`weight_transfer/` |
| `vllm/platforms/` | 硬件平台抽象：`cuda.py`、`rocm.py`、`cpu.py`、`tpu.py`、`xpu.py`、`zen_cpu.py` |
| `vllm/ray/` | Ray 集成 |
| `vllm/device_allocator/` | 显存分配（cuMem、sleep mode） |

### 2.7 推理特性

| 路径 | 职责 |
| --- | --- |
| `vllm/lora/` | LoRA 支持：`model_manager.py`、`punica_wrapper/`、`resolver.py` |
| `vllm/multimodal/` | 多模态：`image.py`、`audio.py`、`video.py`、`processing/`、`cache/`、`video_decoders/` |
| `vllm/reasoning/` | reasoning（思维链）解析器，按模型各一个文件（kimi_k3、deepseek_v4、glm47 等） |
| `vllm/tool_parsers/` | 工具调用（function calling）解析器 |
| `vllm/parser/` | 引擎侧输出解析框架（`chat_parsing/`、`engine/`） |
| `vllm/renderers/` | chat 模板渲染（按模型族：`hf.py`、`deepseek_v4.py`、`hy_v4.py` 等） |
| `vllm/tokenizers/` | 分词器封装（HF、fastokens 及定制模型分词器） |
| `vllm/inputs/` | 输入预处理 |

### 2.8 配置与基础设施

| 路径 | 职责 |
| --- | --- |
| `vllm/config/` | 配置 dataclass：`model.py`、`cache.py`、`parallel.py`、`compilation.py`、`lora.py`、`kv_transfer.py` 等 |
| `vllm/envs.py` | 环境变量集中定义 |
| `vllm/logger.py`、`vllm/logging_utils/` | 日志 |
| `vllm/utils/` | 通用工具集 |
| `vllm/transformers_utils/` | transformers 兼容层（config、tokenizer 加载） |
| `vllm/plugins/` | 插件机制（`io_processors/`、LoRA resolver 等） |
| `vllm/profiler/`、`vllm/tracing/`、`vllm/usage/` | 性能分析、链路追踪、用量统计 |
| `vllm/benchmarks/` | 内置 `vllm bench` 命令实现（`latency.py`、`throughput.py`、`serve.py`、`sweep/`） |
| `vllm/snapshot/` | 引擎快照（checkpoint/restore） |
| `vllm/ir/` | 算子 IR（融合与容差定义） |
| `vllm/assets/`、`vllm/third_party/` | 内置资源、第三方代码 |
| `vllm/collect_env.py`、`vllm/connections.py`、`vllm/forward_context.py`、`vllm/tasks.py`、`vllm/scalar_type.py` | 环境采集、连接管理、forward 上下文、任务类型、标量类型 |

---

## 3. 原生 kernel 层 `csrc/`

编译产物为 `_C` 扩展（`torch_bindings.cpp` 注册算子，`ops.h` 声明接口）。

| 路径 | 职责 |
| --- | --- |
| `csrc/attention/` | 注意力 kernel：`mla/`、`dsa/`、`dcp_utils/`、`merge_attn_states.cu`、octave（ROCm）系列 |
| `csrc/moe/` | MoE kernel（路由、分组 GEMM 等） |
| `csrc/mamba/`、`csrc/gdn/` | Mamba / GDN（线性注意力类）kernel |
| `csrc/quantization/` | 量化 kernel（FP8/NVFP4/INT4 等） |
| `csrc/cutlass_extensions/` | CUTLASS 扩展 |
| `csrc/cpu/`、`csrc/rocm/` | CPU 与 ROCm 平台专用实现 |
| `csrc/core/` | 公共基础设施 |
| **本 fork 定制目录** | `kimi_k3/`、`inkling2/`、`qwen4_exp/`、`beam/` 等模型专属 kernel |
| **本 fork 定制融合算子（根级 .cu）** | `fused_muse_spark_qknorm_rope_kv_insert_*.cu`、`fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`、`fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu`、`dsv3_fused_a_gemm.cu`、`minimax_reduce_rms_kernel.cu`、`hy_v4_ihc_kernels.cu`、`hisparse_kernels.cu`、`nvfp4_*_kernels.cu` 等 |
| 根级通用算子 | `activation_kernels.cu`、`layernorm_kernels.cu`、`pos_encoding_kernels.cu`、`cache_kernels*.cu`、`sampler.cu`、`custom_all_reduce*.cu` 等 |

> 修改 `csrc/` 需遵守 `csrc/AGENTS.md`（kernel 开发规范）。

---

## 4. Rust 前端 `rust/`

实验性的 Rust 原生服务前端（`rust/AGENTS.md` 为权威说明），按 crate 分层，请求流向：`HTTP/gRPC DTO → chat/text request → engine request`。

| Crate（除注明外位于 `rust/src/`） | 职责 |
| --- | --- |
| `server` (`vllm-server`) | HTTP/gRPC 入口、wire DTO 与校验、响应构造 |
| `proto/`（仓库级，`vllm-proto`） | gRPC protobuf 定义 |
| `chat` (`vllm-chat`) | chat 模板渲染、多模态输入、reasoning/tool 解析驱动、响应聚合 |
| `parser` (`vllm-parser`) | 流式 reasoning / tool-call 解析器与输出文法 |
| `text` (`vllm-text`) | prompt 处理、模型加载、增量 detokenize、下沉为引擎请求 |
| `tokenizer` (`vllm-tokenizer`) | 分词器后端与增量解码 |
| `llm` (`vllm-llm`) | token-in/token-out 外观层（计时、指标） |
| `engine-core-client` (`vllm-engine-core-client`) | 到引擎核心的 ZMQ + MessagePack 传输 |
| `cmd` (`vllm-rs`) | CLI 入口与进程装配 |
| `metrics` / `tracing` / `build-info` / `managed-engine` / `mock-engine` / `bench` / `shm-object-storage` | 指标、追踪、构建信息、引擎托管、mock 引擎（压测）、压测工具、共享内存对象存储 |

---

## 5. 测试 `tests/`

按子系统镜像源码结构组织（规范见 `tests/AGENTS.md`）：

- **按功能**：`v1/`（V1 架构）、`engine/`、`entrypoints/`、`model_executor/`、`models/`（模型正确性，含 `core_model` 标记）、`kernels/`、`compile/`、`lora/`、`multimodal/`、`quantization/`、`samplers/`、`reasoning/`、`tool_parsers/`、`tokenizers_/`、`distributed/`、`platforms/`、`plugins_tests/` 等。
- **基础设施**：`conftest.py`（全局 fixture）、`vllm_test_utils/`、`utils.py`、`ci_envs.py`、`standalone_tests/`。
- **评估与基准**：`evals/`（模型评测）、`benchmarks/`。
- CI 选择逻辑在 `.buildkite/test_areas/*.yaml`，新测试需确保被某个 step 选中。

---

## 6. 其余目录

| 路径 | 职责 |
| --- | --- |
| `benchmarks/` | 独立 benchmark：`benchmark_latency.py`、`benchmark_throughput.py`、`benchmark_serving.py`、`kernels/`（kernel 微基准）、`auto_tune/`、`multi_turn/` |
| `docs/` | mkdocs 文档：`getting_started/`、`serving/`、`configuration/`、`contributing/`、`design/`、`api/` 等 |
| `examples/` | 示例：`basic/`、`generate/`、`deployment/`、`disaggregated/`、`features/`、`pooling/`、`reasoning/` 等 |
| `docker/` | 镜像构建：`Dockerfile`（CUDA）、`Dockerfile.cpu`、`Dockerfile.ppc64le`、`docker-bake*.hcl` |
| `tools/` | 构建/开发工具：`build_rust.py`、`generate_cmake_presets.py`、`ep_kernels/`、`pre_commit/`、`ci/`、`profiler/`、`recipes/` |
| `cmake/` | CMake 模块：`cpu_extension.cmake`、`utils.cmake`、`external_projects/`、`patches/` |
| `requirements/` | 依赖：`common.txt`、`cuda.txt`、`cpu.txt`、`build/`、`test/` |
| `recipes/` | 模型部署 recipe（`models/` + `manifest.yaml`） |
| `.buildkite/` | CI：pipeline、`test_areas/`（测试分区）、`scripts/` |
| `.agents/` | AI agent 技能定义（`skills/`，如 pr-checklist、add-cli-arg 等） |

---

## 7. 构建与开发速览

- **安装**（遵循 `AGENTS.md`，一律用 `uv`，禁用系统 `python3`/裸 `pip`）：
  ```bash
  uv venv --python 3.14 && source .venv/bin/activate
  VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
  ```
- **构建链路**：`pip install -e .` → `setup.py` → `CMakeLists.txt`（编译 `csrc/` 为 `_C` 扩展）+ `setuptools-rust`（构建 `rust/`）。C++/CUDA 增量编译见 `docs/contributing/incremental_build.md`。
- **测试**：`uv pip install -r requirements/test/cuda.in` 后，`.venv/bin/python -m pytest tests/path/to/test_file.py -v`。
- **Lint**：`pre-commit run --all-files`（ruff，行宽 88）。
- **Rust**：`cargo nextest run`（见 `rust/AGENTS.md`）。
- **常用入口**：`vllm serve <model>`（起服务）、`vllm bench`（基准）、`LLM(model=...)`（离线推理）。
