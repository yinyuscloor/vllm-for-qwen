# Qwen 专用化精简梳理

> 目标：本仓库（vLLM fork）只用于适配 **Qwen 系列模型**，梳理可以精简/移除的部分，降低维护与构建成本。
>
> 结论先行：vLLM 的模型、reasoning parser、tool parser 都是**注册制 + 懒加载**，删除未被注册引用的文件是安全的；真正的风险点在**注册表同步**（`registry.py`、`__init__.py` 注册项）和 **C++ 绑定层**（`csrc/torch_bindings.cpp`、`ops.h`、`CMakeLists.txt`）。
>
> 体积参考：`vllm/` 70M、`tests/` 44M、`docs/` 36M、`rust/` 9.9M、`csrc/` 7.8M。

---

## 1. 必须保留（Qwen 适配链路）

### 1.1 Qwen 模型实现

| 位置 | 内容 |
| --- | --- |
| `vllm/model_executor/models/qwen*.py`、`colqwen*.py` | 上游 Qwen 全系：qwen2/3 文本、qwen2_vl、qwen2_5_vl、qwen3_vl(_moe)、qwen2_audio、qwen2_5_omni、qwen3_omni、qwen3_next(_mtp)、qwen3_asr 系列、qwen3_eagle3/dflash/dspark/dspine/xpress（投机解码）、qwen2_rm 等，共 ~27 个文件 |
| `vllm/models/qwen3_5/` | 本 fork 定制的 Qwen3.5（`common/` + `amd/`、`nvidia/`、`xpu/` 平台实现） |
| `vllm/models/qwen4_exp/` | 本 fork 定制的 Qwen4 实验版（`common/` + `amd/`、`nvidia/`） |
| `vllm/models/common/` | 模型公共算子（`ops/`），多个定制模型共用，**保留** |
| `vllm/model_executor/models/registry.py` | 模型注册表：Qwen 相关映射保留，非 Qwen 映射删除（懒加载，删映射不影响其他模型） |

### 1.2 Qwen 输出解析链路

| 位置 | 内容 |
| --- | --- |
| `vllm/reasoning/qwen3_engine_reasoning_parser.py` | qwen3 reasoning 解析器（注册名 `qwen3`） |
| `vllm/tool_parsers/qwen3_engine_tool_parser.py` | qwen3 工具调用解析（注册名 `qwen3_coder` / `qwen3_json` / `qwen3_xml`） |
| `vllm/tool_parsers/hermes_tool_parser.py` | Qwen2/2.5 系列通用 hermes 格式工具解析 |
| `vllm/parser/qwen3.py` | 引擎侧 qwen3 解析 |
| `vllm/reasoning/__init__.py`、`vllm/tool_parsers/__init__.py` | 注册表：只保留 Qwen 相关注册项 |

### 1.3 公共基础设施（与模型无关，全部保留）

`vllm/v1/`（引擎）、`vllm/engine/`、`vllm/config/`、`vllm/entrypoints/`、`vllm/distributed/`、`vllm/compilation/`、`vllm/model_executor/layers/`（linear/layernorm/fused_moe/rotary 等）、`vllm/model_executor/model_loader/`、`vllm/multimodal/`（Qwen-VL 需要）、`vllm/lora/`、`vllm/inputs/`、`vllm/outputs.py`、`vllm/platforms/cuda.py` 等。

### 1.4 Qwen 相关 C++ kernel

| 位置 | 内容 |
| --- | --- |
| `csrc/qwen4_exp/hc_up_gate_mix.cu` | Qwen4 实验版专用 kernel |
| 通用 kernel（保留） | `activation/layernorm/pos_encoding/cache/sampler/moe/attention/quantization` 等目录及根级通用 `.cu` |

---

## 2. 可精简候选（按风险从低到高）

### L0 — 纯资产，删除零风险

| 目标 | 说明 |
| --- | --- |
| `tests/`（44M） | 只保留 `conftest.py`、`utils.py` 及 Qwen 相关用例；`tests/models/` 下非 Qwen 模型测试、`tests/lora`、`tests/multimodal` 中非 Qwen 部分等可大量删除。注意同步 `.buildkite/test_areas/*.yaml` |
| `docs/`（36M） | 只保留构建/部署相关章节；或直接不构建文档站 |
| `examples/`、`benchmarks/` | 保留 Qwen 相关示例与通用 benchmark 脚本 |
| `recipes/models/` | 只保留 Qwen 的 recipe |
| `docker/Dockerfile.cpu`、`Dockerfile.ppc64le`、ROCm bake 配置 | 若只部署 CUDA |

### L1 — 注册制懒加载模块，删除安全（需同步注册表）

| 目标 | 规模 | 同步点 |
| --- | --- | --- |
| `vllm/models/` 下非 Qwen 定制模型：`kimi_k3/`、`muse_spark/`、`deepseek_v32/`、`deepseek_v4/`、`deepseek_v41/`、`inkling/`、`inkling2/`、`minimax_m3/`、`glm5next/`、`hy_v4/`、`hy_v4_flash/`、`step5/`、`mistral/`、`dots3_note/`、`flint/`、`nemotron_shortcut/`、`beam/` | 18 个目录 | `registry.py` 中对应映射 |
| `vllm/model_executor/models/` 非 Qwen 上游模型 | ~296 个文件中删 ~270 个 | `registry.py`；注意保留 `interfaces.py`、`utils.py`、`adapters.py` 等公共文件及 Qwen 文件的 import 依赖（先删注册项，再按 import 报错收敛删除集） |
| `vllm/reasoning/` 非 Qwen 解析器 | 44 个 .py 中删 ~40 个 | `reasoning/__init__.py` 的 `_REASONING_PARSERS_TO_REGISTER`；保留 `abs_reasoning_parsers.py`、`basic_parsers.py`、qwen3 |
| `vllm/tool_parsers/` 非 Qwen 解析器 | 64 个 .py 中删 ~60 个 | `tool_parsers/__init__.py` 注册表；保留 `abstract_tool_parser.py`、hermes、qwen3 |
| `vllm/renderers/` 定制渲染器：`deepseek_v32/v4`、`hy_v4`、`inkling*`、`kimi_k3`、`mistral`、`cohere`、`terratorch`、`unlimited_ocr2` 等 | ~10 个文件 | `renderers/registry.py`；保留 `hf.py`（Qwen 走 HF 渲染）与 `base.py`、`chat_utils/`、`inputs/` |
| `vllm/tokenizers/` 定制分词器：`deepseek_*`、`kimi_audio`、`alpamayo2_super`、`mistral`、`fastokens`（按需） | ~9 个文件 | `tokenizers/registry.py`；保留 `hf.py` |
| `vllm/parser/` 非 Qwen 解析器 | 30+ 个文件 | `parser/parser_manager.py` / `__init__.py`；保留 `abstract_parser.py`、`qwen3.py`、`llama_json.py`（hermes 依赖）、`chat_parsing/`、`engine/`、`utils.py` |

### L2 — 平台与后端，删前需确认部署目标

| 目标 | 说明 |
| --- | --- |
| `vllm/platforms/`：`rocm.py`、`tpu.py`、`xpu.py`、`cpu.py`、`zen_cpu.py` | 只部署 NVIDIA CUDA 时可精简为 `cuda.py` + `interface.py`；**注意 `vllm/models/qwen3_5/amd/`、`qwen4_exp/amd/` 是 ROCm 实现，若确认不用 AMD 也可删** |
| `vllm/v1/attention/backends/` | 35+ 后端中 Qwen 只需 `flash_attn.py`（及按需 `flashinfer.py`、`triton_attn.py`、`flex_attention.py`、`cpu_attn.py`）；`rocm_*`、`zentorch_*`、`mamba*`（qwen3_next 是 hybrid，需要 `mamba_attn`/`gdn_attn`，**按实际支持的 Qwen 变体保留**）、`mla`（Qwen 不用 MLA 可删，但 kimi/deepseek 相关已随模型删除） |
| `csrc/` 本 fork 定制 kernel：`kimi_k3/`、`inkling2/`、`beam/`、`fused_muse_spark_*`、`fused_deepseek_v4_*`、`fused_minimax_m3_*`、`minimax_reduce_rms_*`、`hy_v4_ihc_*`、`hisparse_*`、`dsv3_fused_a_gemm.cu`、`nvfp4_ds_mla_*` | 随模型删除后这些 kernel 无调用方；**必须同步清理 `csrc/torch_bindings.cpp`、`ops.h`、`CMakeLists.txt`，否则编译失败** |
| `csrc/rocm/`、`csrc/cpu/` | 同平台精简决策 |
| `vllm/v1/spec_decode/`、`vllm/lora/`、`vllm/multimodal/` | **不建议删**：Qwen 的 eagle3/mtp（qwen3_eagle3、qwen3_next_mtp）、VL/omni 模型都依赖 |

### L3 — 整体不构建的组件

| 目标 | 说明 |
| --- | --- |
| `rust/`（9.9M） | 实验性 Rust 前端；若只用 Python 前端（`vllm serve`），可在 `setup.py`/构建流程中跳过 Rust 构建，目录保留不编译 |
| `vllm/v1/kv_offload/`、`vllm/distributed/kv_transfer/` 的各 connector（mooncake、nixl 等） | 不做 PD 分离/KV 传输时可裁，但注意 import 链 |
| `vllm/benchmarks/`、`vllm/profiler/`、`vllm/tracing/`、`vllm/usage/` | 非核心路径，可按需裁 |

---

## 3. 执行建议与注意事项

1. **顺序**：先 L0（测试/文档/示例）→ L1（模型与解析器，注册表同步）→ L2（平台/kernel，需改 CMake 与绑定）→ L3（构建开关）。每步跑一次 import smoke test（`python -c "import vllm"`）和一次 Qwen 模型端到端推理。
2. **注册表是安全边界**：`registry.py`、`reasoning/__init__.py`、`tool_parsers/__init__.py`、`renderers/registry.py`、`tokenizers/registry.py` 都是懒加载——先删注册项，再用 `grep -r "模块名" vllm/` 确认无引用后删文件。
3. **公共依赖不可删**：`vllm/model_executor/layers/`、`vllm/models/common/`、各 `interfaces.py`/`utils.py`、attention backend 的 `registry.py`/`selector.py` 框架代码。
4. **C++ 侧是唯一硬耦合**：Python 侧删除是惰性的，但 `torch_bindings.cpp` 里注册的算子若在 `.cu` 中不存在会直接编译失败——csrc 精简必须绑、头、CMake 三处一起改。
5. **权衡**：删得越狠，与上游 vLLM rebase/merge 的冲突越少（上游新增模型文件不再引入），但删除 `registry.py` 行本身会成为每次 sync 的冲突点。替代方案是**不删代码，只裁剪构建与打包**（wheel 排除 tests/docs/examples），维护成本最低。
6. **建议先明确的决策点**：
   - 需要支持哪些 Qwen 变体？（纯文本 qwen3/qwen3_moe？VL/omni？qwen3_next hybrid？qwen3_5/qwen4_exp 定制版？）——决定 mamba/multimodal/spec_decode 是否保留。
   - 部署平台只有 NVIDIA CUDA，还是含 ROCm/XPU？——决定 `platforms/`、`models/*/amd|xpu`、`csrc/rocm` 的去留。
   - 是否需要 Rust 前端、gRPC、PD 分离部署？
