# Qwen 专用化精简方案与操作手册

> 目标：本仓库（vLLM fork）只用于适配 **Qwen 系列模型**，通过逐轮精简降低维护与构建成本。
>
> **已定决策**：支持范围为 Qwen 文本 + VL；仅保留 NVIDIA CUDA 平台；验证模型为 **Qwen3.8-27B**。
>
> **核心结论**：vLLM 的模型、reasoning parser、tool parser 都是**注册制 + 懒加载**，删除未被注册引用的文件是安全的；真正的风险点在**注册表同步**（`registry.py`、各 `__init__.py` 注册项）和 **C++ 绑定层**（`csrc/torch_bindings.cpp`、`ops.h`、`CMakeLists.txt`）。
>
> **分工**：本机（Windows）只做删代码 + 引用检查 + 暂存，commit 由用户完成；GPU 机器负责 docker build 和模型验证。每轮一笔提交 + 一个递增镜像 tag（`vllm-slim:rN`），失败只回退当轮。
>
> 体积参考（精简前）：`vllm/` 70M、`tests/` 44M、`docs/` 36M、`rust/` 9.9M、`csrc/` 7.8M。

---

## 1. 必须保留（Qwen 适配链路）

### 1.1 Qwen 模型实现

| 位置                                                     | 内容                                                                                                                                                                                                                          |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `vllm/model_executor/models/qwen*.py`、`colqwen*.py` | 上游 Qwen 全系：qwen2/3 文本、qwen2_vl、qwen2_5_vl、qwen3_vl(_moe)、qwen2_audio、qwen2_5_omni、qwen3_omni、qwen3_next(_mtp)、qwen3_asr 系列、qwen3_eagle3/dflash/dspark/dspine/xpress（投机解码）、qwen2_rm 等，共 ~27 个文件 |
| `vllm/models/qwen3_5/`                                 | 本 fork 定制的 Qwen3.5（`common/` + `amd/`、`nvidia/`、`xpu/` 平台实现；验证模型 Qwen3.8-27B 走 `nvidia/`）                                                                                                         |
| `vllm/models/qwen4_exp/`                               | 本 fork 定制的 Qwen4 实验版（`common/` + `amd/`、`nvidia/`）                                                                                                                                                            |
| `vllm/models/common/`                                  | 模型公共算子（`ops/`），多个定制模型共用，**保留**                                                                                                                                                                    |
| `vllm/model_executor/models/registry.py`               | 模型注册表：Qwen 相关映射保留，非 Qwen 映射删除（懒加载，删映射不影响其他模型）                                                                                                                                               |

### 1.2 Qwen 输出解析链路

| 位置                                                              | 内容                                                                          |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `vllm/reasoning/qwen3_engine_reasoning_parser.py`               | qwen3 reasoning 解析器（注册名 `qwen3`）                                    |
| `vllm/tool_parsers/qwen3_engine_tool_parser.py`                 | qwen3 工具调用解析（注册名 `qwen3_coder` / `qwen3_json` / `qwen3_xml`） |
| `vllm/tool_parsers/hermes_tool_parser.py`                       | Qwen2/2.5 系列通用 hermes 格式工具解析                                        |
| `vllm/parser/qwen3.py`                                          | 引擎侧 qwen3 解析                                                             |
| `vllm/reasoning/__init__.py`、`vllm/tool_parsers/__init__.py` | 注册表：只保留 Qwen 相关注册项                                                |

### 1.3 公共基础设施（与模型无关，全部保留）

`vllm/v1/`（引擎）、`vllm/engine/`、`vllm/config/`、`vllm/entrypoints/`、`vllm/distributed/`、`vllm/compilation/`、`vllm/model_executor/layers/`（linear/layernorm/fused_moe/rotary 等）、`vllm/model_executor/model_loader/`、`vllm/multimodal/`（Qwen-VL 需要）、`vllm/lora/`、`vllm/inputs/`、`vllm/outputs.py`、`vllm/platforms/cuda.py` 等。

### 1.4 Qwen 相关 C++ kernel

| 位置                                 | 内容                                                                                                    |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `csrc/qwen4_exp/hc_up_gate_mix.cu` | Qwen4 实验版专用 kernel                                                                                 |
| 通用 kernel（保留）                  | `activation/layernorm/pos_encoding/cache/sampler/moe/attention/quantization` 等目录及根级通用 `.cu` |

---

## 2. 可精简候选（按风险从低到高）

### L0 — 纯资产，删除零风险（✅ 完成）

| 目标                           | 说明                                                                                    | 状态     |
| ------------------------------ | --------------------------------------------------------------------------------------- | -------- |
| `tests/`（44M）              | 全量删除                                                                                | ✅ r1a   |
| `docs/`（36M）               | 全量删除                                                                                | ✅ r1b   |
| `examples/`、`benchmarks/` | 全量删除                                                                                | ✅ r1c   |
| `recipes/models/` 非 Qwen    | 只保留 `Qwen/`                                                                        | ✅ r1d-1 |
| `docker/` 非 CUDA 文件       | Dockerfile.cpu/xpu/zen/s390x/ppc64le/rocm*、build_vllm_ppc64le.sh、docker-bake-rocm.hcl | ✅ r1d-2 |
| `.buildkite/test_areas/`     | 随 tests 删除而失效的测试分区配置；`.buildkite` 其余 CI 配置保留不动                  | ✅ r1d-3 |

### L1 — 注册制懒加载模块，删除安全（需同步注册表）（✅ 完成）

| 目标                                                                                                                                                                                | 说明                                                                                                                                                                                    | 状态            |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `vllm/models/` 下非 Qwen 定制模型：kimi_k3、muse_spark、deepseek 系列、inkling/2、minimax_m3、glm5next、hy_v4(_flash)、step5、mistral、dots3_note、flint、nemotron_shortcut、beam | 17 个目录、1088 个文件、43 万行已删除；同步清理了 quantization/warmup 启动路径引用与 `registry.py` 映射；现仅剩 `common/`、`qwen3_5/`、`qwen4_exp/`                             | ✅ r2a-1～r2a-4 |
| `vllm/model_executor/models/` 非 Qwen 上游模型                                                                                                                                    | 254 个文件、16.9 万行已删除（按 import 闭包保留 43 个 .py + `transformers/`）；`registry.py` 映射同步删除；误删的 `expert_substitution.py` 已恢复                                 | ✅ r2b-1、r2b-2 |
| `vllm/reasoning/` 非 Qwen 解析器                                                                                                                                                  | 41 个文件、4214 行已删除；仅剩 `__init__.py`（注册表仅 qwen3）、`abs_reasoning_parsers.py`、`qwen3_engine_reasoning_parser.py`                                                    | ✅ r2c-1        |
| `vllm/tool_parsers/` 非 Qwen 解析器                                                                                                                                               | 57 个文件、12224 行已删除；注册表仅 hermes/qwen3_coder/qwen3_json/qwen3_xml；保留 `streaming.py`/`structural_tag_registry.py`/`tool_strict_level.py`/`utils.py` 基础设施        | ✅ r2c-2        |
| `vllm/renderers/` 定制渲染器                                                                                                                                                      | 10 个模型专属渲染器、2624 行已删除；`_VLLM_RENDERERS` 仅保留 hf                                                                                                                       | ✅ r2c-3        |
| `vllm/tokenizers/` 定制分词器                                                                                                                                                     | 8 个模型专属分词器、2548 行已删除；保留 hf/mistral（pixtral 引用）/fastokens（env 门控）                                                                                                | ✅ r2c-4        |
| `vllm/parser/` 非 Qwen 解析器                                                                                                                                                     | 27 个模型专属解析器、7049 行已删除；`registered_adapters.py` 仅 Qwen3；保留 harmony（responses API 依赖，gptoss 改懒加载）、response_template + chat_parsing（组合必经路径）、engine/ | ✅ r2c-5        |

分三组执行完毕：**r2a** 定制模型（清 quantization 引用 → 清 warmup 引用 → 删注册映射 → 删目录）；**r2b** 上游模型（先算 import 闭包再删）；**r2c** 解析链五项（模型删干净后的死代码，一项一步）。

### L2 — 平台与后端（已确认仅 CUDA + 文本/VL）

| 目标                                                                                                                                                                                                                                                                                        | 说明                                                                                                                                                                         | 状态   |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| `vllm/platforms/`：`rocm.py`、`tpu.py`、`xpu.py`、`cpu.py`、`zen_cpu.py`                                                                                                                                                                                                        | 精简为 `cuda.py` + `interface.py`；`vllm/models/qwen3_5/amd`、`qwen3_5/xpu`、`qwen4_exp/amd` 同步删除                                                              | ⬜ r3a |
| `csrc/` 本 fork 定制 kernel：`kimi_k3/`、`inkling2/`、`beam/`、`fused_muse_spark_*`、`fused_deepseek_v4_*`、`fused_minimax_m3_*`、`minimax_reduce_rms_*`、`hy_v4_ihc_*`、`hisparse_*`、`dsv3_fused_a_gemm.cu`、`nvfp4_ds_mla_*`；以及 `csrc/rocm/`、`csrc/cpu/` | 随模型删除后这些 kernel 无调用方；**必须同步清理 `csrc/torch_bindings.cpp`、`ops.h`、`CMakeLists.txt`，否则编译失败**                                            | ⬜ r3b |
| `vllm/v1/attention/backends/`（可选）                                                                                                                                                                                                                                                     | 保留 `flash_attn.py`、`flashinfer.py`（qwen3_5 用 FLASH_ATTN + GDN_ATTN + FLASHINFER）、`gdn_attn.py` 等 Qwen 实际用到的后端；`rocm_*`、`zentorch_*`、`mla` 可删 | ⬜ r3c |
| `vllm/v1/spec_decode/`、`vllm/lora/`、`vllm/multimodal/`                                                                                                                                                                                                                              | **不建议删**：Qwen 的 eagle3/mtp、VL 模型都依赖（保留项，不占轮次）                                                                                                    | 保留   |

分三步执行：**r3a** 纯 Python 平台代码；**r3b** csrc 精简（构建硬耦合，绑定/头文件/CMake 三处一起改）；**r3c** attention 后端精简（可选）。

### L3 — 整体不构建的组件

| 目标                                                                                           | 说明                                                                                     | 状态   |
| ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ------ |
| `rust/`（9.9M）                                                                              | 实验性 Rust 前端；只用 Python 前端（`vllm serve`），构建流程中跳过即可，目录保留不编译 | ⬜ r4a |
| `vllm/v1/kv_offload/`、`vllm/distributed/kv_transfer/` 的各 connector（mooncake、nixl 等） | 不做 PD 分离/KV 传输时可裁，但注意 import 链                                             | ⬜ r4b |
| `vllm/benchmarks/`、`vllm/profiler/`、`vllm/tracing/`、`vllm/usage/`                   | 非核心路径，可按需裁                                                                     | ⬜ r4c |

---

## 3. 执行轮次与状态

每轮 = 本机删除+暂存（用户提交）→ GPU 机器构建对应 tag 镜像 → 运行期验证 → 通过再进下一轮。
轮次编号与第 2 节风险层级对应：**L0 = r1 系列、L1 = r2 系列、L2 = r3 系列、L3 = r4 系列**。

| 轮次      | 内容                                                                                                                         | 状态      |
| --------- | ---------------------------------------------------------------------------------------------------------------------------- | --------- |
| r0        | 验证资产：`Dockerfile.slim`、`slim_check.py`、文档                                                                       | ✅ 已完成 |
| r1a       | 删 `tests/`                                                                                                                | ✅ 已完成 |
| r1b       | 删 `docs/`                                                                                                                 | ✅ 已完成 |
| r1c       | 删 `examples/`、`benchmarks/`                                                                                            | ✅ 已完成 |
| r1d-1/2/3 | 删非 Qwen recipes / 非 CUDA docker 文件 / `.buildkite/test_areas`                                                          | ✅ 已完成 |
| r2a-1～4  | 删 `vllm/models/` 非 Qwen 定制模型（清 quantization 引用 → 清 warmup 引用 → 删注册映射 → 删目录）                       | ✅ 已完成 |
| r2b       | 删 `vllm/model_executor/models/` 非 Qwen 上游模型 + 同步 `registry.py`（含 expert_substitution 误删恢复）                | ✅ 已完成 |
| r2c-1～5  | 解析链精简：reasoning → tool_parsers → renderers → tokenizers → parser，一项一步 + 同步各注册表（含 harmony 懒加载修复） | ✅ 已完成 |
| r3a       | L2 平台 Python 代码精简（`platforms/` 非 CUDA + `models/*/amdxpu`）                                                      | ⬜ 下一步 |
| r3b       | L2 `csrc/` 定制 kernel + `csrc/rocm`、`csrc/cpu` 删除 + 同步 `torch_bindings.cpp`/`ops.h`/`CMakeLists.txt`       | ⬜        |
| r3c       | L2（可选）attention 后端精简                                                                                                 | ⬜        |
| r4a       | L3 `rust/` 前端跳过构建                                                                                                    | ⬜        |
| r4b       | L3 KV connector / kv_offload 裁剪                                                                                            | ⬜        |
| r4c       | L3 `vllm/benchmarks/`、`profiler/`、`tracing/`、`usage/` 裁剪                                                        | ⬜        |

---

## 4. 每轮验证操作（GPU 机器）

```bash
# 0. 同步代码（本机已提交并推送后）
git pull

# 1. 构建镜像（构建上下文必须是仓库根；代理地址按实际改，不需要代理就去掉两个 --build-arg）
docker build \
  --build-arg HTTP_PROXY=http://192.168.1.104:7890 \
  --build-arg HTTPS_PROXY=http://192.168.1.104:7890 \
  -f qwen/Dockerfile.slim -t vllm-slim:rN .
```

**验证方式 A（起服务 + 真实请求，推荐）**：

```bash
docker run -it --rm --name Qwen38-27B-test --gpus all --ipc=host -p 8000:8000 \
  -v /data/models/Qwen3.8-27B:/model --entrypoint vllm vllm-slim:rN \
  serve /model --served-model-name Qwen3.8-27B --tensor-parallel-size 1 \
  --max-model-len 248320 --max-num-seqs 32 --gpu-memory-utilization 0.92 \
  --kv-cache-dtype fp8_e4m3 --enable-prefix-caching --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --host 0.0.0.0 --port 8000

# 另开终端，看到 Application startup complete 后：
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B","messages":[{"role":"user","content":"你好，用一句话介绍杭州"}]}'
# 期望：返回正常文本；VL 模型另验一条带 image_url 的图文请求
```

**验证方式 B（快速冒烟，离线 generate）**：

```bash
docker run --rm --gpus all -v /data/models/Qwen3.8-27B:/model \
  vllm-slim:rN --model /model
# 期望结尾输出：SLIM_CHECK PASSED（退出码 0）
```

## 5. 失败回退

```bash
# 代码回退（本机，由用户执行）
git revert <该轮 commit>        # 或 git reset --hard HEAD~1（未推送时）

# 镜像回退（GPU 机器）
docker run ... vllm-slim:r<N-1> ...   # 回到上一轮通过的镜像
```

---

## 6. 注意事项

**精简操作相关**

1. **注册表是安全边界**：`registry.py`、`reasoning/__init__.py`、`tool_parsers/__init__.py`、`renderers/registry.py`、`tokenizers/registry.py` 都是懒加载——先删注册项，再用 `grep -r "模块名" vllm/` 确认无引用后删文件。注意除注册表外的**无条件 import**（如 `kernel_warmup.py` 对 deepseek_v41 的引用）。
2. **公共依赖不可删**：`vllm/model_executor/layers/`、`vllm/models/common/`、各 `interfaces.py`/`utils.py`、attention backend 的 `registry.py`/`selector.py` 框架代码。
3. **C++ 侧是唯一硬耦合**：Python 侧删除是惰性的，但 `torch_bindings.cpp` 里注册的算子若在 `.cu` 中不存在会直接编译失败——csrc 精简必须绑定、头文件、CMake 三处一起改。
4. **权衡**：删得越狠，与上游 vLLM rebase/merge 的冲突越少，但删除 `registry.py` 行本身会成为每次 sync 的冲突点。替代方案是不删代码、只裁剪构建与打包（wheel 排除 tests/docs/examples），维护成本最低。
5. 每轮验证通过后再开始下一轮；不要合并多轮一次验证，否则难以定位问题轮次。

**r2 踩坑记录**（后续轮次参照）

- **函数内 import ≠ 不执行**：`initialize_model`、`get_quantization_config` 等每次启动都跑的函数，其内部 import 也是硬依赖。判断标准不是"import 写在哪"，而是"这段代码在 Qwen 启动路径上会不会跑到"。
- 误删后恢复的基础设施：`model_executor/models/expert_substitution.py`（模型加载必经）、`tool_parsers/` 的 `streaming.py`/`structural_tag_registry.py`/`tool_strict_level.py`（解析框架与 CLI 参数依赖）。删目录前应先排查同名目录下的"非模型"工具文件。
- `parser/harmony.py` 被 `parser/__init__.py` 和 responses API 模块级引用，保留文件、其 gptoss 依赖改懒加载；`parser/response_template*.py` + `chat_parsing/` 是 parser_manager 组合解析器的必经路径，保留。
- `tokenizers/mistral.py` 被保留的 `pixtral.py`（Qwen VL 依赖）引用，不能按"mistral 是非 Qwen 模型"直接删。
- `qwen4_exp/common/ngram_embedding.py` 对已删 deepseek_v41 engram 的依赖已改懒加载 + ImportError 回退；`qwen3_5/amd` 对 minimax_m3 的依赖留待 r3a 删 amd 目录时解决。

**Dockerfile.slim 机制相关**（已踩过的坑，勿回退这些设计）

6. `FROM` 固定为 `vllm/vllm-openai:v0.33.0`——其 torch 2.14.0 与 `pyproject.toml` 匹配；`latest`（0.28.0/torch 2.13）与仓库代码不兼容。
7. 替换包时必须还原三类**不在 git 仓库里**的文件：`*.so` 编译产物、`vllm_flash_attn/` 与 `third_party/` 下构建期由 setup.py 拷入的 Python 文件、`_version.py`。漏还原会报各种 `ModuleNotFoundError`。
8. fork 的 qwen3_5 GDN kernel 依赖 `flashinfer-python==0.7.4` 的新参数（`checkpoint_state_indices`），Dockerfile 中 pip 升级不可省略；构建期下载需要代理时用 `--build-arg HTTP_PROXY/HTTPS_PROXY` 传入，不要写进 Dockerfile。
9. **没有构建期 import 冒烟检查**（该构建环境下 MKL 会误报），验证统一在运行期；镜像 ENTRYPOINT 是 `slim_check.py`，起服务要用 `--entrypoint vllm` 覆盖。
10. 镜像比基础镜像大约 2.4GB 是备份/替换产生的重复层所致，验证阶段可接受；正式部署镜像再优化（合并层或 overlay）。
