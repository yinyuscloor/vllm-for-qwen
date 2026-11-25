# 精简轮次操作手册（SLIM RUNBOOK）

> 每轮四步：删代码+提交（本机）→ docker build（GPU 机器）→ 跑验证脚本 → 失败回退。
> 镜像 tag 递增：`vllm-slim:r1`、`r2`……每一轮对应一笔 git 提交和一个镜像 tag。

## 各轮范围

| 轮次 | 内容 | 状态 |
| --- | --- | --- |
| r0 | 验证资产（本文件、`Dockerfile.slim`、`slim_check.py`） | ✅ 已提交 |
| r1 | 删除 L0 资产：`tests/`、`docs/`、`examples/`、`benchmarks/`、非 Qwen recipes、非 CUDA docker 文件 | |
| r2 | 删除 `vllm/models/` 非 Qwen 定制模型 + 同步 `registry.py` | |
| r3 | 删除 `vllm/model_executor/models/` 非 Qwen 上游模型 + 同步 `registry.py` | |
| r4 | 解析链精简（reasoning/tool_parsers/parser/renderers/tokenizers）+ 同步各注册表 | |
| r5 | 平台精简（仅 CUDA）+ `csrc/` 定制 kernel + 同步 `torch_bindings.cpp`/`ops.h`/`CMakeLists.txt` | |
| r6 | （可选）attention 后端精简 | |

详细候选清单见 `qwen/QWEN_SLIMMING.md`。

## 每轮操作（GPU 机器上执行第 2–3 步）

```bash
# 1.（本机已完成）删代码 + 同步引用 + git 提交，推送/同步到 GPU 机器
git pull   # 或你的同步方式

# 2. 构建镜像（构建上下文必须是仓库根）
docker build -f qwen/Dockerfile.slim -t vllm-slim:rN .

# 3. 验证：模型加载 + 两条 generate 有非空输出
docker run --rm --gpus all \
  -v /path/to/models:/models \
  vllm-slim:rN --model /models/Qwen3.8-27B
# 期望输出结尾：SLIM_CHECK PASSED（退出码 0）
```

## 失败处理

```bash
# 代码回退（本机）
git revert <该轮 commit>        # 或 git reset --hard HEAD~1（未推送时）

# 镜像回退（GPU 机器）
docker run ... vllm-slim:r<N-1> ...   # 回到上一轮镜像继续工作
```

## 最终验收（最后一轮后）

```bash
# 起服务并打一条真实请求
docker run --rm --gpus all -p 8000:8000 --entrypoint vllm \
  vllm-slim:rFinal serve /models/Qwen3.8-27B

curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/Qwen3.8-27B","messages":[{"role":"user","content":"你好"}]}'
# 期望：返回正常 completion
# VL 模型另验一条带 image_url 的图文请求
```

## 注意事项

- `Dockerfile.slim` 基于官方 `vllm/vllm-openai` 镜像，用仓库 `vllm/` **覆盖**安装包，保留镜像内编译好的 `.so`；若官方镜像的 vLLM 版本与本仓库差异过大导致 import 失败，把 `FROM` 改成与仓库版本接近的 tag。
- 构建镜像时 `import vllm` 已做冒烟检查，构建失败即说明本轮删除有误，无需进入模型验证。
- 每轮验证通过后再开始下一轮；不要合并多轮一次验证，否则难以定位问题轮次。
