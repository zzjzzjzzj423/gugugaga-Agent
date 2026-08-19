# Simple CC（SiliconFlow 版）

Simple CC routes explicitly marked research tasks through a bounded evidence workflow; ordinary tasks continue to use the general agent loop.

项目从 `learn-claude-code` S20 综合示例演化而来，保留精简的 Agent Loop 与工具调用控制流，并通过 SiliconFlow 的 OpenAI 兼容 Chat Completions / Function Calling 接口调用模型。

模型请求只有一个内部边界：`SiliconFlowProvider.create()`。Agent、一次性 Subagent、Team Teammate 和上下文摘要都使用相同的 S20 内容块协议（`text`、`tool_use`、`tool_result`）。

## 功能范围

- Agent Loop、Bash、Read、Write、Edit、Glob
- 权限检查、Hooks、Todo、按需 Skills、一次性 Subagent
- 上下文压缩、长期 Memory、动态 System Prompt、错误恢复
- 持久化 Task DAG、后台任务、Cron
- Agent Team、计划审批、优雅关闭、自主任务认领

明确不包含：

- S18 Worktree Isolation
- S19 MCP Plugin
- Web UI 或 HTTP 服务

## 安装

```powershell
cd E:\AgentLearnProject\simple_cc
python -m pip install -r requirements.txt
python -m pip install -e ".[test]"
```

在启动 Simple CC 的进程环境中设置 SiliconFlow 变量：

```powershell
$env:SILICONFLOW_API_KEY = "your-api-key"
$env:SILICONFLOW_MODEL = "a-function-calling-model"
$env:SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"  # 可选
$env:SILICONFLOW_FALLBACK_MODEL = "fallback-model"           # 可选
```

`SILICONFLOW_API_KEY` 和 `SILICONFLOW_MODEL` 都是必需的。为避免仓库 `.env` 意外覆盖启动配置，程序不会自动读取 `.env`；缺少进程环境中的必需变量时会立即给出配置错误。不要把真实密钥提交到版本库。

## 工作空间

不传参数时，启动目录就是工作空间：

```powershell
python -m simple_cc
```

也可以显式选择：

```powershell
python -m simple_cc --workspace E:\your-project
```

`--workspace` 会在任何状态初始化或工具执行前同步所有派生路径。文件工具、Skills、Tasks、Mailboxes、Memory、Transcripts、大型 Tool Results 和 Cron 持久化都位于选定工作空间下。文件工具拒绝逃逸工作空间的路径；Bash 仍由权限 Hook 控制。

可用退出输入为 `q`、`exit`、`/exit` 和 `/quit`，也可以按 Ctrl+C。使用 `--model MODEL` 可覆盖进程环境中的 `SILICONFLOW_MODEL`。

## 测试与检查

测试使用脚本化 Provider，不访问真实网络，也不需要真实 API Key：

```powershell
python -m pytest -q
python -m compileall simple_cc
python -m simple_cc --help
```

## 金融 Agent benchmark 评测

数据集使用 JSONL，每行至少包含 `task_id`、`question`，可选字段包括
`cutoff`、`benchmark` 和 `task_type`。以下命令分别运行 20 题试验集和
400 题公开集：

```powershell
python -m eval.run_benchmark --dataset eval/data/financegym_20.jsonl --workers 2
python -m eval.run_benchmark --dataset eval/data/benchmark_400_public.jsonl --workers 4 --resume
```

每道题由一个全新的 Python 进程执行，并使用独立且初始为空的 workspace。
benchmark 默认关闭 memory、cron、team、subagent 和交互式授权；文件工具不能
越出 workspace，shell 因没有人工授权回调而拒绝执行。上一题的对话、搜索结果、
缓存、工作文件、后台任务、答案和评测文件都不会注入下一题。

默认输出位于 `eval/runs`，目录结构如下：

```text
eval/runs/<run_id>/
├── task_input.json
├── manifest.json
├── trajectory.jsonl
├── final_answer.txt          # 只有成功完成时存在
├── artifacts/                # 按 SHA-256 命名并去重
└── agent_workspace/          # Agent 唯一可见的文件工作区
```

`trajectory.jsonl` 是即时 `fsync` 的追加式事件流，记录模型请求/响应、Token、
重试次数、模型及工具时延、工具输入输出、cutoff 决策、来源注册和最终引用关联。
较大的请求、响应、抓取正文和 PDF 放入 `artifacts/`，事件只保存相对路径、
SHA-256、媒体类型和字节数。若任一成功模型调用没有返回 usage，相应核心或全量
Token 指标为 `null`，不会把未知值当作 0。

终态包括 `completed`、`failed`、`max_rounds`、`cancelled`、`timed_out`、
`worker_crashed` 和 `trace_invalid`。只有 Agent 正常结束、资源清理完成、最终答案
落盘且 manifest 原子更新后才是 `completed`。超时由父进程终止整个子进程树；
崩溃和超时不会生成成功答案。使用 `--resume` 时，已完成题目会跳过；未完成题目
会创建新的 `run_id` 和目录，并在 `task_input.json`/manifest 中记录
`retry_of_run_id`，不会覆盖旧尝试。

PIT 模式固定披露为 `non_strict_live_web`。`cutoff` 会被强制注入搜索、网页抓取
和 PDF 工具，冲突的 cutoff 会被拒绝，明确晚于 cutoff 的正文不会进入模型可见
artifact。但是搜索引擎的 `before:` 仅是实时搜索提示，不能证明网页在历史时点
已经可见，因此这不是冻结语料或严格历史回放，不能宣称为 strict PIT。

## 对话内网页检索

`simple_cc` 在原有对话循环中提供 `web_search` 和 `web_fetch`。不需要单独
启动搜索界面；直接在对话框里提出研究问题，模型会按需搜索并读取网页。例如：

```text
请仅使用 2025-08-05 及以前公开发布的信息研究这个问题。列出来源 URL，
并明确标注发布日期无法确认的证据。
```

如果问题给出截止日期，Agent 会把相同的 `cutoff` 传给搜索和网页读取工具。
确认晚于截止日期的网页会被拒绝；发布日期未知的网页会带有警告，最终回答也应
披露这一限制。

当前实现使用实时公共网页搜索，属于 **internal preview / non-strict PIT**。
它适合验证检索流程和进行轻量内部对比，但不等价于 FinanceGym 的冻结历史语料，
结果不能直接与官方排行榜或论文分数比较。

## 对话内读取 PDF 财报

`simple_cc` 使用 `pdf_fetch` 读取公开 HTTP/HTTPS URL 上带文本层的 PDF。例如：

```text
请读取 https://example.com/annual-report.pdf 的第 11–20 页，回答时标注 PDF 页码。
```

PDF 页码从 1 开始；每次默认读取 10 页，最多读取 20 页。工具返回
`total_pages`、实际页段、逐页边界和 `has_more`，Agent 可在需要更多证据时继续
读取后续页段。财报表格会附加为制表符分隔文本，回答应同时引用来源 URL 和
PDF 页码。

PDF 元数据无法确认发布日期时，结果会带有提示。首版不支持扫描件 OCR；没有
可提取文本层的页段会返回 `ocr_required`。

端到端测试覆盖 S20 的调用顺序：用户消息 → `tool_use` → 固定 Handler → `tool_result` → 最终文本；同时覆盖一次且仅一次的 prompt-too-long 响应式压缩重试、`compact` 的特殊历史变更、默认/显式工作空间和 CLI 清理。
