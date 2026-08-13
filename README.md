# Simple CC（SiliconFlow 版）

Simple CC 是一个从 `learn-claude-code` S20 综合示例拆分出的教学型 Coding Agent。它保留 S01 Agent Loop 到 S17 Autonomous Agents 的控制流，并通过 SiliconFlow 的 OpenAI 兼容 Chat Completions / Function Calling 接口调用模型。

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
