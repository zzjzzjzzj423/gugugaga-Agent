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

端到端测试覆盖 S20 的调用顺序：用户消息 → `tool_use` → 固定 Handler → `tool_result` → 最终文本；同时覆盖一次且仅一次的 prompt-too-long 响应式压缩重试、`compact` 的特殊历史变更、默认/显式工作空间和 CLI 清理。
