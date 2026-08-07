# Simple CC（SiliconFlow 版）

一个独立、模块化的教学型 Coding Agent。它整理并整合了
`learn-claude-code` 从 S01 Agent Loop 到 S17 Autonomous Agents 的完整能力，
模型调用统一使用硅基流动的 OpenAI 兼容 Chat Completions / Function Calling
接口。

## 包含能力

- Agent Loop、Bash/Read/Write/Edit/Glob
- 权限检查与 Hooks
- Todo、一次性 Subagent、按需 Skills
- 上下文压缩、长期 Memory、动态 System Prompt、错误恢复
- 持久化 Task DAG、后台任务、Cron
- Agent Team、计划审批、优雅关机、自治任务认领

不包含 S18 Worktree、S19 MCP、Web UI 和 HTTP 服务。

## 安装

```powershell
cd E:\AgentLearnProject\simple_cc
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`：

```dotenv
SILICONFLOW_API_KEY=你的密钥
SILICONFLOW_MODEL=硅基流动中支持 Function Calling 的模型名
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
```

硅基流动支持的模型会调整，请在模型广场确认目标模型当前支持 Function
Calling，不要照抄过期模型名。

## 运行

```powershell
python -m simple_cc --workspace E:\your-project
```

交互命令：`/help`、`/status`、`/tasks`、`/team`、`/memory`、`/exit`。

运行状态保存在目标工作区的 `.simple_cc/`，包括 tasks、memory、mailboxes、
transcripts、outputs、cron 和 hooks 日志。所有文件工具都会校验路径不能逃出
目标工作区；危险 Shell 命令会请求 Lead 用户审批，队友不能直接读取 stdin。

## 测试

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
```

测试使用脚本化 Provider，不需要 API Key，也不会访问网络。真实 API 冒烟测试可在
配置 `.env` 后启动 CLI，要求模型创建并读取一个临时文件。

硅基流动接口文档：
[Function Calling](https://docs.siliconflow.cn/cn/userguide/guides/function-calling)。

