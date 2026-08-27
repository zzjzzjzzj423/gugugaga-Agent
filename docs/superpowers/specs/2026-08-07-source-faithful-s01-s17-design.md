# gugugaga 源码忠实拆分设计（S01–S17）

日期：2026-08-07

## 目标

以用户提供的 `pasted-text.txt` 为唯一实现基线。该附件与
`learn-claude-code/s20_comprehensive/code.py` 的 SHA-256 均为
`9EACF2F2C6F6DBE3B31117008A1A0BE44F52EE29585E5AFA0F4126D8D964D213`，
内容完全一致。

在当前 `gugugaga` 仓库的新分支上，将这份综合源码拆成清晰的小型模块，
保留 S01 Agent Loop 到 S17 Autonomous Agents 的教学实现和控制流，完整删除
S18 Worktree Isolation 与 S19 MCP Plugin，并继续通过硅基流动的 OpenAI 兼容
接口调用模型。

## 约束

- 以移动原代码块、修正 import 和收敛依赖为主，不重新发明业务流程。
- 保留当前 Git 历史；重构分支为 `refactor/source-faithful-s01-s17`。
- 不修改用户当前已改动的 `.env.example`，不追踪或修改 `.idea/`。
- 不引入 S18、S19、Web UI 或 HTTP 服务。
- 默认工作空间为当前目录，继续支持 `--workspace` 显式指定目录。
- 当前实现只作为 Git 历史保留，不作为新实现的行为基线。

## 已比较方案

### A. 源码忠实拆分（采用）

保留原函数名、参数和主要调用次序，将连续代码块移动到职责单一的模块。
硅基流动格式转换集中在 Provider 边界。该方案最容易与 S20 原文件逐段核对，
也最符合“在这个基础上进行拆分”的要求。

### B. Runtime 对象重构

把模块级状态全部封装成依赖注入对象。可测试性较强，但会改变大量函数签名、
初始化顺序和线程共享方式，源码差异过大，因此不采用。

### C. 单文件加薄转发模块

保留综合文件，仅从多个模块重新导出函数。源码差异最小，但没有真正完成模块化，
因此不采用。

## 目录和职责

```text
gugugaga/
├─ gugugaga/
│  ├─ __init__.py
│  ├─ __main__.py       # 参数解析、交互循环、退出和 Cron autorun
│  ├─ config.py         # 环境、模型、工作空间和共享常量
│  ├─ provider.py       # 硅基流动请求/响应兼容层
│  ├─ workspace.py      # safe_path、Bash、Read、Write、Edit、Glob
│  ├─ tasks.py          # Todo、Task、JSON 持久化、依赖和认领
│  ├─ skills.py         # Skill 发现、摘要和按需加载
│  ├─ prompts.py        # 动态 System Prompt 拼装
│  ├─ hooks.py          # Hook 注册、权限、日志和 Stop Hook
│  ├─ subagents.py      # 一次性 Subagent 循环
│  ├─ context.py        # Memory、预算、压缩、Transcript
│  ├─ recovery.py       # 429/529、模型降级、max_tokens、超长恢复
│  ├─ background.py     # 慢命令后台执行与结果通知
│  ├─ cron.py           # Cron 校验、持久化、调度和消费
│  ├─ teams.py          # MessageBus、协议、队友线程和自主认领
│  ├─ tools.py          # 固定工具 Schema 与 Handler 注册表
│  └─ agent.py          # prepare_context、call_llm、agent_loop
├─ tests/
├─ SOURCE_MAP.md        # S20 源代码块到新模块的逐段映射
└─ README.md
```

模块继续使用原源码的模块级状态，以避免为了架构形式改变教学逻辑。共享状态只在其
所属模块定义一次，例如任务状态归 `tasks.py`、队友状态归 `teams.py`、后台状态归
`background.py`，上层通过公开函数访问。

## Provider 边界

原源码基于 Anthropic 内容块和工具 Schema。新实现内部继续保留这种教学表示：

- 工具定义使用 `name`、`description`、`input_schema`。
- 模型输出使用带 `type`、`text`、`name`、`input`、`id` 的轻量内容块。
- Agent Loop 继续检查 `tool_use`，并追加 `tool_result`。

`provider.py` 是唯一的协议转换点：

1. 将内部工具 Schema 转为 OpenAI `type=function` Schema。
2. 将历史中的 `tool_use` 转为 assistant `tool_calls`。
3. 将 `tool_result` 转为 OpenAI `tool` 消息。
4. 调用 `SILICONFLOW_BASE_URL` 下的 Chat Completions 接口。
5. 将文本和 Function Call 转回原 Agent Loop 使用的内容块。
6. 将完成原因映射为原循环识别的 `stop_reason`，包括 `max_tokens`。

因此，Agent Loop、Subagent Loop 和 Teammate Loop 不直接依赖 OpenAI SDK 的返回
对象，也无需分别实现协议转换。

配置继续使用：

```dotenv
SILICONFLOW_API_KEY=...
SILICONFLOW_MODEL=...
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
```

## 主数据流

```text
CLI 输入或 Cron 注入
  -> agent_loop
  -> 后台结果注入与上下文预算
  -> Memory/Skill/Team 状态拼装 System Prompt
  -> Provider 转换并调用硅基流动
  -> text/tool_use 内容块
  -> Hook 与权限检查
  -> 固定 Handler 执行
  -> tool_result/后台占位结果
  -> 回到 agent_loop
```

队友沿用原源码的线程、MessageBus、计划审批、关闭协议和空闲轮询。自主队友可以查看
并认领未阻塞任务；所有队友直接使用同一个工作空间，不再有 Worktree 上下文。

## S18 Worktree 删除清单

以下内容从拆分结果中删除，而不是隐藏：

- `Task.worktree` 字段及其序列化数据。
- `WORKTREES_DIR`、名称校验、Git Worktree 命令、事件日志。
- 创建、绑定、统计、删除、保留 Worktree 的所有函数。
- Lead 的 `create_worktree`、`remove_worktree`、`keep_worktree` 工具和 Handler。
- Teammate 的 `wt_ctx`、`_wt_cwd` 及工作区包装函数。
- 自主认领时的 Worktree 切换。
- Task 列表、System Prompt、注释和文档中的 Worktree 描述。

删除后，Task 仍保留持久化、依赖、认领和完成状态；Agent Team 仍保留消息、计划、
关闭和自主认领能力。

## S19 MCP 删除清单

以下内容从拆分结果中删除，而不是保留空实现：

- `MCPClient`、`mcp_clients`、名称规范化和两个模拟服务器。
- `connect_mcp` 及动态工具发现、动态 Handler 闭包。
- `connect_mcp` 工具和 Handler。
- `connected_mcp` 上下文和 System Prompt 内容。
- `mcp__*` 权限判断。
- MCP 注释、文档和工具描述。

原 `assemble_tool_pool()` 的动态合并职责不再存在。`tools.py` 暴露固定的
S01–S17 工具表和 Handler 表，Agent Loop 每轮读取该固定注册表。

## 工作空间与持久化

CLI 默认以启动目录为 `WORKDIR`，`--workspace PATH` 在初始化运行时状态前覆盖它。
保留综合源码中的持久化概念：Task、Memory、Transcript、大型 Tool Result、Skill 和
Cron 数据均相对于 `WORKDIR`。具体路径集中在 `config.py`，避免各模块自行调用
`Path.cwd()`。

文件工具继续通过 `safe_path` 阻止逃逸工作空间。Bash 沿用原权限 Hook；后台 Bash
也经过同一 Handler 和 PostToolUse Hook。队友不读取 CLI 的标准输入。

## 错误处理

- Provider 配置缺失时在启动阶段给出明确错误。
- 未知工具由统一 Handler 调度返回可读错误，不使循环崩溃。
- 429/529、模型降级、`max_tokens` 和 prompt-too-long 沿用原恢复流程。
- 队友线程异常必须回传结果并清理 `active_teammates`。
- Cron 持久化文件损坏时沿用教学实现的容错加载策略。

## 测试与源码忠实度审计

测试分为五层：

1. Provider 转换：System Prompt、普通文本、多工具调用、Tool Result、完成原因。
2. 基础能力：工作空间文件工具、权限、Hooks、Todo、Skill、Memory 和压缩。
3. 调度能力：Task DAG、后台任务、Cron、重试和上下文恢复。
4. Team 能力：MessageBus、计划审批、关闭协议、队友执行和自主任务认领。
5. CLI：`--workspace`、普通问答、工具循环、`q`/`exit`/Ctrl+C。

额外增加静态审计：

- `SOURCE_MAP.md` 映射原 S20 的每个保留区块和顶层定义。
- 搜索源码确保不存在 `worktree`、`MCPClient`、`connect_mcp`、`mcp__`、
  `connected_mcp` 等 S18/S19 标识符。
- 固定工具集合测试确保只暴露 S01–S17 工具。
- 使用脚本 Provider 进行端到端测试，不访问真实网络。
- 最后运行完整 pytest、compileall 和 CLI help/退出烟雾测试。

## 迁移和 Git 安全

当前模块化实现会在本分支中被替换，但所有旧版本都保留在已有提交中。实现时使用
小步提交，先建立源码映射与失败测试，再拆分源码和切换 Provider，最后更新文档。
整个过程不暂存、不修改 `.env.example` 和 `.idea/`。

