# gugugaga

gugugaga 是一个面向单用户、本地 Workspace 的 Agent Runtime。它把 Main Agent、Turn 内 Subagent、长期在线的 Team Agent、Task System、Mailbox、长期记忆、上下文压缩、工具执行和 Web Console 放进同一套可观察、可恢复的运行环境。

这个项目关注的不只是“Agent 能调用多少工具”，而是把协作语义做清楚：谁负责当前工作、消息是否送达、多个 Agent 如何避免抢到同一任务、并发修改文件时如何发现冲突，以及记忆为什么被召回。

> [!IMPORTANT]
> 项目仍处于开发阶段，适合本地实验和架构验证。它不是多租户服务，也不应直接暴露到公网。

## 当前能力

| 模块 | 状态 | 说明 |
|---|---|---|
| Main Agent | 可用 | 多轮 Tool Calling、规划协调、权限控制、会话恢复和运行事件记录 |
| Web Console | 可用 | Agent Runtime Graph、Task 看板、Team Graph、Memory、Database、历史对话和本地配置 |
| Task System | 可用 | Lead 候选匹配、人工覆盖、原子领取、依赖、队列、不匹配重分配和等待原因解释 |
| Team Agent | 可用 | 长期成员、多成员并行、Mailbox 通信、停止/重启/删除、创建后编辑角色/Prompt/工具 |
| Subagent | 可用 | 当前 Turn 内并发、权限审批、取消、超时、实时事件和历史摘要 |
| Mailbox | 可用 | 普通消息唤醒空闲 Agent、claim/ack/nack、遗留 inflight 恢复和 dead-letter |
| Workspace 并发保护 | 可用 | 并行读取、SHA-256 乐观并发、FIFO 写队列、路径锁/全局锁和原子提交 |
| Context Modes | 可用 | CC、Hermes、Pi；会话开始时选择，首条消息后锁定 |
| Memory | 可用 | Semantic、Episodic、Hot/Cold Conversation Evidence、混合召回、反馈、后台整合和审计 |
| Web Search | 可选 | 配置 Tavily 后注册 `web_search` |

## 架构总览

### 1. Agent 协作架构

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontFamily":"Inter, Segoe UI, Microsoft YaHei","lineColor":"#94A3B8","clusterBkg":"#F8FAFC","clusterBorder":"#E2E8F0"}}}%%
flowchart LR
    U(["👤 用户"]) --> M["Main Agent<br/>推理 · 规划 · 协调"]

    M -->|"Turn 内并行"| S["Subagents<br/>临时执行单元"]

    M -->|"投递"| MB[("✉ Mailbox")]
    MB -->|"收取"| M
    MB -->|"唤醒 / 投递"| T["Team Agents<br/>长期协作成员"]
    T -->|"发送 / 回复"| MB

    M -->|"维护候选与理由"| TS[("Task System")]
    U -->|"人工指定"| TS
    TS -->|"候选资格 / 人工预留"| T
    T -->|"原子领取 / 完成"| TS

    M --> G["Workspace Guard"]
    S --> G
    T --> G
    G --> W[("Workspace")]

    classDef user fill:#F8FAFC,stroke:#94A3B8,color:#334155,stroke-width:1.5px;
    classDef main fill:#EEF2FF,stroke:#6366F1,color:#312E81,stroke-width:2px;
    classDef sub fill:#EFF6FF,stroke:#60A5FA,color:#1E3A8A,stroke-width:1.5px;
    classDef team fill:#ECFDF5,stroke:#34D399,color:#065F46,stroke-width:1.5px;
    classDef store fill:#FFF7ED,stroke:#FB923C,color:#9A3412,stroke-width:1.5px;
    classDef guard fill:#F8FAFC,stroke:#64748B,color:#334155,stroke-width:1.5px;

    class U user;
    class M main;
    class S sub;
    class T team;
    class MB,TS,W store;
    class G guard;
```

这张图强调四条边界：Main Agent 负责对话与协调；Subagent 是当前 Turn 的临时并行执行单元；Team Agent 是可跨 Turn 存活的长期成员；Task System 和 Mailbox 分别承载“工作归属”和“消息投递”，两者不能互相替代。所有执行单元最终都通过同一个 Workspace Guard 访问文件，避免各自实现一套不一致的并发规则。

Main Agent 在 Team System 中使用固定协议身份 `lead`。`Lead`、`Leader` 和 `main` 在通信层都会规范化为 `lead`，不会被创建成普通 Team Agent。

### 2. Main Agent 与 Memory

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontFamily":"Inter, Segoe UI, Microsoft YaHei","lineColor":"#94A3B8","clusterBkg":"#F8FAFC","clusterBorder":"#E2E8F0"}}}%%
flowchart TB
    subgraph TURN["Main Agent · Foreground Turn"]
        direction LR
        I(["用户输入"]) --> R["Memory Retrieval<br/>混合检索 · Top 10 · 预算 2000"]
        R --> C["Working<br/>Context"]
        C --> G{"Compression<br/>Gate"}
        G --> A["LLM Agent"]
        A <--> T["Tools"]
        A --> O(["回复用户"])
    end

    subgraph MEMORY["Memory · Background"]
        direction LR
        E[("Conversation<br/>Evidence")]
        MC["Consolidation<br/>与生命周期"]
        LM[("Semantic · Episodic · Evidence<br/>FTS 全部原文 · Vector 热窗口 10000")]
        E --> MC --> LM
    end

    O -->|"保存完整 Exchange"| E
    LM -.->|"下一 Turn 按需召回"| R

    classDef input fill:#F8FAFC,stroke:#94A3B8,color:#334155,stroke-width:1.5px;
    classDef process fill:#EEF2FF,stroke:#818CF8,color:#312E81,stroke-width:1.5px;
    classDef agent fill:#EDE9FE,stroke:#7C3AED,color:#4C1D95,stroke-width:2px;
    classDef memory fill:#FFF7ED,stroke:#FB923C,color:#9A3412,stroke-width:1.5px;
    classDef gate fill:#FEFCE8,stroke:#EAB308,color:#713F12,stroke-width:1.5px;

    class I,O input;
    class R,C,T,MC process;
    class A agent;
    class G gate;
    class E,LM memory;

    style TURN fill:#FAFAFF,stroke:#C7D2FE,stroke-width:1px
    style MEMORY fill:#FFFCF7,stroke:#FED7AA,stroke-width:1px
```

前台 Turn 与后台 Memory 解耦：召回只负责为当前输入补充必要证据，Compression Gate 只负责控制 Working Context 大小；回复完成后，完整 Exchange 才进入后台整合。Consolidation 或向量索引失败不会阻断主对话，后续由重试机制恢复。

### 3. Task 与文件并发安全

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontFamily":"Inter, Segoe UI, Microsoft YaHei","lineColor":"#94A3B8","clusterBkg":"#F8FAFC","clusterBorder":"#E2E8F0"}}}%%
flowchart TB
    subgraph TASKS["Task Concurrency"]
        direction LR
        TR["空闲轮询 / 模型 claim_task"] --> TL["成员锁 → 任务 RLock<br/>→ 跨进程 .state.lock"]
        TL --> TV{"状态 · 依赖 · 占用 · 开关<br/>候选资格 / 人工指定"}
        TV -->|"通过"| TO["原子领取<br/>一任务一 Owner"]
        TV -->|"冲突"| TX["拒绝领取"]
    end

    subgraph FILES["Workspace Concurrency"]
        direction LR
        FR["read_file<br/>无锁并行读取"] --> FH["SHA-256<br/>版本快照"]
        FH -.->|"expected_sha256"| FQ["FIFO Mutation Queue"]
        FW["write · edit · bash"] --> FQ
        FQ --> FL["Path Lock<br/>或 Global Lock"]
        FL --> FC{"版本匹配?"}
        FC -->|"是"| FA["临时文件 + fsync<br/>原子替换"]
        FC -->|"否"| RR["Conflict<br/>重新读取"]
        RR -.-> FR
    end

    classDef source fill:#F8FAFC,stroke:#94A3B8,color:#334155,stroke-width:1.5px;
    classDef lock fill:#EEF2FF,stroke:#6366F1,color:#312E81,stroke-width:1.5px;
    classDef decision fill:#FEFCE8,stroke:#EAB308,color:#713F12,stroke-width:1.5px;
    classDef success fill:#ECFDF5,stroke:#34D399,color:#065F46,stroke-width:1.5px;
    classDef conflict fill:#FEF2F2,stroke:#F87171,color:#991B1B,stroke-width:1.5px;

    class TR,FR,FW source;
    class TL,FH,FQ,FL lock;
    class TV,FC decision;
    class TO,FA success;
    class TX,RR conflict;

    style TASKS fill:#FAFAFF,stroke:#C7D2FE,stroke-width:1px
    style FILES fill:#F8FAFC,stroke:#CBD5E1,stroke-width:1px
```

`read_file` 本身不进入写队列，因此多个 Agent 可以并行读取。读取时可返回 SHA-256；后续写入或编辑把这个值作为 `expected_sha256` 提交。如果文件已经被其他 Agent 改动，写操作会返回 `Conflict`，调用者必须重新读取后再决定如何合并。真正的提交点使用临时文件、`fsync` 和 `os.replace`，避免读到半写文件。

Task 领取则把“读取状态—校验—写入 Owner”放在同一个临界区中。即使多个 Agent 同时看到 pending，也只会有一个通过校验；同一个 Owner 也不能同时拥有两个 `in_progress` Task。

## 三种 Agent 的能力边界

三类 Agent 不是能力从小到大的继承关系，而是三个不同生命周期和责任边界的执行角色。

| 边界 | Main Agent | Subagent | Team Agent |
|---|---|---|---|
| 生命周期 | 用户对话与 Runtime 同生共存 | 只属于当前 Main Turn，结束前必须进入终态 | 配置持久化，可跨 Turn 停止、重启和继续工作 |
| 核心职责 | 理解用户、规划、调用工具、创建/协调任务、汇总结果 | 承担一次临时且边界清晰的并行子工作 | 承担持续角色和 Task System 中的实际工作 |
| Task 所有权 | 可创建、查看和协调，但不能 claim/complete | 不拥有 Task System 任务 | 唯一可以成为 Task Owner 并完成任务的 Agent 类型 |
| 并发方式 | 单个前台 Turn 作为协调中心 | 默认最多 4 个并行执行 | 多成员并行，但每个成员同时最多一个 Task |
| Agent 间通信 | 以 `lead` 身份收取/发送 Mailbox 消息 | 结果回到创建它的 Main Turn | 通过 Mailbox 向 Lead 或其他成员发送消息 |
| Memory | 执行召回并注入 Working Context；可显式 `save_note` | 不维护独立长期记忆 | 不直接拥有独立长期记忆，依靠角色 Prompt、Task 与 Mailbox 上下文 |
| 配置能力 | 模型、上下文模式、Runtime 与 Memory 配置 | 创建时给定任务；工具集合固定 | 创建后可编辑角色、系统 Prompt 和允许工具，并安全重载 |
| Workspace 写入 | 受权限策略与 Workspace Guard 约束 | Bash/写入/编辑需要审批，受同一写协调器约束 | 只有启用对应工具后才能写入，受同一写协调器约束 |
| 适合场景 | 对话、拆解、协调、最终交付 | 当前问题内的并行检索、分析或局部实现 | 产品、前端、后端、测试等长期分工和异步协作 |

### 工具范围

Main Agent 当前注册 30 个内置工具；启用显式记忆后还会增加 `save_note`：

- Workspace：`bash`、`read_file`、`write_file`、`edit_file`、`glob`
- 规划与上下文：`todo_write`、`load_skill`、`compact`、`web_search`
- Subagent：`spawn_subagent`、`check_subagent`、`wait_subagents`、`cancel_subagent`、`review_subagent_permission`
- Task：`create_task`、`list_tasks`、`get_task`、`update_task`、`list_task_candidates_context`、`set_task_candidates`、`claim_task`、`complete_task`
- Cron：`schedule_cron`、`list_crons`、`cancel_cron`
- Team：`spawn_teammate`、`send_message`、`stop_teammate`、`restart_teammate`、`check_inbox`、`request_shutdown`、`request_plan`、`review_plan`

其中 Main Agent 的 `claim_task` 与 `complete_task` 是显式边界保护：调用时会返回错误，防止 Lead 绕过 Team 调度语义成为实际 Owner。

Subagent 的工具固定为：`bash`、`read_file`、`write_file`、`edit_file`、`glob`、`web_search`。它不能创建 Team Agent、领取 Task 或直接操作 Mailbox。

Team Agent 的六个协作核心工具始终启用且在界面中锁定：

- `send_message`
- `list_tasks`
- `get_task`
- `claim_task`
- `complete_task`
- `report_task_mismatch`

以下九个工具可以在 Team Agent 详情中自由开关：

- `read_file`
- `glob`
- `todo_write`
- `submit_plan`
- `load_skill`
- `web_search`
- `write_file`
- `edit_file`
- `bash`

默认配置在核心工具之外启用 `read_file`、`submit_plan`、`write_file` 和 `bash`。核心工具不可关闭，是因为 Team Agent 必须始终保留领取、完成和汇报任务的最低协作能力。

## Web Console

Web Console 是推荐入口。它使用本地 HTTP Server，不需要前端构建步骤。

### Agent Overview

![Agent Overview](docs/images/agent-overview.png)

Agent Overview 只显示 Main Agent 当前 Turn 的真实运行事件：输入、记忆检索、选择与注入、Working Context、Compression Gate、LLM、Tools 和回复。Team Agent 与 Subagent 有独立区域，不会误点亮主流程图。图底部同时展示 Procedural、Semantic、Episodic 和 Conversation Evidence，Consolidation 与 Vector Index 则作为独立后台状态出现。

### Task System

![Task System](docs/images/task-system.png)

任务看板按 pending、in progress 和 completed 分栏，分别显示候选成员、分配理由、人工指定、实际 Owner 与等待原因。分配列表默认显示候选，可切换“查看全部成员”指定候选外成员。手动分配只写入 Assignee，真正开始执行时仍需经过原子 claim。

### Team Agent Graph 与邮件传递

![Team Agent 发出邮件](docs/images/team-mail-outbound.png)

Team Agent Graph 把 Lead 与长期成员画成节点。消息发送时，信封图标沿发送方向在线上移动；这不是装饰性的假状态，而是 Mailbox 投递事件的可视反馈。图中 Alice 在线，其余 stopped 成员仍保留配置，可在之后重启。

![Lead 收到 Team Agent 回复](docs/images/team-mail-reply.png)

反向移动的信封表示 Team Agent 正在回复 Lead。右侧 Chat 显示同一次对话中的发送确认与回复，因此用户可以同时确认“消息已经进入 Mailbox”和“接收方已经产生响应”。普通 Mailbox 消息也会唤醒没有 Task 的空闲 Team Agent，不再要求先分配任务才能聊天。

### 创建后编辑 Team Agent

创建 Team Agent 时仍保留原有的快捷创建方式。创建完成后，可从详情页修改：

- 角色描述（成员名称作为协议身份保持不变）；
- 自定义 Prompt；
- 可选工具列表；
- 恢复为初始角色、初始 Prompt 和默认工具。

配置会持久化到 Workspace。运行中的成员保存配置后会进入安全重载流程，使新角色、Prompt 和工具集合用于后续执行；stopped 成员会在下一次启动时加载新配置。六个核心协作工具始终保留，避免编辑后产生无法领取任务或无法汇报结果的“失联 Agent”。

## Task System 与 Team Agent

### 分配流程架构

Lead 根据任务要求和成员能力维护候选范围；用户可以指定具体成员；Task System 在原子领取时确定实际 Owner。自动轮询和模型调用共用同一个领取入口。

![任务分配架构：Lead 维护候选，用户明确指定，Agent 原子领取](docs/images/task-assignment-architecture.svg)

<details>
<summary>查看可编辑的 Mermaid 架构源码</summary>

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontFamily":"Inter, Segoe UI, Microsoft YaHei","lineColor":"#97A6BB","primaryTextColor":"#17243D","clusterBkg":"#F8FAFC","clusterBorder":"#DFE6F0"},"flowchart":{"curve":"basis","nodeSpacing":30,"rankSpacing":42}}}%%
flowchart TB
    CHANGE["任务创建 / 要求变化<br/>成员创建 / 配置变化 / 不匹配退出"]
    EVENTS["任务 JSON 中的待匹配事件<br/>合并原因 · 递增 matching_revision"]
    LEAD["Lead 匹配<br/>任务要求 × 角色 / Prompt / 工具"]
    SAVE{"set_task_candidates<br/>版本仍有效且任务可重新匹配？"}
    STORE[("Task System<br/>候选 · 理由 · 人工指定 · Owner")]
    USER["用户分配界面<br/>默认候选 / 查看全部成员"]
    MANUAL["人工预留 assignee<br/>检查依赖与成员占用"]
    POLL["空闲 Agent 轮询<br/>只读候选 / 指定结果"]
    TOOL["Agent 模型调用 claim_task"]
    CLAIM["统一原子 claim<br/>重读任务并检查全部领取条件"]
    OWNER["写入 owner<br/>进入 in_progress"]
    BOARD["看板解释<br/>候选 · 理由 · 人工指定 · Owner · 等待原因"]

    CHANGE -->|"仅未领取、未人工指定的任务"| EVENTS
    EVENTS -->|"合并通知，等待 Lead 可处理"| LEAD
    LEAD --> SAVE
    SAVE -->|"通过：保存候选与理由"| STORE
    SAVE -->|"版本过期：读取最新上下文"| LEAD
    SAVE -->|"已领取或已人工指定：保留归属"| STORE
    USER --> MANUAL --> STORE
    STORE --> POLL
    POLL --> CLAIM
    TOOL --> CLAIM
    STORE -.->|"锁内重新读取"| CLAIM
    CLAIM -->|"校验通过"| OWNER
    CLAIM -->|"拒绝本次领取，显示当前状态"| BOARD
    OWNER --> STORE
    STORE --> BOARD

    classDef context fill:#F0EDF9,stroke:#E3DDF5,color:#6958B2,rx:12,ry:12;
    classDef lead fill:#7562D8,stroke:#7562D8,color:#FFFFFF,rx:14,ry:14;
    classDef store fill:#FFFFFF,stroke:#DFE6F0,color:#17243D,rx:12,ry:12;
    classDef manual fill:#EDF4FF,stroke:#D9E6FC,color:#377DDD,rx:12,ry:12;
    classDef execute fill:#EAF6F3,stroke:#D4EAE4,color:#147B6D,rx:12,ry:12;
    class CHANGE,EVENTS,SAVE context;
    class LEAD lead;
    class STORE,BOARD store;
    class USER,MANUAL manual;
    class POLL,TOOL,CLAIM,OWNER execute;
```

</details>

匹配和执行分别由事件与领取规则驱动。候选更新只保存资格，不直接启动成员；自动领取关闭时仍可更新候选，执行继续等待人工指定。没有候选、匹配尚未完成或候选都忙时，任务不会开放给其他成员。

| 数据 | 含义 | 更新方 |
|---|---|---|
| `candidate_members` | 允许自动领取的成员列表；匹配完成且为空时表示没有合适成员 | Lead，通过版本校验后保存 |
| `assignment_reason` | 根据任务要求、成员角色、Prompt 和工具做出匹配的理由 | Lead；无人适合时也必须给出理由 |
| `assignee` | 用户指定的成员，优先于候选范围；尚不代表已经开始执行 | 人工分配或现有用户队列/干预流程 |
| `owner` | 已经成功领取任务的实际执行者 | 原子 claim；完成时保留以记录归属 |
| `matching_status` | `pending` 待匹配、`matched` 已匹配、`rematch_required` 不匹配退出后待重新分配 | 任务事件、Lead 匹配与不匹配报告 |
| `matching_revision` / `matching_notified_revision` | 当前匹配版本与已确认通知版本，用于拒绝过期结果、合并和恢复通知 | Task System 与 Lead 收件箱处理 |
| `mismatch_reports` | 原成员、不匹配原因、已有工作摘要与时间 | 当前 Owner 报告，Task System 追加保存 |

### 分配与领取

Workspace 有一个 Team 自动领取总开关：

- **关闭**：用户只能把 ready Task 手动预留给一个在线、空闲的 Team Agent。
- **开启**：空闲 Team Agent 只能竞争领取自己在候选名单内、匹配已完成且依赖已完成的 pending Task。人工指定的任务只有指定成员可以领取。

旧任务缺少候选字段时保持待匹配，不默认允许所有成员领取；已有人工指定与执行归属继续保留。

Task System 是工作归属的唯一权威来源。创建 Team Agent 会触发候选重新匹配，不代表已经分配任务；关闭自动领取时，创建成员或更新候选都不会自行启动任务。

多个 Agent 同时领取时，系统按以下顺序处理：

1. 运行时先获取成员锁，再使用任务 `RLock` 和跨进程 `.tasks/.state.lock` 包住完整状态转换；
2. 在锁内重新读取 Task，而不是相信轮询时看到的旧快照；
3. 校验 Task 仍是 pending、没有 Owner、所有依赖已完成；人工指定时验证指定者，否则验证匹配状态与候选资格；
4. 校验领取者在线、可领取、没有其他 `in_progress` Task，且自动领取已开启或有人工指定；
5. 写入 Owner 和 `in_progress` 状态，再原子保存 JSON。自动领取不改写 Assignee。

因此多个 Agent 可以同时“发现”一个候选任务，但只能有一个成功成为 Owner。后到者会看到状态或 Owner 已改变并收到拒绝结果。完成任务时再次校验 Owner，其他 Agent 无法代替实际 Owner 标记完成。

人工指定可以超出候选范围，其他候选成员也不能抢走已指定任务。看板默认显示候选成员，可切换“查看全部成员”，范围外成员标注“非候选成员”；忙碌或离线成员不可选。人工指定同样受依赖、在线状态和任务占用检查约束。

运行时把“正在响应模型消息”和“已占用执行任务”分开记录。`dispatch_available` 表示成员是否可以接受任务；停止、重启、等待配置重载或持有执行任务的成员不可领取。模型主动领取与空闲轮询都要经过这个校验。

### 候选更新与不匹配退出

任务创建、未执行任务要求变化、成员创建或角色/Prompt/工具变化，以及不匹配退出，会触发 Lead 重新匹配。第一版对成员变化检查所有未领取、未被人工指定的任务，由 Lead 判断相关性。正在执行的任务保留归属，人工预留也不会被自动改派；运行中调整方向继续使用现有干预流程。

重新匹配时可以保留上次候选和理由供查看，但 `matching_status` 会使旧名单暂时失去自动领取效力，直到 Lead 保存当前版本的匹配结果。

| 触发事件 | 匹配处理 |
|---|---|
| 新任务创建 | 保存待匹配状态并通知 Lead |
| 未领取、未人工指定任务的要求变化 | 递增该任务版本，等待 Lead 重新匹配 |
| 新成员创建 | 重新评估可匹配任务，包括此前候选为空的任务 |
| 成员角色、Prompt 或工具变化 | 使可匹配任务的旧匹配结果失效，由 Lead 加入或移除成员 |
| 成员删除 | 重新评估可匹配任务，清理过期候选关系 |
| 当前 Owner 报告不匹配 | 保存原因和工作摘要，释放归属并通知 Lead |
| 成员忙闲、在线状态变化或空闲轮询 | 只重新检查能否领取，不调用 Lead 重做能力匹配 |

待匹配事件和版本随任务 JSON 持久保存，多次变化合并为一次 Lead 通知。Lead 使用 `list_task_candidates_context` 查看任务版本和成员配置，通过 `set_task_candidates` 写入名单与理由；过期版本会被拒绝，须读取新上下文。空闲轮询只读取匹配结果，不调用 Lead。匹配完成后不会因任务仍在等待而反复通知；Lead 未完成匹配时保留事件，复用收件箱失败退避后重试。

Lead 收件箱把这些持久事件合成为 `assignment_match_requested` 消息。任务内容变化发生在匹配期间时，旧版本结果不能覆盖新状态，旧通知的确认也不能清除新版本事件。仅查看通知、模型失败或未保存匹配结果，都不会把同版本的待匹配工作确认完成；重启后仍能发现尚未处理的事件。

成员可调用 `report_task_mismatch(task_id, reason, work_summary)` 报告职责或能力不匹配。任务保留原因、已有工作摘要和干预记录，清空 Owner 与人工指定，回到 pending 并标记等待重新分配。Lead 完成匹配前不能再次自动领取。普通执行失败的自动重试不属于此流程。

![任务状态流转：候选匹配、人工预留、执行完成与不匹配重新分配](docs/images/task-assignment-lifecycle.svg)

<details>
<summary>查看可编辑的 Mermaid 状态源码</summary>

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontFamily":"Inter, Segoe UI, Microsoft YaHei","primaryColor":"#FFFFFF","primaryTextColor":"#17243D","primaryBorderColor":"#DFE6F0","lineColor":"#97A6BB","tertiaryColor":"#F8FAFC"}}}%%
stateDiagram-v2
    direction LR
    state "待匹配（pending / pending）" as Matching
    state "已匹配，等待领取（pending / matched）" as Ready
    state "人工预留（pending，assignee 已指定）" as Reserved
    state "执行中（in_progress，owner 已确定）" as Running
    state "等待重新分配（pending / rematch_required）" as Rematching
    state "已完成（completed，保留 owner）" as Done

    [*] --> Matching: 创建任务
    Matching --> Ready: Lead 保存候选与理由
    Ready --> Matching: 要求或成员配置变化
    Ready --> Running: 自动领取开启且原子校验通过
    Matching --> Reserved: 用户明确指定
    Ready --> Reserved: 用户明确指定
    Reserved --> Running: 指定成员通过原子校验
    Reserved --> Matching: 一般任务取消指定
    Reserved --> Rematching: 不匹配任务取消指定
    Running --> Rematching: Owner 报告不匹配并释放归属
    Rematching --> Ready: Lead 完成重新匹配
    Rematching --> Reserved: 用户明确覆盖指定
    Running --> Done: Owner 完成任务

    classDef match fill:#F0EDF9,stroke:#DCD6F5,color:#6958B2
    classDef manual fill:#EDF4FF,stroke:#CADCF6,color:#377DDD
    classDef execute fill:#EAF6F3,stroke:#C9E3DD,color:#147B6D
    classDef rematch fill:#FCF3E5,stroke:#EDDCC1,color:#9C6927
    class Matching,Ready match
    class Reserved manual
    class Running,Done execute
    class Rematching rematch
```

</details>

图中的 `pending / matched` 分别表示任务状态和匹配状态。匹配完成可以得到空候选列表，也可能仍有未完成依赖；只有领取条件全部满足才会进入执行中。用户停止、转向及错误归属释放继续使用现有干预流程，图中仅展示候选分配的主路径。

### 看板上的等待原因

看板从任务、团队设置和成员状态生成等待解释，不触发模型匹配。候选名单、分配理由、人工指定和实际 Owner 分别展示。

人工指定任务优先检查依赖和指定成员状态；其他任务按不匹配报告、匹配状态、依赖、自动领取开关、候选可用性的顺序给出主要等待原因。自动领取条件全部通过时，`waiting_reason` 为 `null`。

| 等待原因 | 下一步条件 |
|---|---|
| 尚未确定候选成员 | Lead 完成当前版本匹配 |
| 尚无合适的候选成员 | 要求或团队能力发生变化后重新匹配，或用户手动指定 |
| 依赖未完成 | 所有依赖任务完成 |
| 候选成员都忙或不在线 | 候选成员恢复可领取状态，领取范围保持不变 |
| 报告不匹配，等待重新分配 | Lead 完成重新匹配，或用户明确覆盖指定 |
| 自动领取已关闭，等待人工指定 | 用户选择执行成员 |
| 人工指定成员忙碌或不在线 | 指定成员恢复可领取状态，或用户更改指定 |

任务要求可通过 Lead 的 `update_task` 或 `PUT /api/tasks/{id}` 更新；这些入口拒绝直接修改正在执行的任务。实现入口集中在 `tasks.py`（持久化与原子状态转换）、`teams.py`（Lead 事件、成员运行时与统一领取）、`tools.py`（模型工具）和 `web.py` / `web_assets/app.js`（分配界面与解释）。

### 生命周期

用户可以从 Web Console：

- 停止运行中的 Team Agent；
- 重启 stopped Team Agent；
- 编辑角色、Prompt 和工具；
- 删除 stopped 且没有未完成任务的 Team Agent；
- 删除 pending/completed 且没有被其他任务依赖的 Task。

运行中或 stopping 的 Team Agent 不能删除，`in_progress` Task 不能删除。删除 Agent 配置不会抹除历史审计、消息和已完成任务记录。

### Mailbox 可靠性

```text
mailbox.jsonl
    │ claim（原子改名）
    ▼
.inflight.jsonl
    ├── 成功 → ack  → 删除 inflight
    └── 失败 → nack → 放回 mailbox
```

这套协议提供“至少一次处理”基础：

- 普通消息、`result`、`error` 和 `plan_approval_request` 都能产生待处理事件；
- 空闲 Team Agent 可被普通消息唤醒并像聊天对象一样回复；
- Lead 空闲时，未读事件可以触发新的处理循环；
- 成功处理后才 ACK，失败则 NACK 回队列；
- 无法解析的行进入 dead-letter；
- 启动时恢复遗留 inflight 和旧的 `Leader/main` 别名邮箱；
- 原始 `<team-inbox>` JSON 只作为内部上下文，不显示成用户聊天气泡。

Mailbox 不保证 exactly-once。消费者仍应使用消息 ID 对可能的重复投递做幂等处理。

## Subagent

Subagent 是 Main Agent 当前 Turn 内的结构化子工作：

- 默认最多同时运行 4 个；
- 适合并行读取、搜索、分析和边界明确的局部实现；
- Bash、写入和编辑受权限审批与 Workspace Guard 约束；
- Main Turn 结束前，所有 Subagent 必须进入终态；
- 当前 Turn 展示完整事件，历史 Turn 只保留摘要；
- 不直接拥有 Task System 中的任务，也不作为长期在线成员保存。

需要跨 Turn 的稳定分工时使用 Team Agent；只需加速当前问题时使用 Subagent。

## Workspace 并发控制

Workspace Guard 使用 `WorkspaceMutationCoordinator` 统一管理 Main、Subagent 和 Team Agent 的写操作：

- 多个 `read_file` 无锁并行，不会因为别的 Agent 正在读取而排队；
- `read_file(include_hash=true)` 返回内容及 SHA-256 版本快照；
- `write_file` 与 `edit_file` 对已有文件执行 `expected_sha256` 乐观并发校验；
- 写请求进入 FIFO 队列；同一路径串行，不相交路径可以并行；
- Bash 必须声明 `write_paths`，路径明确时只锁相关文件，未声明或范围模糊时使用 Global Lock；
- 等待锁和实际写入阶段都进入运行事件，Web 可显示 waiting/writing/committed；
- 写入最终通过临时文件、`fsync` 和 `os.replace` 原子提交。

这套设计没有让 `read_file` 获取共享锁，因为读取锁会放大等待并降低多 Agent 扫描 Workspace 的吞吐量。系统采用“读取快照 + 提交前版本校验”，把冲突发现放在真正会破坏数据的写入点。

## 运行中干预

用户必须明确选择干预语义，系统不会猜测新消息与当前任务的关系：

| 动作 | 语义 |
|---|---|
| `steer` | 把补充要求注入当前执行；暂时无法注入时转为 pending message |
| `queue` | 当前任务结束后，在新 Turn 执行独立任务；Team Agent queue 会创建可见 Task |
| `redirect` | 修正当前方向；LLM 阶段可取消并重发，工具阶段等待工具结束后注入 |
| `stop` | 强制停止当前目标；用于必须终止正在运行工具的情况 |

CLI 示例：

```text
/steer main 补充输出风险清单
/queue main 完成后再生成部署文档
/redirect team:Alice 不要使用 React，改成原生 HTML
/stop team:Alice
```

## Context Modes

每个新会话可以选择一种上下文处理方式：

- `cc`：分层处理大型工具结果、旧工具结果和历史中段，必要时生成摘要。
- `hermes`：保留会话开头和近期尾部，对中间历史持续合并摘要。
- `pi`：根据 Token 预算寻找工具协议安全切点，并生成 Compaction Entry。

模式只能在空白会话开始前选择，首条消息发出后锁定。

默认计数器会根据模型名选择 Qwen、DeepSeek、GLM、OpenAI、Claude、
Llama、Mistral 或 Gemma 的估算配置；未知模型使用保守回退配置。
CC、Hermes 和 Pi 的自动触发都使用同一个近似 Token 计数器。该计数器
不等价于模型官方 tokenizer；需要精确计数时，可通过 `TokenCounterRegistry`
注册 Provider 对应的 tokenizer 实现。

执行过程中的策略对比可使用 [Context Benchmark](eval/context_bench/README.md)：
它在独立工作副本中运行相同的本地编码任务，记录逐轮验收、Token 用量、耗时、
压缩和重复操作候选，区分离线流程演练与真实模型实验。当前示例用于验证流程，
正式比较前需校准任务长度，确保实际触发要研究的压缩路径。
公开任务可使用 [Terminal-Bench 2.1 入口](eval/context_bench/TERMINAL_BENCH.md)，
通过 Harbor 在隔离容器执行，并由官方评分器验收。

## Memory 架构

默认检索配置采用当前冻结 200 题实验中表现最好的 R1 组合：Fact/Episode + 原文混合检索、现有类型配额、Top 10、召回预算 2000，BM25/Vector 各取 Top 20。原文热窗口从 30 扩大到 10000 个已整合 Exchange，覆盖当前实验全部历史原文；超过该窗口仍会转 Cold，并非无限保留向量。启用向量检索需配置 `BAAI/bge-m3`。

实验使用 `Qwen/Qwen3.6-35B-A3B`；同批新 Top 5 基线与 Top 10 的 Token F1 为 34.79 / 38.57，回答输入 Token 增加 76.96%。Top 10 的预算从 2000 提至 16000 未改变本批最终上下文，因此默认预算保持 2000。参见[冻结实验报告](eval/locomo_refined/runs/topk-answers-20260906-step4/REPORT.md)；该结果仅针对这批数据，不等同于其他模型或用户数据的保证。

启动时会将窗口内已有 Cold Evidence 恢复为 Hot，并排队补建缺失向量；后台索引完成后获得相应语义检索覆盖。显式环境变量仍可覆盖默认窗口与 Top K。

### 4. 召回、RRF 与反馈闭环

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontFamily":"Inter, Segoe UI, Microsoft YaHei","lineColor":"#94A3B8","clusterBkg":"#F8FAFC","clusterBorder":"#E2E8F0"}}}%%
flowchart TB
    Q(["用户输入"]) --> P{"Hard Pre-Gate"}
    P -->|"关闭 / 预算 0 / 空输入 / 寒暄"| SKIP["跳过召回"]
    P -->|"直接引用历史 · 规则路由"| H["Hybrid Recall"]
    P -->|"其他输入"| IG{"LLM Intent & Route Gate"}
    IG -->|"高置信度 skip"| SKIP
    IG -->|"retrieve + route<br/>异常时规则兜底"| H

    subgraph SEARCH["候选检索与证据展开"]
        C[("Active Facts · Episodes<br/>Hot 原文：最近 10000 个已整合 Exchange<br/>未整合原文也保持 Hot")]
        CE[("Cold 原文<br/>超出热窗口的已整合 Exchange")]
        B["FTS5 / BM25<br/>Top 20"]
        V["Vector · Top 20<br/>当前配置 bge-m3"]
        C --> B
        C --> V
        CE -->|"仅词法检索"| B
        B --> R["RRF 名次融合"]
        V --> R
        R --> X["按 Exchange 合并原文命中<br/>展开 user / assistant 双方消息"]
    end
    H --> B
    H --> V

    X --> RR["手写重排<br/>相关性 · 重要度 · 反馈 · 时间 · 频率"]
    RR --> D["最低分 0.20<br/>词面 / 语义去重 · subject 多样性"]
    D --> S["按路由配额选 Top 10<br/>Fact / Episode / Evidence / Mixed"]
    H -.->|"传递 route"| S
    S --> BUD["预算 2000 × 4 = 8000 字符<br/>先从末尾删整条，单条仍超限才截断"]
    BUD --> WC["untrusted_memory<br/>注入 Working Context · 最多 10 条"]
    BUD --> RI[("实际注入结果<br/>Recall Impression")]
    RI -->|"Web 👍 / 👎<br/>仅 Fact / Episode"| FB["反馈计数"]
    FB -.->|"影响后续重排"| RR

    classDef input fill:#F8FAFC,stroke:#94A3B8,color:#334155,stroke-width:1.5px;
    classDef process fill:#EEF2FF,stroke:#818CF8,color:#312E81,stroke-width:1.5px;
    classDef decision fill:#FEFCE8,stroke:#EAB308,color:#713F12,stroke-width:1.5px;
    classDef memory fill:#FFF7ED,stroke:#FB923C,color:#9A3412,stroke-width:1.5px;
    classDef success fill:#ECFDF5,stroke:#34D399,color:#065F46,stroke-width:1.5px;

    class Q,SKIP input;
    class P,IG decision;
    class H,B,V,R,X,RR,D,S,BUD process;
    class C,CE,RI memory;
    class WC,FB success;
    style SEARCH fill:#FAFAFF,stroke:#C7D2FE,stroke-width:1px
```

图中 Top 10 是预算裁剪前的条数上限，最终注入可能不足 10 条。可选 `trace` 会记录 Gate、BM25、Vector、RRF、Exchange 展开、重排、配额与预算阶段的候选、分数及移除原因，用于定位证据在哪一步丢失。

召回语料包含 active Fact、active Episode 和 Conversation Evidence。Hot Evidence 同时进入 FTS 与向量索引；Cold Evidence 仍保留在 FTS 中作为低成本原文兜底，但从向量索引移除。

Hard Pre-Gate 先排除记忆关闭、预算为 0、空输入和简单寒暄。明确引用过去信息的请求直接进入召回；其他请求经过一次 LLM Intent & Route Gate，同时返回 `retrieve/skip` 和 `fact/episode/evidence/mixed`。只有高置信度 `skip` 才会关闭检索；路由低置信度、超时、非法输出或 Provider 异常时使用确定性规则分类并 fail-open。

FTS5/BM25 与可选向量检索各取默认 Top 20。RRF 不比较两种检索器不可直接对齐的原始分数，而只融合名次。对候选项 `d`，当前实现为：

```text
RRF_raw(d) = Σᵣ 1 / (60 + rankᵣ(d))
RRF_norm(d) = min(1, RRF_raw(d) / (m / 61))
```

`r` 是包含该候选项的有效排名列表，`m` 是当前非空检索列表数量。向量模型未配置或调用失败时，`m = 1`，系统自然退化为只使用 BM25；RRF 仍然成立。

RRF 之后的当前手写重排公式为：

```text
final_score = 0.70 × relevance
            + 0.10 × importance
            + 0.08 × feedback
            + 0.07 × recency
            + 0.05 × frequency

feedback = (helpful + 1) / (helpful + irrelevant + 2)
```

最终结果还要经过最低分、词面/向量语义去重和同 subject 多样性，再根据 Gate 给出的路由分配默认 Top 10：Fact 为 `6 Fact + 2 Episode + 2 Evidence`，Episode 为 `6 Episode + 4 Evidence`，Evidence 为 `2 Fact + 2 Episode + 6 Evidence`，Mixed 为 `4 Fact + 2 Episode + 4 Evidence`。修改 Top K 时按原 Top 5 比例缩放配额。某层候选不足时从其他已通过相关性阈值的候选补位，之后按 `Token Budget × 4` 的字符上限执行预算限制：超限时从选择结果末尾整条删除，剩余单条仍超限才截断；再以 `<untrusted_memory>` 注入 Working Context。

Web 会为实际注入的结果保存 Recall Impression，记录查询、来源排名、RRF 相关性、最终分数和位置。用户只能对这次真实召回中的 active Fact/Episode 点 👍 或 👎；Conversation Evidence 不开放反馈。反馈可幂等重放，也可以从 helpful 切换为 irrelevant，计数在同一事务中增减，并影响下一次重排。这一约束可防止前端伪造任意 Memory ID 来污染训练信号。

### 5. Consolidation 与 Evidence 生命周期

**先分清三件事：原文一直保存在 `chat_log`；一次整合任务记录在 `consolidation_batches`；提取出的记忆另外写入 `facts` / `episodes`。** 整合会新增记忆、更新处理状态，不会把原文“搬走”。

![长期记忆整合分支流程图：原文落库、六轮领取、模型提取、原子提交、失败重试与崩溃回收](docs/images/memory-consolidation-flow.svg)

**图里的表分别做什么？**

| 表 | 保存什么 | 在整合中的作用 |
| --- | --- | --- |
| `chat_log` | 用户和助手的原文，每条消息一行 | 用 `turn_id` 组成 Exchange；`consolidation_status` 表示这条原文是否已被整理 |
| `consolidation_batches` | 一次领取任务的批次 ID、来源轮次、租约、尝试次数和错误 | 记录这次任务是 `processing`、`consolidated` 还是 `failed`；**没有 `pending` 状态** |
| `facts` | 稳定偏好、身份、长期目标等事实 | 保存提取出的长期语义记忆 |
| `episodes` | 带时间边界的经历、活动、决定或计划 | 保存提取出的事件记忆 |
| `memory_sources` | 记忆与来源 `turn_id` 的关联 | 从摘要追溯原文；当前关联整个来源批次，不是精确定位某一句话 |
| `memory_audit` | 保存、整合成功或失败等操作记录 | 用于排查发生了什么 |
| `memory_fts` | 原文和记忆的全文检索内容，属于 FTS5 虚拟表 | 由触发器同步维护，支持关键词检索 |
| `memory_index_outbox` | 待执行的向量新增、更新或删除任务 | 让向量索引在提交后异步更新，并单独管理重试 |
| `memory_embeddings` | 已生成的向量及模型信息 | 供向量检索使用 |

**沿主线读一遍：**

1. **原文先存，六轮再整合。** 每条消息先写入 `chat_log`；用户消息起初是 `incomplete`，助手最终回复落库后，同一 `turn_id` 的双方变成 `pending`。一轮 Exchange 是“一条用户消息 + 一条助手消息”，六轮通常对应 12 行原文。原文插入时就同步登记 FTS 和向量待办，不需要等待摘要提取。
2. **Worker 领取的是原文，顺便创建一张“任务单”。** 后台先回收过期租约，再从 `chat_log` 选最早的、重试时间已到的 6 个完整 `pending` Exchange；不足六轮就等待。在 `BEGIN IMMEDIATE` 事务内，将选中的消息改成 `processing`、绑定 `batch_id`、增加尝试次数，并创建 `processing` 批次记录和默认 600 秒租约。**领取事务提交后才调用模型**，不会持有数据库写锁等待模型返回。
3. **模型整理，程序校验入库条件。** 先遮蔽送入模型的凭据，保留数据库原文；模型输出受约束 JSON，最多 10 条 Fact、5 条 Episode。提示词要求模型判断长期价值、排除临时请求和调试状态；程序检查结构、凭据、重要度阈值，以及 Fact 的 `durability` 是否为 `long_term`。候选被过滤是正常结果；即使最终没有值得保存的记忆，也可以成功完成这个批次。
4. **成功时，内容与进度一起提交。** 再检查租约有效、原文仍归当前批次所有，在同一 SQLite 事务中写入记忆、来源和审计，由触发器维护 FTS 和 Outbox，并将 `chat_log.consolidation_status` 与 `consolidation_batches.status` 一起设为 `consolidated`。中途失败则回滚这次事务，避免出现“进度显示完成，记忆却没保存”的半成品。

**右侧两条异常分支：**

- **A · 调用失败、超时、校验或提交失败：** 旧批次变为 `failed`；仍属于该批次的原文回到 `pending`，清除批次绑定和租约，写入 `next_retry_at`。整合按 60、300、1800、7200、86400 秒退避，此后保持一天的间隔。到期后重新参与凑批，**下一次领取会创建新批次 ID**，可能与其他符合条件的对话重新组合。
- **B · Worker 崩溃：** 原文和批次暂时停在 `processing`。租约到期后，由启动或下次处理的 Worker 主动回收：旧批次记为 `failed / lease_expired`，原文回到 `pending`。租约不会自己触发回收；过期任务后来返回的结果，也不能越过提交前的租约与归属检查。

例如：第 1～6 轮被 `batch_A` 领取，随后模型超时。此时 **`batch_A` 留在批次表里，状态是 `failed`；这 6 轮原文留在 `chat_log`，状态是 `pending`**。重试时间到了，Worker 再领取符合条件的六轮，创建 `batch_B`。因此，“领取 pending 对话”查的是原文表，不是批次表。

**提交后，索引和原文生命周期继续更新。** 同一后台 Worker 后续处理 Outbox，调用当前配置的 `bge-m3` 并写入 `memory_embeddings`；向量失败只重试索引任务，不重新提取摘要，也不撤销已提交记忆。默认最多自动尝试 3 次，前两次失败分别等待 5 秒、30 秒，第 3 次失败后记为 `failed`，等待显式重试。

整合成功后、服务启动时都会校准 Evidence 生命周期：最近 10000 个已整合 Exchange 保持 Hot，更早的转为 Cold；Cold 保留原文和 FTS，删除现有向量并登记 Outbox 删除任务，重新进入热窗口时再排队建向量。未整合或未完整的原文始终保持 Hot。Web 中的 `Retry pending · N` 表示仍有等待重试或等待凑批的 pending 记录，不代表主对话已经失败。

## 快速开始

### 环境要求

- Python 3.11+
- 支持 Tool Calling 的 SiliconFlow 模型
- 可选 Tavily API Key

### 安装

```powershell
$projectDir = "C:\path\to\gugugaga"
Set-Location $projectDir

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

### 启动 Web Console

```powershell
$projectDir = "C:\path\to\gugugaga"
$workspaceDir = "C:\path\to\your-workspace"
Set-Location $projectDir

.\.venv\Scripts\python.exe -m gugugaga.web `
  --workspace $workspaceDir `
  --host 127.0.0.1 `
  --port 8765
```

打开：<http://127.0.0.1:8765>

Web 可以在未配置模型时启动。点击左下角配置按钮填写主模型、SiliconFlow API Key、可选 Consolidation 模型、Embedding 模型和 Tavily Key。配置保存在：

```text
<workspace>/.gugugaga/web_config.json
```

同一个 `host:port` 只能启动一个实例。如果出现 Windows `WinError 10048`，说明端口已被另一个进程占用，应关闭旧实例或改用其他端口。

### 使用环境变量

```powershell
$env:SILICONFLOW_API_KEY = "your-key"
$env:SILICONFLOW_MODEL = "Qwen/Qwen3.6-35B-A3B"
$env:GUGUGAGA_MEMORY_CONSOLIDATION_MODEL = "Qwen/Qwen3.6-35B-A3B"
$env:GUGUGAGA_MEMORY_INTENT_GATE_MODEL = "Qwen/Qwen3.6-35B-A3B"
$env:GUGUGAGA_MEMORY_EMBEDDING_MODEL = "BAAI/bge-m3"
$env:TAVILY_API_KEY = "tvly-your-key"

$workspaceDir = "C:\path\to\your-workspace"
.\.venv\Scripts\python.exe -m gugugaga.web --workspace $workspaceDir
```

### 启动 CLI

```powershell
$workspaceDir = "C:\path\to\your-workspace"
.\.venv\Scripts\python.exe -m gugugaga --workspace $workspaceDir
```

指定 Context Mode：

```powershell
.\.venv\Scripts\python.exe -m gugugaga --workspace $workspaceDir --context-mode hermes
.\.venv\Scripts\python.exe -m gugugaga --workspace $workspaceDir --context-mode pi
```

常用 CLI 命令：

```text
/help
/status
/tasks
/team
/memory status
/memory list
/memory search <text>
/memory show <id>
/memory update <fact_id> <new text>
/memory forget <id>
/memory feedback <id> <helpful|irrelevant>
/memory retry
/exit
```

## 配置

### Provider

| 环境变量 | 必需 | 默认值 | 说明 |
|---|---:|---|---|
| `SILICONFLOW_API_KEY` | CLI 必需 | — | SiliconFlow API Key |
| `SILICONFLOW_MODEL` | CLI 必需 | — | 主模型 |
| `SILICONFLOW_BASE_URL` | 否 | `https://api.siliconflow.cn/v1` | OpenAI 兼容地址 |
| `SILICONFLOW_FALLBACK_MODEL` | 否 | — | Provider 失败时的候选模型 |
| `TAVILY_API_KEY` | 否 | — | 配置后启用 `web_search` |

### Runtime

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `GUGUGAGA_MAX_ROUNDS` | `40` | 单 Turn 最大 Agent Loop 轮数 |
| `GUGUGAGA_MAX_TOKENS` | `8192` | 单次模型输出上限 |
| `GUGUGAGA_IDLE_POLL` | `1` | CLI 空闲轮询间隔（秒） |
| `GUGUGAGA_IDLE_TIMEOUT` | `30` | CLI 空闲等待超时（秒） |

### Memory

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `GUGUGAGA_MEMORY_ENABLED` | `true` | Memory 总开关 |
| `GUGUGAGA_MEMORY_EXPLICIT_ENABLED` | `true` | 显式记忆开关 |
| `GUGUGAGA_MEMORY_CONSOLIDATION_ENABLED` | `true` | 后台整合开关 |
| `GUGUGAGA_MEMORY_CONSOLIDATION_EXCHANGES` | `6` | 每批完整 Exchange 数量 |
| `GUGUGAGA_MEMORY_CONSOLIDATION_MODEL` | 主模型 | 整理记忆使用的模型 |
| `GUGUGAGA_MEMORY_CONSOLIDATION_TIMEOUT` | `90` | 单次整合超时（秒） |
| `GUGUGAGA_MEMORY_CONSOLIDATION_LEASE` | `600` | 整合租约（秒） |
| `GUGUGAGA_MEMORY_CONSOLIDATION_MAX_FACTS` | `10` | 单批最大 Fact 候选数 |
| `GUGUGAGA_MEMORY_CONSOLIDATION_MIN_IMPORTANCE` | `0.8` | Fact 最低重要度 |
| `GUGUGAGA_MEMORY_CONSOLIDATION_MAX_EPISODES` | `5` | 单批最大 Episode 候选数 |
| `GUGUGAGA_MEMORY_CONSOLIDATION_EPISODE_MIN_IMPORTANCE` | `0.6` | Episode 独立最低重要度 |
| `GUGUGAGA_MEMORY_EVIDENCE_HOT_EXCHANGES` | `10000` | 保留向量索引的最近已整合 Exchange 数量；更早 Evidence 仍支持词法召回 |
| `GUGUGAGA_MEMORY_RECALL_TOKENS` | `2000` | 单次召回 Token 预算 |
| `GUGUGAGA_MEMORY_INTENT_GATE_ENABLED` | `true` | 是否启用召回前 LLM Intent Gate |
| `GUGUGAGA_MEMORY_INTENT_GATE_MODEL` | 整理模型/主模型 | Intent Gate 专用模型 |
| `GUGUGAGA_MEMORY_INTENT_GATE_TIMEOUT` | `5` | Intent Gate 超时（秒） |
| `GUGUGAGA_MEMORY_EMBEDDING_MODEL` | — | 可选记忆向量模型 |
| `GUGUGAGA_MEMORY_RETRIEVAL_CANDIDATES` | `20` | 每路候选数量 |
| `GUGUGAGA_MEMORY_RETRIEVAL_TOP_K` | `10` | 最终最多注入单元数 |
| `GUGUGAGA_MEMORY_RETRIEVAL_MIN_SCORE` | `0.20` | 重排最低分阈值 |

## 本地数据

所有运行状态位于用户选择的 Workspace：

```text
<workspace>/
├── .gugugaga/
│   ├── state.db                 # Chat、Memory、Recall Impression、Consolidation、Audit
│   ├── web_config.json          # Web 本地配置和密钥
│   ├── traces/YYYY-MM-DD.jsonl  # 结构化运行事件
│   ├── usage.jsonl              # 模型和 Token 使用记录
│   ├── team-agents.json         # Team Agent 角色、Prompt、工具和生命周期配置
│   ├── team-settings.json       # Workspace Team 自动领取设置
│   ├── agent-interactions.json  # steer/queue/redirect/stop 状态
│   └── skills/                  # Runtime 与 Web 共用的 Workspace Skills
├── .tasks/                      # Task JSON 与 .state.lock
├── .mailboxes/                  # Team Agent 与 Lead Mailbox
├── .transcripts/                # 上下文模式会话记录
├── .memory/                     # 兼容 Memory 文件
├── .task_outputs/               # 大型工具结果
└── .scheduled_tasks.json        # Cron 持久化状态
```

## 测试

测试使用 Fake Provider 和确定性输入，不需要真实 API Key：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q gugugaga
.\.venv\Scripts\python.exe -m gugugaga --help
.\.venv\Scripts\python.exe -m gugugaga.web --help
```

## 下一阶段：从手写重排演进到 LambdaMART

当前阶段保留手写权重是有意的：样本量小、行为容易解释、出现问题时可以直接定位是 RRF、重要度、反馈、时间还是访问频率造成的。过早引入 Learning to Rank 容易让模型学习到稀疏反馈、位置偏差和单用户短期习惯。

当 Recall Impression 与 Web 反馈积累到足够规模后，计划使用 LambdaMART 替换 RRF 之后的手写线性重排，但不会替换 RRF 本身：

```text
BM25 ─┐
      ├─ RRF 候选融合 ─ LambdaMART 重排 ─ 去重 / Top K / Token Budget
Vector┘                    │
                           └─ 模型不可用时回退到手写重排
```

演进原则：

- **RRF 长期保留**：继续承担 BM25 与 Vector 的候选融合，不依赖训练数据；向量检索失败时仍可退化到 BM25。
- **LambdaMART 只替换重排层**：输入可以包含 RRF 分数、BM25/Vector rank、记忆类型、importance、recency、frequency、helpful/irrelevant、query-memory 相似度等特征。
- **保留模型失效兜底**：模型文件缺失、加载失败、特征版本不一致或预测异常时，立即回退到当前手写公式。
- **先去偏再训练**：训练数据按 query/turn 分组，区分“未展示”和“展示但未反馈”，处理位置偏差，避免把没有点击简单等同于 irrelevant。
- **时间切分评测**：使用按时间划分的 train/validation/test，避免同一会话或近重复记忆泄漏到不同集合。
- **离线与在线双验收**：离线观察 NDCG@K、MRR、Recall@K 和错误注入率；上线时同时监控无记忆请求的误召回、回退率和用户纠正率。
- **可解释和可回滚**：每个 Recall Impression 保存 feature schema/model version，Web 展示来源排名、最终分数与当前排序器，允许按 Workspace 一键退回手写重排。

建议满足以下条件后再启用 LambdaMART：有足够多的独立 query group、正负反馈不再极端稀疏、关键记忆类型都有覆盖，并且时间外验证稳定优于手写基线。具体阈值应由评测曲线决定，而不是仅以总反馈条数决定。

## 已知限制

- 仍是单用户、本地运行模型，不支持多租户隔离。
- Team Agent 数量没有资源配额，实际并发受本机线程、内存和模型 API 限制。
- Mailbox 提供至少一次处理，不保证 exactly-once。
- Workspace Guard 防止静默覆盖，但不会自动语义合并两个 Agent 的冲突修改。
- Bash 权限较大，生产化前仍需要更严格的进程和文件系统沙箱。
- Context 压缩、Intent Gate 和 Memory Consolidation 可能调用远程模型，应按数据隐私要求决定是否启用。
- 当前 Memory 重排权重是人工设定，不代表已经完成针对真实用户反馈的统计校准。
- 尚未完成长时间运行、进程崩溃、磁盘写满和高并发故障注入。

## 项目结构

```text
gugugaga/
├── __main__.py          # CLI、Runtime 构建、Lead inbox 循环
├── agent.py             # Main Agent Loop 与 Turn 处理
├── provider.py          # SiliconFlow Provider
├── tools.py             # Main Agent 工具定义和注册
├── workspace.py         # Workspace 工具与原子文件提交
├── mutations.py         # FIFO 层次写协调器
├── context.py           # Working Context
├── context_modes.py     # CC、Hermes、Pi
├── memory/              # Repository、Service、Retrieval、Validation
├── tasks.py             # Task System 与原子领取
├── interactions.py      # steer、queue、redirect、stop
├── subagents.py         # Turn 内 Subagent
├── teams.py             # Team Agent、Profile、Lead identity、Mailbox
├── permissions.py       # 权限策略和审批
├── stateio.py           # 原子状态写入和跨进程锁
├── observability.py     # Observer、Trace、Usage、Chat Log
├── web.py               # 本地 Web Server 和 API
├── web_config.py        # Web 配置持久化
└── web_assets/          # Web Console
```

## License

本项目采用 [PolyForm Noncommercial License 1.0.0](LICENSE) 授权。

在遵守许可证条款的前提下，可以出于个人学习、研究、实验、教学和其他非商业目的使用、复制、修改、分发及二次开发本项目。未经版权所有者另行书面授权，不得将本项目或其衍生作品用于商业用途、商业产品、收费服务或其他预期商业应用。

需要商业使用时，请通过项目仓库联系版权所有者，取得单独的商业授权。

> [!NOTE]
> 本项目属于 **Source-Available Software（源代码可用软件）**，不属于 OSI 定义的 Open Source Software。MIT、Apache-2.0、GPL 等标准开源许可证都允许商业使用，因此不适用于本项目当前的“禁止商用”目标。

## 安全说明

- 不要提交真实 API Key、`.gugugaga/web_config.json` 或 Workspace 私有数据。
- Web 写接口默认只允许回环地址调用，不要直接绑定公网地址。
- Trace 会遮蔽常见 Key、Token、Authorization 和密码字段，但这不等于完整的数据防泄漏方案。
- Memory 和远程模型调用可能包含用户对话内容，使用前应确认数据处理与留存要求。
