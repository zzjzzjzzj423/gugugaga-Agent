(() => {
  const state = {
    page: 'overview',
    memoryKind: 'procedural',
    memoryItems: [],
    table: null,
    tableView: 'rows',
    eventId: 0,
    eventQueue: [],
    eventPlayback: false,
    currentStage: null,
    currentSessionId: null,
    viewingSessionId: null,
    sessions: [],
    contextMode: 'cc',
    contextModeLocked: false,
    turnRunning: false,
    permission: null,
    teamAgents: [],
    teamMessageIds: new Set(),
    teamMessagesInitialized: false,
    teamDetailName: null,
    teamDetail: null,
    teamConfigDirty: false,
    pageScroll: { overview: 0, tasks: 0, memory: 0, database: 0 },
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  async function api(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
      cache: 'no-store',
    });
    let payload = {};
    try { payload = await response.json(); } catch (_) { payload = {}; }
    if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
    return payload;
  }

  function renderPermission() {
    const item = state.permission;
    const backdrop = $('#permission-backdrop');
    backdrop.hidden = !item;
    if (!item) return;
    const changed = backdrop.dataset.permissionId !== item.permission_id;
    backdrop.dataset.permissionId = item.permission_id;
    const source = item.source || {};
    $('#permission-source').textContent = `来源：${source.agent_type || 'main'}${source.agent_id ? ` · ${source.agent_id}` : ''}${source.turn_id ? ` · ${source.turn_id}` : ''}`;
    $('#permission-tool').textContent = item.tool || 'unknown tool';
    $('#permission-arguments').textContent = JSON.stringify(item.arguments || {}, null, 2);
    $('#permission-countdown').textContent = `剩余 ${Math.max(0, Math.ceil(Number(item.remaining_seconds || 0)))} 秒`;
    if (changed) {
      $('#permission-error').textContent = '';
      $('#permission-feedback').value = '';
    }
  }

  async function loadPermissions() {
    try {
      const payload = await api('/api/permissions');
      state.permission = (payload.items || [])[0] || null;
      renderPermission();
    } catch (_) {}
  }

  async function reviewPermission(approve) {
    if (!state.permission) return;
    const buttons = [$('#permission-allow'), $('#permission-deny')];
    buttons.forEach((button) => { button.disabled = true; });
    try {
      await api('/api/permissions/review', {
        method: 'POST',
        body: JSON.stringify({
          permission_id: state.permission.permission_id,
          approve,
          feedback: $('#permission-feedback').value,
        }),
      });
      $('#permission-feedback').value = '';
      await loadPermissions();
    } catch (error) {
      $('#permission-error').textContent = error.message;
    } finally {
      buttons.forEach((button) => { button.disabled = false; });
    }
  }

  function formatDate(value) {
    if (!value) return '—';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return new Intl.DateTimeFormat('zh-CN', {
      month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
    }).format(date);
  }

  function setPage(page) {
    const target = $(`#page-${page}`);
    if (!target) return;

    const workspace = $('.workspace');
    const selectPage = () => {
      state.pageScroll[state.page] = workspace.scrollTop;
      state.page = page;
      $$('.nav-item').forEach((button) => {
        const active = button.dataset.page === page;
        button.classList.toggle('is-active', active);
        if (active) button.setAttribute('aria-current', 'page');
        else button.removeAttribute('aria-current');
      });
      $$('.page').forEach((panel) => {
        const active = panel === target;
        panel.classList.toggle('is-active', active);
        panel.setAttribute('aria-hidden', String(!active));
      });
      workspace.scrollTop = state.pageScroll[page] || 0;
    };

    if (page !== state.page) {
      const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      if (document.startViewTransition && !reducedMotion) {
        try {
          document.startViewTransition(selectPage).finished.catch(() => {});
        } catch (_) {
          selectPage();
        }
      } else {
        selectPage();
        target.classList.remove('page-enter');
        requestAnimationFrame(() => {
          target.classList.add('page-enter');
          target.addEventListener('animationend', () => target.classList.remove('page-enter'), { once: true });
        });
      }
    }

    if (page === 'overview') loadOverview();
    if (page === 'tasks') loadTasks();
    if (page === 'memory') loadMemories();
    if (page === 'database') loadTables();
  }

  $$('.nav-item').forEach((button) => button.addEventListener('click', () => setPage(button.dataset.page)));

  const taskColumns = [
    { status: 'pending', title: '待处理', hint: '等待领取或解除依赖' },
    { status: 'in_progress', title: '进行中', hint: 'Agent 正在执行' },
    { status: 'completed', title: '已完成', hint: '任务已经交付' },
  ];

  function taskBadge(task) {
    if (task.blocked) return ['blocked', '被阻塞'];
    if (task.status === 'in_progress') return ['in-progress', '进行中'];
    if (task.status === 'completed') return ['completed', '已完成'];
    return ['ready', '可开始'];
  }

  function makeTaskCard(task) {
    const card = document.createElement('article');
    card.className = `task-card status-${task.status}${task.blocked ? ' is-blocked' : ''}`;
    const top = document.createElement('div'); top.className = 'task-card-top';
    const id = document.createElement('code'); id.textContent = task.id;
    const [badgeClass, badgeText] = taskBadge(task);
    const badge = document.createElement('span'); badge.className = `task-badge ${badgeClass}`; badge.textContent = badgeText;
    top.append(id, badge);
    const title = document.createElement('h3'); title.textContent = task.subject;
    card.append(top, title);
    if (task.description) {
      const description = document.createElement('p'); description.textContent = task.description; card.append(description);
    }
    if (task.dependencies?.length) {
      const dependencies = document.createElement('div'); dependencies.className = 'task-dependencies';
      const label = document.createElement('small'); label.textContent = '依赖'; dependencies.append(label);
      task.dependencies.forEach((dependency) => {
        const chip = document.createElement('span');
        chip.className = `dependency-chip is-${dependency.status}`;
        chip.textContent = dependency.id;
        chip.title = `${dependency.id}: ${dependency.status}`;
        dependencies.append(chip);
      });
      card.append(dependencies);
    }
    if (task.status === 'pending' && task.ready) {
      const controls = document.createElement('div'); controls.className = 'task-assignment-controls';
      if (task.assignee) {
        const assigned = document.createElement('span'); assigned.textContent = `已指派 · ${task.assignee}`;
        const cancel = document.createElement('button'); cancel.type = 'button'; cancel.textContent = '取消指派';
        cancel.addEventListener('click', async () => {
          cancel.disabled = true;
          try {
            await api(`/api/tasks/${encodeURIComponent(task.id)}/unassign`, { method: 'POST', body: '{}' });
            await loadTasks();
          } catch (error) { window.alert(error.message); }
          finally { cancel.disabled = false; }
        });
        controls.append(assigned, cancel);
      } else {
        const candidates = state.teamAgents.filter((agent) => agent.online && agent.status === 'idle' && !agent.current_task_id);
        const select = document.createElement('select'); select.setAttribute('aria-label', `为 ${task.subject} 选择 Team Agent`);
        const placeholder = document.createElement('option'); placeholder.value = ''; placeholder.textContent = candidates.length ? '选择空闲 Agent' : '没有空闲 Agent'; select.append(placeholder);
        candidates.forEach((agent) => {
          const option = document.createElement('option'); option.value = agent.name; option.textContent = `${agent.name} · ${agent.role || 'teammate'}`; select.append(option);
        });
        const assign = document.createElement('button'); assign.type = 'button'; assign.textContent = '指派'; assign.disabled = !candidates.length;
        select.addEventListener('change', () => { assign.disabled = !select.value; });
        assign.addEventListener('click', async () => {
          if (!select.value) return;
          assign.disabled = true;
          try {
            await api(`/api/tasks/${encodeURIComponent(task.id)}/assign`, { method: 'POST', body: JSON.stringify({ teammate: select.value }) });
            await loadTasks();
          } catch (error) { window.alert(error.message); }
          finally { assign.disabled = !select.value; }
        });
        controls.append(select, assign);
      }
      card.append(controls);
    }
    if (task.status === 'in_progress') {
      const activeOwner = state.teamAgents.some((agent) => agent.online && agent.name === task.owner && agent.current_task_id === task.id);
      if (!activeOwner) {
        const controls = document.createElement('div'); controls.className = 'task-assignment-controls';
        const release = document.createElement('button'); release.type = 'button'; release.textContent = '释放错误认领';
        release.addEventListener('click', async () => {
          if (!window.confirm(`确认将「${task.subject}」恢复为待处理吗？`)) return;
          release.disabled = true;
          try {
            await api(`/api/tasks/${encodeURIComponent(task.id)}/release`, { method: 'POST', body: '{}' });
            await loadTasks();
          } catch (error) { window.alert(error.message); }
          finally { release.disabled = false; }
        });
        controls.append(release); card.append(controls);
      }
    }
    const deleteControls = document.createElement('div'); deleteControls.className = 'task-delete-controls';
    const deleteButton = document.createElement('button'); deleteButton.type = 'button';
    const taskRunning = task.status === 'in_progress';
    deleteButton.textContent = taskRunning ? '运行中不可删除' : '删除任务';
    deleteButton.disabled = taskRunning;
    deleteButton.addEventListener('click', async () => {
      if (!window.confirm(`确认永久删除任务「${task.subject}」吗？历史审计事件仍会保留。`)) return;
      deleteButton.disabled = true;
      try {
        await api(`/api/tasks/${encodeURIComponent(task.id)}`, { method: 'DELETE' });
        await loadTasks();
      } catch (error) { window.alert(error.message); }
      finally { deleteButton.disabled = taskRunning; }
    });
    deleteControls.append(deleteButton); card.append(deleteControls);
    const footer = document.createElement('footer');
    const owner = document.createElement('span'); owner.textContent = task.owner ? `负责人 · ${task.owner}` : (task.assignee ? `预留给 · ${task.assignee}` : '尚未分配');
    const updated = document.createElement('time'); updated.textContent = `更新于 ${formatDate(task.updated_at)}`;
    footer.append(owner, updated); card.append(footer);
    return card;
  }

  function teamGraphNode(name) {
    return $(`.team-agent-node[data-agent-name="${CSS.escape(String(name || '').toLowerCase())}"]`);
  }

  function teamGraphPoint(node, graphRect) {
    const rect = node.getBoundingClientRect();
    return { x: rect.left - graphRect.left + (rect.width / 2), y: rect.top - graphRect.top + (rect.height / 2) };
  }

  function teamEdgeKey(from, to) {
    return [String(from).toLowerCase(), String(to).toLowerCase()].sort().join('--');
  }

  function appendTeamGraphEdge(from, to, transient = false) {
    const graph = $('#team-agent-graph');
    const svg = $('#team-graph-edges');
    const fromNode = teamGraphNode(from); const toNode = teamGraphNode(to);
    if (!graph || !svg || !fromNode || !toNode) return null;
    const graphRect = graph.getBoundingClientRect();
    const start = teamGraphPoint(fromNode, graphRect); const end = teamGraphPoint(toNode, graphRect);
    const edge = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    edge.setAttribute('x1', String(start.x)); edge.setAttribute('y1', String(start.y));
    edge.setAttribute('x2', String(end.x)); edge.setAttribute('y2', String(end.y));
    edge.setAttribute('class', `team-graph-edge${transient ? ' is-transient' : ''}`);
    edge.dataset.edgeKey = teamEdgeKey(from, to);
    svg.append(edge);
    return edge;
  }

  function drawTeamGraphEdges() {
    const graph = $('#team-agent-graph'); const svg = $('#team-graph-edges');
    if (!graph || !svg || graph.clientWidth === 0 || graph.clientHeight === 0) return;
    svg.setAttribute('viewBox', `0 0 ${graph.clientWidth} ${graph.clientHeight}`);
    svg.replaceChildren();
    state.teamAgents.forEach((agent) => appendTeamGraphEdge('lead', agent.name));
  }

  function animateTeamMessage(message) {
    const from = String(message.from_agent || message.from || '').toLowerCase();
    const to = String(message.to_agent || message.to || '').toLowerCase();
    const graph = $('#team-agent-graph'); const layer = $('#team-mail-layer');
    const fromNode = teamGraphNode(from); const toNode = teamGraphNode(to);
    if (!from || !to || !graph || !layer || !fromNode || !toNode || graph.clientWidth === 0) return;
    drawTeamGraphEdges();
    const key = teamEdgeKey(from, to);
    let edge = $(`.team-graph-edge[data-edge-key="${CSS.escape(key)}"]`, $('#team-graph-edges'));
    const transient = !edge;
    if (!edge) edge = appendTeamGraphEdge(from, to, true);
    edge?.classList.add('is-active');

    const graphRect = graph.getBoundingClientRect();
    const start = teamGraphPoint(fromNode, graphRect); const end = teamGraphPoint(toNode, graphRect);
    const mail = document.createElement('span'); mail.className = 'team-mail'; mail.textContent = '✉';
    mail.style.left = `${start.x - 14}px`; mail.style.top = `${start.y - 14}px`;
    mail.setAttribute('aria-label', `${from} 向 ${to} 发送了 ${message.message_type || message.type || '消息'}`);
    layer.append(mail); fromNode.classList.add('is-sending');
    const finish = () => {
      mail.remove(); fromNode.classList.remove('is-sending'); toNode.classList.add('is-receiving');
      edge?.classList.remove('is-active');
      if (transient) edge?.remove();
      window.setTimeout(() => toNode.classList.remove('is-receiving'), 650);
    };
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches || !mail.animate) {
      finish(); return;
    }
    const flight = mail.animate([
      { transform: 'translate3d(0, 0, 0) scale(.86)', opacity: .35 },
      { transform: `translate3d(${(end.x - start.x) * .5}px, ${(end.y - start.y) * .5 - 10}px, 0) scale(1.08)`, opacity: 1, offset: .52 },
      { transform: `translate3d(${end.x - start.x}px, ${end.y - start.y}px, 0) scale(.92)`, opacity: .5 },
    ], { duration: 920, easing: 'cubic-bezier(.22,.8,.28,1)', fill: 'forwards' });
    flight.finished.then(finish).catch(finish);
  }

  function rememberTeamMessage(message, animate = true) {
    const id = String(message.message_id || message.id || '');
    if (!id || state.teamMessageIds.has(id)) return;
    state.teamMessageIds.add(id);
    if (state.teamMessageIds.size > 500) state.teamMessageIds = new Set([...state.teamMessageIds].slice(-300));
    if (animate) animateTeamMessage(message);
  }

  function renderTeamCommunications(payload) {
    const initial = !state.teamMessagesInitialized;
    (payload.items || []).forEach((message) => rememberTeamMessage(message, !initial));
    state.teamMessagesInitialized = true;
  }

  function renderTeamAgents(payload) {
    state.teamAgents = payload.items || [];
    const online = state.teamAgents.filter((agent) => agent.online);
    $('#team-agent-count').textContent = `${online.length} online`;
    const list = $('#team-agent-list'); list.replaceChildren();
    if (!state.teamAgents.length) {
      const empty = document.createElement('div'); empty.className = 'team-empty'; empty.textContent = '尚未创建 Team Agent'; list.append(empty);
      requestAnimationFrame(drawTeamGraphEdges); return;
    }
    state.teamAgents.forEach((agent) => {
      const card = document.createElement('article'); card.className = `team-agent-item team-agent-node is-${agent.status || 'idle'}${agent.online ? ' is-online' : ''}`;
      card.dataset.agentName = String(agent.name).toLowerCase();
      card.tabIndex = 0;
      card.setAttribute('role', 'button');
      card.setAttribute('aria-label', `查看 Team Agent ${agent.name} 的工作详情`);
      const dot = document.createElement('span'); dot.className = `team-status-dot${agent.online ? ' is-online' : ''}`;
      const body = document.createElement('div');
      const name = document.createElement('strong'); name.textContent = agent.name;
      const role = document.createElement('small'); role.textContent = `${agent.role || 'teammate'} · ${agent.status || 'unknown'}`;
      body.append(name, role);
      const task = document.createElement('code'); task.textContent = agent.current_task_id || 'idle';
      const actions = document.createElement('div'); actions.className = 'team-agent-actions';
      const action = document.createElement('button'); action.type = 'button'; action.className = 'team-agent-action';
      const canStop = Boolean(agent.online) && agent.status !== 'stopping';
      action.textContent = canStop ? '关闭' : (agent.status === 'stopping' ? '关闭中' : '重新启动');
      action.disabled = agent.status === 'stopping';
      action.addEventListener('click', async (event) => {
        event.stopPropagation();
        const operation = canStop ? 'stop' : 'restart';
        const verb = canStop ? '关闭' : '重新启动';
        if (!window.confirm(`确认${verb} Team Agent「${agent.name}」吗？`)) return;
        action.disabled = true;
        try {
          await api(`/api/team/agents/${encodeURIComponent(agent.name)}/${operation}`, { method: 'POST', body: '{}' });
          await loadTasks();
        } catch (error) { window.alert(error.message); }
        finally { action.disabled = false; }
      });
      const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'team-agent-action danger'; remove.textContent = '删除';
      remove.disabled = Boolean(agent.online) || agent.status === 'stopping';
      remove.title = remove.disabled ? '运行中的 Team Agent 不能删除' : '删除 Team Agent';
      remove.addEventListener('click', async (event) => {
        event.stopPropagation();
        if (!window.confirm(`确认永久删除 Team Agent「${agent.name}」吗？运行轨迹和审计记录会保留。`)) return;
        remove.disabled = true;
        try {
          await api(`/api/team/agents/${encodeURIComponent(agent.name)}`, { method: 'DELETE' });
          if (state.teamDetailName === agent.name) closeTeamAgentDetail();
          await loadTasks();
        } catch (error) { window.alert(error.message); }
        finally { remove.disabled = Boolean(agent.online) || agent.status === 'stopping'; }
      });
      card.addEventListener('click', () => openTeamAgentDetail(agent.name));
      card.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault(); openTeamAgentDetail(agent.name);
        }
      });
      actions.append(action, remove);
      card.append(dot, body, task, actions); list.append(card);
    });
    requestAnimationFrame(drawTeamGraphEdges);
  }

  function appendTeamEmpty(container, text) {
    const empty = document.createElement('div'); empty.className = 'team-detail-empty'; empty.textContent = text; container.append(empty);
  }

  function teamEventSummary(event) {
    if (event.type === 'teammate_activity' || event.type === 'agent_phase') return event.summary || event.phase || event.status || event.type;
    if (event.type === 'tool') return `${event.tool || 'tool'} · ${event.status || '完成'}`;
    if (event.type === 'llm') return `模型请求 · ${event.status || '完成'}`;
    if (event.type === 'agent_interaction') return `${event.action || 'interaction'} · ${event.status || 'pending'}`;
    if (event.type === 'teammate_error') return event.error || 'Team Agent 执行失败';
    return event.summary || event.status || event.type;
  }

  function teamConfigurationState(configuration) {
    const states = {
      active: '当前已生效',
      pending: '任务结束后生效',
      restarting: '正在重启生效',
      next_start: '下次启动生效',
    };
    return states[configuration.apply_state] || '配置状态未知';
  }

  function renderTeamConfiguration(configuration) {
    $('#team-config-state').textContent = teamConfigurationState(configuration);
    if (state.teamConfigDirty) return;
    const editable = configuration.editable !== false;
    $('#team-config-role').value = configuration.role || '';
    $('#team-config-role').disabled = !editable;
    $('#team-config-prompt').value = configuration.prompt || '';
    $('#team-config-prompt').disabled = !editable;
    $('#team-config-save').disabled = !editable;
    $('#team-config-reset').disabled = !editable;
    const allowed = new Set(configuration.allowed_tools || []);
    const tools = $('#team-config-tools'); tools.replaceChildren();
    (configuration.tool_catalog || []).forEach((tool) => {
      const label = document.createElement('label');
      label.className = `team-config-tool${tool.required ? ' is-required' : ''}`;
      const input = document.createElement('input'); input.type = 'checkbox';
      input.dataset.toolName = tool.name;
      input.checked = tool.required || allowed.has(tool.name);
      input.disabled = Boolean(tool.required) || !editable;
      input.addEventListener('change', () => { state.teamConfigDirty = true; });
      const body = document.createElement('span');
      const name = document.createElement('strong'); name.textContent = tool.name;
      const description = document.createElement('small');
      description.textContent = `${tool.label || tool.name} · ${tool.group || '其他'}${tool.required ? ' · 必需' : ''}`;
      body.append(name, description); label.append(input, body); tools.append(label);
    });
  }

  function renderTeamAgentDetail(payload) {
    state.teamDetail = payload;
    const agent = payload.agent || {};
    const phase = agent.interaction_phase || {};
    $('#team-detail-title').textContent = agent.name || state.teamDetailName || 'Team Agent';
    $('#team-detail-meta').textContent = `${agent.role || 'teammate'} · ${agent.status || 'unknown'} · ${agent.online ? 'online' : 'offline'}`;
    $('#team-detail-phase').textContent = phase.phase || agent.activity_phase || agent.status || '—';
    $('#team-detail-task').textContent = payload.current_task ? `${payload.current_task.id} · ${payload.current_task.subject}` : '当前无执行任务';
    $('#team-detail-disclosure').textContent = payload.disclosure || '展示可审计的工作摘要，不展示模型隐藏思维链。';
    renderTeamConfiguration(payload.configuration || {});

    const activity = $('#team-detail-activity'); activity.replaceChildren();
    const activityText = phase.summary || agent.activity_summary;
    if (activityText) {
      const paragraph = document.createElement('p'); paragraph.textContent = activityText; activity.append(paragraph);
    } else appendTeamEmpty(activity, '尚无工作摘要');

    const queue = $('#team-detail-queue'); queue.replaceChildren();
    const queuedTasks = payload.queued_tasks || [];
    if (!queuedTasks.length) appendTeamEmpty(queue, '没有排队任务');
    queuedTasks.forEach((task) => {
      const item = document.createElement('article');
      const title = document.createElement('strong'); title.textContent = `${task.queue_position || '—'} · ${task.subject}`;
      const meta = document.createElement('small'); meta.textContent = `${task.id} · ${task.status}`;
      const description = document.createElement('p'); description.textContent = task.description || '无任务描述';
      item.append(title, meta, description); queue.append(item);
    });

    const interactions = $('#team-detail-interactions'); interactions.replaceChildren();
    const interactionItems = payload.interactions || [];
    if (!interactionItems.length) appendTeamEmpty(interactions, '尚无用户干预');
    interactionItems.forEach((item) => {
      const row = document.createElement('article');
      const title = document.createElement('strong'); title.textContent = `${item.action} · ${item.status}`;
      const meta = document.createElement('small'); meta.textContent = `${item.task_id || '未绑定任务'} · ${formatDate(item.created_at ? item.created_at * 1000 : null)}`;
      const content = document.createElement('p'); content.textContent = item.content || '停止当前 Agent';
      row.append(title, meta, content); interactions.append(row);
    });

    const timeline = $('#team-detail-timeline'); timeline.replaceChildren();
    const events = payload.events || [];
    if (!events.length) appendTeamEmpty(timeline, '尚无活动事件');
    events.slice().reverse().forEach((event) => {
      const row = document.createElement('details');
      const summary = document.createElement('summary'); summary.textContent = `${formatDate(event.timestamp)} · ${event.type} · ${teamEventSummary(event)}`;
      const content = document.createElement('pre'); content.textContent = JSON.stringify(event, null, 2);
      row.append(summary, content); timeline.append(row);
    });

    const lifecycle = $('#team-detail-lifecycle');
    lifecycle.textContent = agent.online ? '关闭 Agent' : '重新启动';
    lifecycle.dataset.operation = agent.online ? 'stop' : 'restart';
    lifecycle.disabled = agent.status === 'stopping';
    const remove = $('#team-detail-delete');
    remove.disabled = Boolean(agent.online) || agent.status === 'stopping';
    remove.title = remove.disabled ? '运行中的 Team Agent 不能删除' : '删除 Team Agent';
    const action = $('#team-interaction-action');
    [...action.options].forEach((option) => {
      option.disabled = !agent.online && option.value !== 'queue';
    });
    if (action.selectedOptions[0]?.disabled) action.value = 'queue';
  }

  async function loadTeamAgentDetail() {
    if (!state.teamDetailName || $('#team-detail-backdrop').hidden) return;
    try {
      const payload = await api(`/api/team/agents/${encodeURIComponent(state.teamDetailName)}`);
      renderTeamAgentDetail(payload);
    } catch (error) {
      $('#team-interaction-result').textContent = error.message;
    }
  }

  async function openTeamAgentDetail(name) {
    state.teamDetailName = name;
    state.teamDetail = null;
    state.teamConfigDirty = false;
    $('#team-detail-title').textContent = name;
    $('#team-detail-backdrop').hidden = false;
    document.body.classList.add('has-modal');
    $('#team-interaction-result').textContent = '';
    await loadTeamAgentDetail();
  }

  function closeTeamAgentDetail() {
    $('#team-detail-backdrop').hidden = true;
    document.body.classList.remove('has-modal');
    state.teamDetailName = null;
    state.teamDetail = null;
    state.teamConfigDirty = false;
  }

  function renderSubagents(current, history) {
    $('#subagent-current-turn').textContent = current.turn_id || 'no active turn';
    const live = $('#subagent-current'); live.replaceChildren();
    const events = current.events || [];
    if (!events.length && !(current.items || []).length) {
      const empty = document.createElement('div'); empty.className = 'team-empty'; empty.textContent = '当前没有 Subagent 活动'; live.append(empty);
    } else {
      (current.items || []).forEach((job) => {
        const card = document.createElement('article'); card.className = 'subagent-job';
        const title = document.createElement('strong'); title.textContent = `${job.subagent_id} · ${job.status}`;
        const description = document.createElement('p'); description.textContent = job.description || '未提供任务描述';
        card.append(title, description); live.append(card);
      });
      events.forEach((event) => {
        const row = document.createElement('details'); row.className = 'subagent-event';
        const summary = document.createElement('summary'); summary.textContent = `${formatDate(event.timestamp)} · ${event.agent_id || 'subagent'} · ${event.type}`;
        const content = document.createElement('pre'); content.textContent = JSON.stringify(event, null, 2);
        row.append(summary, content); live.append(row);
      });
    }
    const past = $('#subagent-history'); past.replaceChildren();
    const items = history.items || [];
    if (!items.length) {
      const empty = document.createElement('div'); empty.className = 'team-empty'; empty.textContent = '暂无历史 Subagent 摘要'; past.append(empty); return;
    }
    items.forEach((item) => {
      const card = document.createElement('article'); card.className = 'subagent-history-item';
      const title = document.createElement('strong'); title.textContent = `${item.subagent_id} · ${item.status}`;
      const meta = document.createElement('small'); meta.textContent = `${item.turn_id} · ${item.tool_count} tools · ${formatDate(item.started_at)}`;
      const description = document.createElement('p'); description.textContent = item.description || item.summary || '无摘要';
      card.append(title, meta, description); past.append(card);
    });
  }

  function renderTasks(data) {
    const counts = data.counts || {};
    $('#task-count-total').textContent = counts.total || 0;
    $('#task-count-progress').textContent = counts.in_progress || 0;
    $('#task-count-blocked').textContent = counts.blocked || 0;
    $('#task-count-completed').textContent = counts.completed || 0;
    $('#task-count-scheduled').textContent = counts.scheduled || 0;
    $('#task-progress-label').textContent = `整体进度 ${counts.progress || 0}%`;
    $('#task-progress-bar').style.width = `${counts.progress || 0}%`;

    const board = $('#task-board'); board.replaceChildren();
    taskColumns.forEach((column) => {
      const tasks = (data.tasks || []).filter((task) => task.status === column.status);
      const section = document.createElement('section'); section.className = `task-column column-${column.status}`;
      const heading = document.createElement('header');
      const title = document.createElement('div');
      const name = document.createElement('strong'); name.textContent = column.title;
      const hint = document.createElement('small'); hint.textContent = column.hint;
      const count = document.createElement('span'); count.textContent = tasks.length;
      title.append(name, hint); heading.append(title, count); section.append(heading);
      const list = document.createElement('div'); list.className = 'task-column-list';
      if (!tasks.length) {
        const empty = document.createElement('div'); empty.className = 'task-column-empty'; empty.textContent = '暂无任务'; list.append(empty);
      } else tasks.forEach((task) => list.append(makeTaskCard(task)));
      section.append(list); board.append(section);
    });

    const scheduled = $('#scheduled-list'); scheduled.replaceChildren();
    const jobs = data.scheduled_tasks || [];
    $('#scheduled-count-label').textContent = `${jobs.length} schedules`;
    if (!jobs.length) {
      const empty = document.createElement('div'); empty.className = 'scheduled-empty'; empty.textContent = '暂无定时任务'; scheduled.append(empty);
      return;
    }
    jobs.forEach((job) => {
      const row = document.createElement('article'); row.className = 'scheduled-task';
      const clock = document.createElement('span'); clock.className = 'scheduled-clock'; clock.textContent = '◷';
      const body = document.createElement('div'); body.className = 'scheduled-body';
      const top = document.createElement('div');
      const name = document.createElement('strong'); name.textContent = job.prompt || '未命名定时任务';
      const id = document.createElement('code'); id.textContent = job.id;
      top.append(name, id);
      const meta = document.createElement('div'); meta.className = 'scheduled-meta';
      const cron = document.createElement('code'); cron.textContent = job.cron;
      const recurring = document.createElement('span'); recurring.textContent = job.recurring ? '循环' : '单次';
      const durable = document.createElement('span'); durable.textContent = job.durable ? '持久化' : '本次会话';
      meta.append(cron, recurring, durable); body.append(top, meta);
      const next = document.createElement('div'); next.className = 'scheduled-next';
      const label = document.createElement('small'); label.textContent = '下次运行';
      const time = document.createElement('strong'); time.textContent = job.next_run ? formatDate(job.next_run) : '无可用时间';
      next.append(label, time); row.append(clock, body, next); scheduled.append(row);
    });
  }

  async function loadTasks() {
    const button = $('#task-refresh');
    button.disabled = true;
    try {
      const session = state.currentSessionId ? `?session_id=${encodeURIComponent(state.currentSessionId)}` : '';
      const [tasks, settings, agents, communications, current, history] = await Promise.all([
        api('/api/tasks'), api('/api/team/settings'), api('/api/team/agents'),
        api('/api/team/communications'),
        api('/api/subagents'), api(`/api/subagents/history${session}`),
      ]);
      renderTeamAgents(agents);
      renderTeamCommunications(communications);
      renderTasks(tasks);
      $('#team-auto-claim').checked = Boolean(settings.auto_claim_enabled);
      $('#team-auto-state').textContent = settings.auto_claim_enabled ? '已开启' : '已关闭';
      renderSubagents(current, history);
    } catch (error) {
      const board = $('#task-board'); board.replaceChildren();
      const failed = document.createElement('div'); failed.className = 'task-load-error'; failed.textContent = `任务数据加载失败：${error.message}`; board.append(failed);
    } finally { button.disabled = false; }
  }

  async function loadTeamCommunications() {
    try { renderTeamCommunications(await api('/api/team/communications')); } catch (_) { /* best-effort animation */ }
  }

  $('#task-refresh').addEventListener('click', loadTasks);
  $('#team-auto-claim').addEventListener('change', async (event) => {
    const input = event.currentTarget; input.disabled = true;
    try {
      const settings = await api('/api/team/settings', { method: 'PUT', body: JSON.stringify({ auto_claim_enabled: input.checked }) });
      input.checked = Boolean(settings.auto_claim_enabled);
      $('#team-auto-state').textContent = settings.auto_claim_enabled ? '已开启' : '已关闭';
    } catch (error) {
      input.checked = !input.checked; window.alert(error.message);
    } finally { input.disabled = false; }
  });

  async function loadStatus() {
    try {
      const status = await api('/api/status');
      state.eventId = Math.max(state.eventId, Number(status.event_id || 0));
      if (status.session_id) {
        state.currentSessionId = status.session_id;
        if (!state.viewingSessionId) state.viewingSessionId = status.session_id;
      }
      if (status.context_mode) state.contextMode = status.context_mode;
      state.contextModeLocked = Boolean(status.context_mode_locked);
      $('#config-banner').hidden = status.chat_configured;
      $('#search-config-dot').classList.toggle('is-ready', Boolean(status.web_search_configured));
      $('#settings-button').title = status.web_search_configured ? '配置 · Tavily 已连接' : '配置 · Tavily 未连接';
      $('#sidebar-status').textContent = status.runtime_ready ? 'Agent online' : 'Console online';
      updateContextModeControl();
    } catch (error) {
      $('#sidebar-status').textContent = 'Backend offline';
      $('#config-banner').hidden = false;
      $('#config-banner').textContent = error.message;
    }
  }

  function setSettingsOpen(open) {
    $('#settings-backdrop').hidden = !open;
    if (open) $('#settings-model').focus();
  }

  function renderConfiguration(data) {
    $('#settings-model').value = data.model || '';
    $('#settings-small-model').value = data.consolidation_model || '';
    $('#settings-intent-model').value = data.intent_gate_model || '';
    $('#settings-embedding-model').value = data.embedding_model || '';
    $('#settings-siliconflow-key').value = '';
    $('#settings-tavily-key').value = '';
    $('#settings-siliconflow-key').placeholder = data.siliconflow_api_key_configured
      ? `${data.siliconflow_api_key_hint || '已配置'} · 留空保持不变`
      : '输入 SiliconFlow Key';
    $('#settings-tavily-key').placeholder = data.tavily_api_key_configured
      ? `${data.tavily_api_key_hint || '已配置'} · 留空保持不变`
      : 'tvly-…';
    $('#siliconflow-key-state').textContent = data.siliconflow_api_key_configured
      ? `已配置 ${data.siliconflow_api_key_hint || ''}`
      : '尚未配置';
    $('#tavily-key-state').textContent = data.tavily_api_key_configured
      ? `已配置 ${data.tavily_api_key_hint || ''} · web_search 可用`
      : '尚未配置 · web_search 不可用';
  }

  async function openSettings() {
    if (state.turnRunning) return;
    const result = $('#settings-result');
    result.textContent = '正在读取配置…'; result.classList.remove('is-error');
    setSettingsOpen(true);
    try {
      const data = await api('/api/config');
      renderConfiguration(data);
      result.textContent = '';
    } catch (error) {
      result.textContent = error.message; result.classList.add('is-error');
    }
  }

  $('#settings-button').addEventListener('click', openSettings);
  $('#settings-close').addEventListener('click', () => setSettingsOpen(false));
  $('#settings-cancel').addEventListener('click', () => setSettingsOpen(false));
  $('#settings-backdrop').addEventListener('click', (event) => {
    if (event.target === $('#settings-backdrop')) setSettingsOpen(false);
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !$('#settings-backdrop').hidden) setSettingsOpen(false);
  });
  $('#settings-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (state.turnRunning) return;
    const save = $('#settings-save');
    const result = $('#settings-result');
    save.disabled = true; result.classList.remove('is-error'); result.textContent = '正在保存并重载 Agent…';
    try {
      const data = await api('/api/config', {
        method: 'POST',
        body: JSON.stringify({
          model: $('#settings-model').value.trim(),
          consolidation_model: $('#settings-small-model').value.trim(),
          intent_gate_model: $('#settings-intent-model').value.trim(),
          embedding_model: $('#settings-embedding-model').value.trim(),
          siliconflow_api_key: $('#settings-siliconflow-key').value.trim(),
          tavily_api_key: $('#settings-tavily-key').value.trim(),
        }),
      });
      renderConfiguration(data);
      if (data.reload_error) throw new Error(`配置已保存，但 Agent 重载失败：${data.reload_error}`);
      result.textContent = data.runtime_reloaded ? '配置已保存，Agent 已重载。' : '配置已保存，将在首次对话时生效。';
      await Promise.all([loadStatus(), loadOverview(), loadSessions()]);
      setTimeout(() => setSettingsOpen(false), 500);
    } catch (error) {
      result.textContent = error.message; result.classList.add('is-error');
    } finally {
      save.disabled = false;
    }
  });

  const backgroundStateLabels = {
    idle: 'Idle', running: 'Running', failed: 'Failed',
    retrying: 'Retry pending',
    disabled: 'Disabled', indexing: 'Indexing', synced: 'Synced', failed: 'Failed',
  };

  function setBackgroundState(kind, value) {
    const id = kind === 'consolidation' ? '#consolidation-state' : '#vector-index-state';
    const target = $(id);
    if (!target) return;
    const normalized = String(value || (kind === 'consolidation' ? 'idle' : 'disabled')).toLowerCase();
    target.textContent = backgroundStateLabels[normalized] || normalized;
    const container = target.parentElement;
    container?.classList.toggle('is-active', normalized === 'running' || normalized === 'indexing' || normalized === 'retrying');
    container?.classList.toggle('is-error', normalized === 'failed');
    container?.classList.toggle('is-disabled', normalized === 'disabled');
  }

  async function loadOverview(sessionId = state.viewingSessionId || state.currentSessionId) {
    try {
      const query = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
      const data = await api(`/api/overview${query}`);
      $('#metric-turn').textContent = `#${data.turn_count || 0}`;
      $('#metric-turn-time').textContent = data.last_turn_at ? `更新于 ${formatDate(data.last_turn_at)}` : '尚无记录';
      $('#metric-latency').textContent = Number.isFinite(data.last_latency_ms) ? `${(data.last_latency_ms / 1000).toFixed(1)}s` : '—';
      $('#metric-memory').textContent = data.memory_hits || 0;
      $('#metric-memory-total').textContent = `Fact ${Number(data.memory?.facts || 0)} · Episode ${Number(data.memory?.episodes || 0)} · Evidence Hot ${Number(data.memory?.evidence_hot || 0)} / Cold ${Number(data.memory?.evidence_cold || 0)}`;
      const consolidationState = String(data.memory?.consolidation_state || 'idle').toLowerCase();
      setBackgroundState('consolidation', consolidationState);
      const consolidationTarget = $('#consolidation-state');
      const consolidationPending = Number(data.memory?.pending || 0);
      const consolidationFailure = data.memory?.last_failure || {};
      if (consolidationTarget && consolidationState === 'retrying') {
        consolidationTarget.textContent = `Retry pending · ${consolidationPending}`;
      }
      if (consolidationTarget) {
        const error = consolidationFailure.error_code || '';
        const attempts = Number(consolidationFailure.attempt_count || 0);
        consolidationTarget.parentElement.title = error
          ? `最近错误：${error} · 尝试 ${attempts} 次 · ${consolidationPending} 轮等待重试`
          : `${consolidationPending} 轮等待整理`;
      }
      setBackgroundState('vector', data.memory?.vector_state);
      const jobs = data.memory?.index_jobs || {};
      const vectorState = $('#vector-index-state');
      if (vectorState) vectorState.parentElement.title = `${Number(data.memory?.indexed || 0)} indexed · ${Number(jobs.pending || 0)} pending · ${Number(jobs.failed || 0)} failed`;
      const retrieval = data.retrieval || {};
      const strategy = retrieval.strategy && retrieval.strategy !== 'none'
        ? retrieval.strategy
        : retrieval.reason === 'intent_gate_skip' ? 'intent skip' : '—';
      $('#retrieval-strategy').textContent = `Retrieval · ${strategy}`;
      $('#metric-context').textContent = Number.isFinite(data.context_ratio) ? `${data.context_ratio}%` : '—';
      $('#metric-context-mode').textContent = `${data.context?.display_name || 'CC'} mode · ${data.context?.successful_compactions || 0} 次压缩`;
      if (data.context?.mode) state.contextMode = data.context.mode;
      state.contextModeLocked = Boolean(data.context?.locked);
      updateContextModeControl();
      $('#sidebar-model').textContent = data.model || 'not configured';
      $('#chat-model').textContent = data.model || 'Agent';
    } catch (error) {
      $('#runtime-label').textContent = error.message;
    }
  }

  const stageOrder = ['input', 'retrieval_gate', 'memory_injection', 'working_context', 'compression_gate', 'compression', 'agent', 'tools', 'reply'];
  const stageLabels = {
    input: '输入入口', retrieval_gate: '记忆检索', memory_injection: '选择与注入',
    working_context: 'Working Context', compression_gate: 'Compression Gate',
    compression: 'Context Compression', agent: 'LLM Agent', tools: 'Tools',
    reply: '回复用户', consolidation: 'Memory Consolidation', error: '运行失败',
  };
  const stageHoldMs = 320;
  const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

  function resetRuntimeGraph() {
    $$('[data-stage]').forEach((node) => node.classList.remove('is-active', 'is-complete', 'is-error', 'is-skipped'));
    $$('[data-memory-stage]').forEach((node) => node.classList.remove('is-active'));
    state.currentStage = null;
  }

  function activateMemoryKinds(kinds = []) {
    kinds.forEach((kind) => $(`[data-memory-stage="${kind}"]`)?.classList.add('is-active'));
  }

  function activateStage(stage, status = 'active') {
    const nodes = $$('[data-stage]');
    if (stage === 'input') resetRuntimeGraph();
    const previous = state.currentStage ? $(`[data-stage="${state.currentStage}"]`) : null;
    if (previous && state.currentStage !== stage) {
      previous.classList.remove('is-active');
      if (!previous.classList.contains('is-error') && !previous.classList.contains('is-skipped')) previous.classList.add('is-complete');
    }
    const active = $(`[data-stage="${stage}"]`);
    if (active) {
      active.classList.toggle('is-skipped', status === 'skipped');
      active.classList.toggle('is-active', status !== 'error' && status !== 'skipped');
      active.classList.toggle('is-error', status === 'error');
    }
    state.currentStage = stage;
    $('#runtime-label').textContent = status === 'error' ? '运行失败' : `正在执行 · ${stageLabels[stage] || stage}`;
    $('#event-chip').textContent = stageLabels[stage] || stage;
    $('#typing-label').textContent = stageLabels[stage] || 'Agent 正在工作';
    if (stage === 'reply' && status === 'complete') {
      setTimeout(() => {
        nodes.forEach((node) => node.classList.remove('is-active'));
        state.currentStage = null;
        $('#runtime-label').textContent = '等待输入';
        $('#event-chip').textContent = 'idle';
      }, 900);
    }
  }

  function normalizeEvent(event) {
    // Agent Overview visualizes the Lead only. Child-agent events remain
    // available to their task/detail panels but never drive this graph.
    if (event.agent_type && event.agent_type !== 'main') return null;
    let stage = event.stage;
    if (!stage) {
      if (event.type === 'turn_start') return null;
      else if (event.type === 'llm') {
        if (event.call_type === 'context_summary' || String(event.call_type || '').startsWith('memory_')) return null;
        stage = 'agent';
      }
      else if (event.type === 'tool') stage = 'tools';
      else if (event.type === 'context') {
        if (event.status === 'success' && event.result_code === 'SUCCESS') stage = 'compression';
        else if (event.status === 'skipped') return { action: 'skip-compression' };
        else if (event.status === 'failed') return { stage: 'compression', status: 'error' };
        else return null;
      }
      else if (event.type === 'memory') {
        if (event.action === 'intent_gate' && event.status === 'active') stage = 'retrieval_gate';
        else if (event.action === 'consolidate') {
          const next = event.status === 'active' ? 'running' : event.status === 'failed' ? 'retrying' : 'idle';
          return { action: 'background-state', kind: 'consolidation', value: next };
        }
        else if (event.action === 'index_outbox') {
          const next = event.status === 'active' ? 'indexing' : event.status === 'failed' ? 'failed' : 'synced';
          return { action: 'background-state', kind: 'vector', value: next };
        }
        else if (event.action === 'recall' && event.status === 'hit') return { action: 'memory-hit', memoryKinds: event.kinds || [] };
        else return null;
      }
      else if (event.type === 'turn_end') stage = 'reply';
      else if (event.type === 'turn_error') stage = 'error';
    }
    if (stage === 'error') return { stage: 'agent', status: 'error' };
    return stage ? { stage, status: event.status, memoryKinds: event.memory_kinds || [], strategy: event.strategy } : null;
  }

  async function playEventQueue() {
    if (state.eventPlayback) return;
    state.eventPlayback = true;
    try {
      while (state.eventQueue.length) {
        const item = state.eventQueue.shift();
        if (item.action === 'skip-compression') {
          const compression = $('[data-stage="compression"]');
          compression?.classList.remove('is-active', 'is-complete', 'is-error');
          compression?.classList.add('is-skipped');
          continue;
        }
        if (item.action === 'memory-hit') {
          activateMemoryKinds(item.memoryKinds);
          continue;
        }
        if (item.action === 'background-state') {
          setBackgroundState(item.kind, item.value);
          if (item.value !== 'running' && item.value !== 'indexing') loadOverview();
          continue;
        }
        if (!item.stage) continue;
        if (item.stage === 'retrieval_gate' && item.strategy && item.strategy !== 'none') {
          $('#retrieval-strategy').textContent = `Retrieval · ${item.strategy}`;
        }
        activateMemoryKinds(item.memoryKinds);
        if (item.stage === state.currentStage && item.status !== 'error') continue;
        activateStage(item.stage, item.status);
        await sleep(item.stage === 'reply' ? 400 : stageHoldMs);
      }
    } finally {
      state.eventPlayback = false;
      if (state.eventQueue.length) playEventQueue();
    }
  }

  function enqueueEvent(event) {
    if (event.type === 'permission_requested' || event.type === 'permission_resolved') {
      loadPermissions();
    }
    if (event.type === 'team_inbox_unread') {
      $('#event-chip').textContent = 'Team 消息未读';
      if (state.page === 'tasks') loadTasks();
    }
    if (event.type === 'team_message') {
      rememberTeamMessage(event);
      if (state.page === 'tasks') $('#event-chip').textContent = `${event.from_agent || 'Agent'} → ${event.to_agent || 'Agent'}`;
    }
    if (event.type === 'lead_inbox_reply') {
      if (!event.session_id || event.session_id === state.currentSessionId) {
        Promise.all([loadHistory(), loadSessions(), loadOverview()]);
      }
    }
    const item = normalizeEvent(event);
    if (!item) return;
    const previous = state.eventQueue[state.eventQueue.length - 1];
    if (previous && item.stage && previous.stage === item.stage && previous.status === item.status) return;
    state.eventQueue.push(item);
    playEventQueue();
  }

  async function pollEvents() {
    while (true) {
      try {
        const payload = await api(`/api/events?after=${state.eventId}&timeout=20`);
        for (const event of payload.items || []) {
          state.eventId = Math.max(state.eventId, Number(event.event_id || 0));
          enqueueEvent(event);
        }
      } catch (_) {
        await new Promise((resolve) => setTimeout(resolve, 1200));
      }
    }
  }

  function memoryLabel(kind) {
    return { procedural: '程序性记忆', semantic: '语义记忆', episodic: '情景记忆' }[kind] || kind;
  }

  function memoryInitial(kind) {
    return { procedural: 'P', semantic: 'S', episodic: 'E' }[kind] || 'M';
  }

  function memoryColor(kind) {
    return { procedural: 'purple', semantic: 'teal', episodic: 'amber' }[kind] || 'purple';
  }

  function selectMemory(item, button) {
    $$('.memory-item').forEach((value) => value.classList.toggle('is-active', value === button));
    const detail = $('#memory-detail');
    detail.className = 'memory-detail';
    detail.replaceChildren();
    const title = document.createElement('h3');
    title.textContent = item.subject || memoryLabel(item.kind);
    const text = document.createElement('p');
    text.textContent = item.text || '';
    detail.append(title, text, divider());
    const fields = [
      ['Memory ID', item.id], ['类型', memoryLabel(item.kind)], ['状态', item.status || 'active'],
      ['来源', item.source || '—'], ['发生时间', formatDate(item.occurred_at)],
    ];
    fields.forEach(([label, value]) => {
      const row = document.createElement('div');
      row.className = 'detail-row';
      const key = document.createElement('span'); key.textContent = label;
      const content = document.createElement(label === 'Memory ID' ? 'code' : 'strong'); content.textContent = value || '—';
      row.append(key, content); detail.append(row);
    });
  }

  function divider() {
    const element = document.createElement('div');
    element.className = 'detail-divider';
    return element;
  }

  function renderMemories(data) {
    state.memoryItems = data.items || [];
    $('#count-procedural').textContent = data.counts?.procedural || 0;
    $('#count-semantic').textContent = data.counts?.semantic || 0;
    $('#count-episodic').textContent = data.counts?.episodic || 0;
    $('#memory-list-title').textContent = memoryLabel(state.memoryKind);
    $('#memory-result-count').textContent = `${state.memoryItems.length} items`;
    const list = $('#memory-list');
    list.replaceChildren();
    if (!state.memoryItems.length) {
      const empty = document.createElement('div'); empty.className = 'memory-detail empty-state'; empty.textContent = '没有匹配的记忆'; list.append(empty);
      $('#memory-detail').className = 'memory-detail empty-state'; $('#memory-detail').textContent = '选择一条记忆'; return;
    }
    state.memoryItems.forEach((item, index) => {
      const button = document.createElement('button'); button.type = 'button'; button.className = `memory-item${index === 0 ? ' is-active' : ''}`;
      const icon = document.createElement('span'); icon.className = `memory-item-icon ${memoryColor(item.kind)}`; icon.textContent = memoryInitial(item.kind);
      const body = document.createElement('span');
      const title = document.createElement('strong'); title.textContent = item.subject || item.text;
      const meta = document.createElement('small'); meta.textContent = `${item.source || 'memory'} · ${item.status || 'active'}`;
      body.append(title, meta);
      const time = document.createElement('time'); time.textContent = formatDate(item.occurred_at);
      button.append(icon, body, time); button.addEventListener('click', () => selectMemory(item, button)); list.append(button);
    });
    selectMemory(state.memoryItems[0], $('.memory-item'));
  }

  async function loadMemories() {
    const query = $('#memory-search').value.trim();
    try {
      const data = await api(`/api/memories?kind=${encodeURIComponent(state.memoryKind)}&q=${encodeURIComponent(query)}`);
      renderMemories(data);
    } catch (error) {
      $('#memory-list').textContent = error.message;
    }
  }

  $$('.memory-type').forEach((button) => button.addEventListener('click', () => {
    state.memoryKind = button.dataset.memoryKind;
    $$('.memory-type').forEach((item) => {
      const active = item === button; item.classList.toggle('is-active', active); item.setAttribute('aria-selected', String(active));
    });
    loadMemories();
  }));

  let searchTimer;
  $('#memory-search').addEventListener('input', () => {
    clearTimeout(searchTimer); searchTimer = setTimeout(loadMemories, 250);
  });

  async function loadTables() {
    try {
      const data = await api('/api/database/tables');
      const items = data.items || [];
      $('#table-count').textContent = `${items.length} tables`;
      const list = $('#table-list'); list.replaceChildren();
      items.forEach((item, index) => {
        const button = document.createElement('button'); button.type = 'button'; button.className = `table-button${item.name === state.table || (!state.table && index === 0) ? ' is-active' : ''}`; button.dataset.table = item.name;
        const name = document.createElement('span'); name.textContent = item.name;
        const count = document.createElement('span'); count.textContent = item.row_count;
        button.append(name, count); button.addEventListener('click', () => selectTable(item.name)); list.append(button);
      });
      if (!state.table && items.length) state.table = items[0].name;
      if (state.table) loadTableView();
    } catch (error) {
      $('#table-list').textContent = error.message;
    }
  }

  function selectTable(name) {
    state.table = name;
    $$('.table-button').forEach((button) => button.classList.toggle('is-active', button.dataset.table === name));
    loadTableView();
  }

  function queryForView() {
    if (state.tableView === 'schema') return `PRAGMA table_info(${state.table});`;
    if (state.tableView === 'indexes') return `PRAGMA index_list(${state.table});`;
    return `SELECT * FROM ${state.table} ORDER BY rowid DESC LIMIT 50;`;
  }

  async function loadTableView() {
    if (!state.table) return;
    try {
      const data = await api(`/api/database/table?name=${encodeURIComponent(state.table)}&view=${state.tableView}&limit=50`);
      $('#selected-table').textContent = state.table;
      $('#selected-table-meta').textContent = `${data.columns.length} columns · ${state.tableView}`;
      $('#query-text').textContent = queryForView();
      $('#row-count').textContent = `${data.rows.length} rows`;
      const head = $('#data-head'); const body = $('#data-body'); head.replaceChildren(); body.replaceChildren();
      const headerRow = document.createElement('tr');
      data.columns.forEach((column) => { const cell = document.createElement('th'); cell.scope = 'col'; cell.textContent = column; headerRow.append(cell); });
      head.append(headerRow);
      data.rows.forEach((row) => {
        const tableRow = document.createElement('tr');
        data.columns.forEach((column) => {
          const cell = document.createElement('td'); const value = row[column];
          if (value === null || value === undefined) { const span = document.createElement('span'); span.className = 'null-value'; span.textContent = 'NULL'; cell.append(span); }
          else if (Array.isArray(value)) cell.textContent = value.join(', ');
          else if (typeof value === 'object') cell.textContent = JSON.stringify(value);
          else cell.textContent = String(value);
          tableRow.append(cell);
        });
        body.append(tableRow);
      });
    } catch (error) {
      $('#data-body').textContent = error.message;
    }
  }

  $$('.view-tabs button').forEach((button) => button.addEventListener('click', () => {
    state.tableView = button.dataset.dbView;
    $$('.view-tabs button').forEach((item) => {
      const active = item === button; item.classList.toggle('is-active', active); item.setAttribute('aria-selected', String(active));
    });
    loadTableView();
  }));

  const inlineRules = [
    { type: 'code', expression: /`([^`\n]+)`/g },
    { type: 'strong', expression: /\*\*([^\n]+?)\*\*/g },
    { type: 'strong', expression: /__([^\n]+?)__/g },
    { type: 'strike', expression: /~~([^\n]+?)~~/g },
    { type: 'link', expression: /\[([^\]\n]+)\]\(([^)\s]+)\)/g },
    { type: 'emphasis', expression: /\*([^*\n]+?)\*/g },
    { type: 'emphasis', expression: /_([^_\n]+?)_/g },
  ];

  function nextInlineToken(text, offset) {
    let selected = null;
    inlineRules.forEach((rule, priority) => {
      rule.expression.lastIndex = offset;
      const match = rule.expression.exec(text);
      if (!match) return;
      if (!selected || match.index < selected.match.index || (match.index === selected.match.index && priority < selected.priority)) {
        selected = { rule, match, priority };
      }
    });
    return selected;
  }

  function safeExternalUrl(value) {
    try {
      const url = new URL(value);
      return ['http:', 'https:'].includes(url.protocol) ? url.href : null;
    } catch (_) {
      return null;
    }
  }

  function appendInlineMarkdown(parent, source) {
    const text = String(source || '');
    let offset = 0;
    while (offset < text.length) {
      const token = nextInlineToken(text, offset);
      if (!token) {
        parent.append(document.createTextNode(text.slice(offset)));
        break;
      }
      if (token.match.index > offset) parent.append(document.createTextNode(text.slice(offset, token.match.index)));

      let element = null;
      if (token.rule.type === 'code') {
        element = document.createElement('code');
        element.textContent = token.match[1];
      } else if (token.rule.type === 'strong') {
        element = document.createElement('strong');
        appendInlineMarkdown(element, token.match[1]);
      } else if (token.rule.type === 'strike') {
        element = document.createElement('del');
        appendInlineMarkdown(element, token.match[1]);
      } else if (token.rule.type === 'emphasis') {
        element = document.createElement('em');
        appendInlineMarkdown(element, token.match[1]);
      } else if (token.rule.type === 'link') {
        const href = safeExternalUrl(token.match[2]);
        if (href) {
          element = document.createElement('a');
          element.href = href;
          element.target = '_blank';
          element.rel = 'noopener noreferrer';
          appendInlineMarkdown(element, token.match[1]);
        }
      }

      if (element) parent.append(element);
      else parent.append(document.createTextNode(token.match[0]));
      offset = token.match.index + token.match[0].length;
    }
  }

  function splitTableRow(line) {
    const value = line.trim().replace(/^\|/, '').replace(/\|$/, '');
    const cells = [];
    let current = '';
    for (let index = 0; index < value.length; index += 1) {
      if (value[index] === '\\' && value[index + 1] === '|') {
        current += '|'; index += 1; continue;
      }
      if (value[index] === '|') {
        cells.push(current.trim()); current = ''; continue;
      }
      current += value[index];
    }
    cells.push(current.trim());
    return cells;
  }

  function isTableDelimiter(line) {
    const cells = splitTableRow(line);
    return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
  }

  function isMarkdownBlockStart(lines, index) {
    const line = lines[index] || '';
    if (!line.trim()) return true;
    if (/^\s*```/.test(line) || /^\s{0,3}#{1,4}\s+/.test(line)) return true;
    if (/^\s*>\s?/.test(line) || /^\s*(?:[-*+]\s+|\d+[.)]\s+)/.test(line)) return true;
    if (/^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) return true;
    return index + 1 < lines.length && line.includes('|') && isTableDelimiter(lines[index + 1]);
  }

  function appendMarkdownTable(parent, headerCells, rows) {
    const wrap = document.createElement('div'); wrap.className = 'markdown-table-wrap';
    const table = document.createElement('table');
    const head = document.createElement('thead'); const headRow = document.createElement('tr');
    headerCells.forEach((value) => { const cell = document.createElement('th'); appendInlineMarkdown(cell, value); headRow.append(cell); });
    head.append(headRow); table.append(head);
    const body = document.createElement('tbody');
    rows.forEach((values) => {
      const row = document.createElement('tr');
      headerCells.forEach((_, index) => { const cell = document.createElement('td'); appendInlineMarkdown(cell, values[index] || ''); row.append(cell); });
      body.append(row);
    });
    table.append(body); wrap.append(table); parent.append(wrap);
  }

  function renderMarkdown(source) {
    const fragment = document.createDocumentFragment();
    const normalized = String(source || '').replace(/\r\n?/g, '\n').trim().replace(/\n{3,}/g, '\n\n');
    if (!normalized) return fragment;
    const lines = normalized.split('\n');
    let index = 0;

    while (index < lines.length) {
      const line = lines[index];
      if (!line.trim()) { index += 1; continue; }

      const fence = line.match(/^\s*```\s*([\w.+-]*)\s*$/);
      if (fence) {
        const codeLines = []; index += 1;
        while (index < lines.length && !/^\s*```\s*$/.test(lines[index])) { codeLines.push(lines[index]); index += 1; }
        if (index < lines.length) index += 1;
        const pre = document.createElement('pre'); const code = document.createElement('code');
        if (fence[1]) code.dataset.language = fence[1].toLowerCase();
        code.textContent = codeLines.join('\n'); pre.append(code); fragment.append(pre); continue;
      }

      if (index + 1 < lines.length && line.includes('|') && isTableDelimiter(lines[index + 1])) {
        const headers = splitTableRow(line); const rows = []; index += 2;
        while (index < lines.length && lines[index].trim() && lines[index].includes('|')) {
          rows.push(splitTableRow(lines[index])); index += 1;
        }
        appendMarkdownTable(fragment, headers, rows); continue;
      }

      const heading = line.match(/^\s{0,3}(#{1,4})\s+(.+)$/);
      if (heading) {
        const element = document.createElement(`h${Math.min(heading[1].length + 2, 6)}`);
        appendInlineMarkdown(element, heading[2].trim()); fragment.append(element); index += 1; continue;
      }

      if (/^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
        fragment.append(document.createElement('hr')); index += 1; continue;
      }

      if (/^\s*>\s?/.test(line)) {
        const values = [];
        while (index < lines.length && /^\s*>\s?/.test(lines[index])) { values.push(lines[index].replace(/^\s*>\s?/, '')); index += 1; }
        const quote = document.createElement('blockquote'); appendInlineMarkdown(quote, values.join(' ')); fragment.append(quote); continue;
      }

      const listMatch = line.match(/^\s*([-*+]|\d+[.)])\s+(.+)$/);
      if (listMatch) {
        const ordered = /^\d/.test(listMatch[1]); const list = document.createElement(ordered ? 'ol' : 'ul');
        while (index < lines.length) {
          const itemMatch = lines[index].match(/^\s*([-*+]|\d+[.)])\s+(.+)$/);
          if (!itemMatch || /^\d/.test(itemMatch[1]) !== ordered) break;
          const item = document.createElement('li'); appendInlineMarkdown(item, itemMatch[2]); list.append(item); index += 1;
        }
        fragment.append(list); continue;
      }

      const paragraphLines = [line]; index += 1;
      while (index < lines.length && lines[index].trim() && !isMarkdownBlockStart(lines, index)) {
        paragraphLines.push(lines[index]); index += 1;
      }
      const paragraph = document.createElement('p');
      paragraphLines.forEach((value, lineIndex) => {
        const hardBreak = /\s{2}$/.test(value);
        appendInlineMarkdown(paragraph, value.trimEnd());
        if (lineIndex < paragraphLines.length - 1) paragraph.append(hardBreak ? document.createElement('br') : document.createTextNode(' '));
      });
      fragment.append(paragraph);
    }
    return fragment;
  }

  function recallKindLabel(kind) {
    return ({ fact: 'Semantic Fact', episode: 'Episodic', chat: 'Conversation Evidence' })[kind] || 'Memory';
  }

  function updateRecallFeedbackState(item, buttons, status) {
    buttons.forEach((button) => {
      const selected = item.feedback === button.dataset.feedback;
      button.classList.toggle('is-selected', selected);
      button.setAttribute('aria-pressed', String(selected));
      button.disabled = !item.feedback_enabled || Boolean(item.feedbackSaving);
    });
    if (item.kind === 'chat') status.textContent = '原始对话证据 · 仅查看';
    else if (!item.memory_available) status.textContent = 'Memory 已失效，不能继续反馈';
    else if (item.feedbackSaving) status.textContent = '正在记录反馈…';
    else if (item.feedback) {
      const label = item.feedback === 'helpful' ? '👍 Helpful' : '👎 Irrelevant';
      status.textContent = `已记录 ${label} · Helpful ${Number(item.helpful_count || 0)} / Irrelevant ${Number(item.irrelevant_count || 0)}`;
    } else status.textContent = '这条记忆对本次回答有帮助吗？';
  }

  function renderRecallPanel(memories, open = false) {
    if (!Array.isArray(memories) || !memories.length) return null;
    const panel = document.createElement('details'); panel.className = 'recall-panel'; panel.open = open;
    const summary = document.createElement('summary');
    const summaryLabel = document.createElement('span'); summaryLabel.textContent = `本轮召回 ${memories.length} 条记忆`;
    const summaryHint = document.createElement('small'); summaryHint.textContent = '查看来源与反馈';
    summary.append(summaryLabel, summaryHint); panel.append(summary);
    const list = document.createElement('div'); list.className = 'recall-list';
    memories.forEach((item) => {
      const card = document.createElement('article'); card.className = `recall-card kind-${item.kind || 'unknown'}`;
      const heading = document.createElement('header');
      const badge = document.createElement('span'); badge.className = 'recall-kind'; badge.textContent = recallKindLabel(item.kind);
      const title = document.createElement('strong'); title.textContent = item.subject || recallKindLabel(item.kind);
      heading.append(badge, title);
      const body = document.createElement('p'); body.textContent = String(item.text || '');
      const sources = (item.retrieval_sources || []).map((value) => String(value).toUpperCase()).join(' + ');
      const meta = document.createElement('small'); meta.className = 'recall-score';
      const score = Number(item.final_score);
      meta.textContent = `${sources || 'RECALL'}${Number.isFinite(score) ? ` · Final ${(score * 100).toFixed(1)}%` : ''}`;
      const feedbackRow = document.createElement('footer');
      const status = document.createElement('small'); status.className = 'recall-feedback-status';
      const controls = document.createElement('span'); controls.className = 'recall-feedback-controls';
      const feedbackOptions = item.kind === 'chat' ? [] : ['helpful', 'irrelevant'];
      const buttons = feedbackOptions.map((feedback) => {
        const button = document.createElement('button'); button.type = 'button'; button.dataset.feedback = feedback;
        button.textContent = feedback === 'helpful' ? '👍' : '👎';
        button.title = feedback === 'helpful' ? '这条记忆有帮助' : '这条记忆不相关';
        button.setAttribute('aria-label', button.title);
        button.addEventListener('click', async () => {
          if (!item.feedback_enabled || item.feedbackSaving || item.feedback === feedback) return;
          item.feedbackSaving = true; updateRecallFeedbackState(item, buttons, status);
          let feedbackError = null;
          try {
            const updated = await api('/api/memories/feedback', {
              method: 'POST',
              body: JSON.stringify({
                session_id: item.session_id,
                turn_id: item.turn_id,
                memory_key: item.memory_key,
                feedback,
              }),
            });
            Object.assign(item, updated);
          } catch (error) {
            feedbackError = error;
          } finally {
            item.feedbackSaving = false;
            updateRecallFeedbackState(item, buttons, status);
            if (feedbackError) status.textContent = `反馈失败：${feedbackError.message}`;
          }
        });
        controls.append(button); return button;
      });
      updateRecallFeedbackState(item, buttons, status);
      feedbackRow.append(status);
      if (buttons.length) feedbackRow.append(controls);
      card.append(heading, body, meta, feedbackRow); list.append(card);
    });
    panel.append(list); return panel;
  }

  function addMessage(role, content, createdAt = new Date().toISOString(), recalledMemories = [], recallOpen = false) {
    const empty = $('.chat-empty'); if (empty) empty.remove();
    const wrapper = document.createElement('article'); wrapper.className = `message ${role}`;
    if (role !== 'user') {
      const avatar = document.createElement('span'); avatar.className = 'message-avatar';
      avatar.setAttribute('aria-hidden', 'true'); wrapper.append(avatar);
    }
    const bubble = document.createElement('div'); bubble.className = 'message-bubble';
    const messageContent = document.createElement('div'); messageContent.className = 'message-content';
    if (role === 'user') messageContent.textContent = String(content || '').trim();
    else { messageContent.classList.add('markdown-body'); messageContent.append(renderMarkdown(content)); }
    bubble.append(messageContent);
    if (role !== 'user') {
      const recallPanel = renderRecallPanel(recalledMemories, recallOpen);
      if (recallPanel) bubble.append(recallPanel);
    }
    const time = document.createElement('time'); time.className = 'message-time'; time.textContent = formatDate(createdAt); bubble.append(time); wrapper.append(bubble);
    $('#chat-messages').append(wrapper); $('#chat-messages').scrollTop = $('#chat-messages').scrollHeight;
  }

  function isViewingHistory() {
    return Boolean(state.viewingSessionId && state.viewingSessionId !== state.currentSessionId);
  }

  function selectedSession() {
    return state.sessions.find((item) => item.session_id === state.viewingSessionId) || null;
  }

  function updateContextModeControl() {
    const select = $('#context-mode-select');
    if (!select) return;
    const historical = isViewingHistory();
    const selected = selectedSession();
    const visibleMode = historical ? (selected?.context_mode || 'cc') : state.contextMode;
    select.value = visibleMode;
    select.disabled = historical || state.contextModeLocked || state.turnRunning;
    $('#context-mode-hint').textContent = historical
      ? '历史会话的既定模式'
      : state.contextModeLocked
        ? '本对话已开始，模式已锁定'
        : '仅在对话开始前可选择';
  }

  function updateConversationMode() {
    const historical = isViewingHistory();
    const selected = selectedSession();
    $('#history-mode-banner').hidden = !historical;
    $('#chat-input').disabled = historical;
    $('#send-button').disabled = historical;
    $('#busy-interaction-control').hidden = !state.turnRunning || historical;
    $('#composer-hint').textContent = state.turnRunning ? '明确选择 Steer / Queue / Redirect 后发送' : 'Enter 发送 · Shift+Enter 换行';
    $('#chat-input').placeholder = historical ? '历史对话为只读，返回当前对话后可继续聊天' : (state.turnRunning ? '输入对当前运行的干预…' : 'Message gugugaga…');
    if (historical) $('#chat-session-label').textContent = `历史对话 · ${selected?.turn_count || 0} Turns`;
    else if (state.currentSessionId) $('#chat-session-label').textContent = `当前对话 · ${selected?.turn_count || 0} Turns`;
    else $('#chat-session-label').textContent = '当前对话 · 尚未开始';
    updateContextModeControl();
  }

  function setHistoryDrawer(open) {
    $('#session-drawer').hidden = !open;
    $('#history-toggle').setAttribute('aria-expanded', String(open));
  }

  function renderSessions() {
    const container = $('#session-list'); container.replaceChildren();
    if (!state.sessions.length) {
      const empty = document.createElement('div'); empty.className = 'session-empty'; empty.textContent = '还没有历史对话'; container.append(empty); return;
    }
    state.sessions.forEach((item) => {
      const button = document.createElement('button'); button.type = 'button'; button.className = `session-item${item.session_id === state.viewingSessionId ? ' is-active' : ''}`;
      const top = document.createElement('span'); top.className = 'session-item-top';
      const title = document.createElement('strong'); title.textContent = item.title || '未命名对话';
      const time = document.createElement('time'); time.textContent = item.updated_at ? formatDate(item.updated_at) : '刚刚';
      const preview = document.createElement('p'); preview.textContent = item.preview || '暂无消息';
      const meta = document.createElement('small'); meta.textContent = item.session_id === state.currentSessionId ? `当前 · ${(item.context_mode || 'cc').toUpperCase()} · ${item.turn_count || 0} Turns` : `${(item.context_mode || 'cc').toUpperCase()} · ${item.turn_count || 0} Turns`;
      top.append(title, time); button.append(top, preview, meta);
      button.addEventListener('click', async () => {
        state.viewingSessionId = item.session_id;
        renderSessions(); updateConversationMode(); setHistoryDrawer(false);
        await Promise.all([loadHistory(item.session_id), loadOverview(item.session_id)]);
      });
      container.append(button);
    });
  }

  async function loadSessions() {
    try {
      const data = await api('/api/sessions');
      state.sessions = data.items || [];
      if (data.current_session_id) {
        state.currentSessionId = data.current_session_id;
        if (!state.viewingSessionId) state.viewingSessionId = data.current_session_id;
      }
      renderSessions(); updateConversationMode();
    } catch (error) {
      const container = $('#session-list'); container.replaceChildren(); const empty = document.createElement('div'); empty.className = 'session-empty'; empty.textContent = error.message; container.append(empty);
    }
  }

  async function loadHistory(sessionId = state.viewingSessionId || state.currentSessionId) {
    try {
      const container = $('#chat-messages'); container.replaceChildren();
      if (!sessionId) {
        const empty = document.createElement('div'); empty.className = 'chat-empty'; empty.textContent = '开始一个新对话，长期记忆会继续为你服务。'; container.append(empty); updateConversationMode(); return;
      }
      const data = await api(`/api/chat/history?session_id=${encodeURIComponent(sessionId)}`);
      if (!(data.items || []).length) {
        const empty = document.createElement('div'); empty.className = 'chat-empty'; empty.textContent = '这是一个新对话，长期记忆仍然可用。'; container.append(empty); updateConversationMode(); return;
      }
      data.items.forEach((item) => addMessage(item.role, item.content, item.created_at, item.recalled_memories || []));
      updateConversationMode();
    } catch (error) {
      const empty = document.createElement('div'); empty.className = 'chat-empty'; empty.textContent = error.message; $('#chat-messages').append(empty);
    }
  }

  async function startNewConversation() {
    if (state.turnRunning) return;
    const input = $('#chat-input');
    if (input.value.trim() && !window.confirm('输入框中还有未发送内容，仍要开启新对话吗？')) return;
    $('#new-chat-button').disabled = true;
    try {
      const data = await api('/api/session/new', {
        method: 'POST', body: JSON.stringify({ context_mode: $('#context-mode-select').value }),
      });
      state.currentSessionId = data.session_id;
      state.viewingSessionId = data.session_id;
      state.contextMode = data.context_mode || $('#context-mode-select').value;
      state.contextModeLocked = false;
      state.eventQueue.length = 0;
      input.value = '';
      resetRuntimeGraph();
      setHistoryDrawer(false);
      await Promise.all([loadHistory(data.session_id), loadSessions(), loadOverview()]);
    } catch (error) {
      addMessage('assistant', `无法开启新对话：${error.message}`);
    } finally {
      $('#new-chat-button').disabled = false;
      updateConversationMode();
      if (!isViewingHistory()) input.focus();
    }
  }

  async function resumeConversation() {
    if (state.turnRunning || !isViewingHistory()) return;
    const sessionId = state.viewingSessionId;
    const input = $('#chat-input');
    if (input.value.trim() && !window.confirm('当前对话的输入框中还有未发送内容，仍要继续这条历史对话吗？')) return;
    $('#resume-session-button').disabled = true;
    $('#new-chat-button').disabled = true;
    try {
      const data = await api('/api/session/resume', {
        method: 'POST', body: JSON.stringify({ session_id: sessionId }),
      });
      state.currentSessionId = data.session_id;
      state.viewingSessionId = data.session_id;
      state.contextMode = data.context_mode || selectedSession()?.context_mode || 'cc';
      state.contextModeLocked = true;
      state.eventQueue.length = 0;
      input.value = '';
      resetRuntimeGraph();
      setHistoryDrawer(false);
      await Promise.all([loadHistory(data.session_id), loadSessions(), loadOverview()]);
    } catch (error) {
      addMessage('assistant', `无法继续这条对话：${error.message}`);
    } finally {
      $('#resume-session-button').disabled = false;
      $('#new-chat-button').disabled = false;
      updateConversationMode();
      if (!isViewingHistory()) input.focus();
    }
  }

  $('#history-toggle').addEventListener('click', async () => {
    const open = $('#session-drawer').hidden;
    setHistoryDrawer(open);
    if (open) await loadSessions();
  });
  $('#history-close').addEventListener('click', () => setHistoryDrawer(false));
  $('#new-chat-button').addEventListener('click', startNewConversation);
  $('#resume-session-button').addEventListener('click', resumeConversation);
  $('#context-mode-select').addEventListener('change', async (event) => {
    const previous = state.contextMode;
    const requested = event.target.value;
    if (state.contextModeLocked || isViewingHistory()) { event.target.value = previous; return; }
    event.target.disabled = true;
    try {
      const data = await api('/api/session/mode', {
        method: 'POST', body: JSON.stringify({ context_mode: requested }),
      });
      state.contextMode = data.context_mode;
      state.contextModeLocked = Boolean(data.locked);
      if (data.session_id) {
        state.currentSessionId = data.session_id;
        state.viewingSessionId = data.session_id;
      }
      await loadSessions();
    } catch (error) {
      state.contextMode = previous;
      addMessage('assistant', `无法切换上下文模式：${error.message}`);
    } finally {
      updateContextModeControl();
    }
  });
  $('#back-current-button').addEventListener('click', async () => {
    state.viewingSessionId = state.currentSessionId;
    renderSessions(); updateConversationMode();
    await Promise.all([loadHistory(state.currentSessionId), loadOverview(state.currentSessionId)]);
    $('#chat-input').focus();
  });

  async function sendMessage(message) {
    if (isViewingHistory()) return;
    state.turnRunning = true; state.contextModeLocked = true; updateConversationMode(); $('#new-chat-button').disabled = true; $('#settings-button').disabled = true; $('#typing-state').hidden = false; addMessage('user', message);
    try {
      const data = await api('/api/chat', { method: 'POST', body: JSON.stringify({ message }) });
      if (data.session_id) {
        state.currentSessionId = data.session_id;
        state.viewingSessionId = data.session_id;
      }
      addMessage('assistant', data.reply || 'Agent 未返回文本。', new Date().toISOString(), data.recalled_memories || [], true);
      await loadOverview();
      await loadSessions();
      if (state.page === 'memory') await loadMemories();
      if (state.page === 'tasks') await loadTasks();
      if (state.page === 'database') await loadTables();
    } catch (error) {
      addMessage('assistant', `无法完成本次请求：${error.message}`);
      activateStage('agent', 'error');
    } finally {
      state.turnRunning = false; $('#new-chat-button').disabled = false; $('#settings-button').disabled = false; $('#typing-state').hidden = true; updateConversationMode(); if (!isViewingHistory()) $('#chat-input').focus();
    }
  }

  $('#chat-form').addEventListener('submit', (event) => {
    event.preventDefault(); if (isViewingHistory()) return;
    const input = $('#chat-input'); const message = input.value.trim(); if (!message) return;
    input.value = '';
    if (state.turnRunning) {
      const action = $('#chat-interaction-action').value;
      api('/api/interactions', {
        method: 'POST', body: JSON.stringify({ target: 'main', action, content: message }),
      }).then(() => {
        addMessage('user', `[${action}] ${message}`);
      }).catch((error) => {
        addMessage('assistant', `无法提交 ${action}：${error.message}`);
      });
      return;
    }
    sendMessage(message);
  });

  $('#chat-input').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); $('#chat-form').requestSubmit(); }
  });
  $('#permission-allow').addEventListener('click', () => reviewPermission(true));
  $('#permission-deny').addEventListener('click', () => reviewPermission(false));
  $('#team-detail-close').addEventListener('click', closeTeamAgentDetail);
  $('#team-detail-backdrop').addEventListener('click', (event) => {
    if (event.target === $('#team-detail-backdrop')) closeTeamAgentDetail();
  });
  $('#team-config-form').addEventListener('input', () => { state.teamConfigDirty = true; });
  $('#team-config-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!state.teamDetailName) return;
    const role = $('#team-config-role').value.trim();
    const prompt = $('#team-config-prompt').value.trim();
    if (!role) {
      $('#team-config-result').textContent = '角色不能为空';
      return;
    }
    if (!prompt) {
      $('#team-config-result').textContent = 'Prompt 不能为空';
      return;
    }
    const allowedTools = $$('[data-tool-name]', $('#team-config-tools'))
      .filter((input) => input.checked)
      .map((input) => input.dataset.toolName);
    const save = $('#team-config-save'); const reset = $('#team-config-reset');
    save.disabled = true; reset.disabled = true;
    $('#team-config-result').textContent = '正在保存…';
    try {
      const result = await api(`/api/team/agents/${encodeURIComponent(state.teamDetailName)}/profile`, {
        method: 'PUT', body: JSON.stringify({ role, prompt, allowed_tools: allowedTools }),
      });
      state.teamConfigDirty = false;
      await Promise.all([loadTeamAgentDetail(), loadTasks()]);
      $('#team-config-result').textContent = teamConfigurationState(result);
    } catch (error) {
      $('#team-config-result').textContent = error.message;
    } finally { save.disabled = false; reset.disabled = false; }
  });
  $('#team-config-reset').addEventListener('click', async () => {
    if (!state.teamDetailName) return;
    if (!window.confirm('确认恢复创建时的角色、Prompt 和默认工具列表吗？')) return;
    const save = $('#team-config-save'); const reset = $('#team-config-reset');
    save.disabled = true; reset.disabled = true;
    $('#team-config-result').textContent = '正在恢复默认配置…';
    try {
      const result = await api(`/api/team/agents/${encodeURIComponent(state.teamDetailName)}/profile`, {
        method: 'PUT', body: JSON.stringify({ reset: true }),
      });
      state.teamConfigDirty = false;
      await Promise.all([loadTeamAgentDetail(), loadTasks()]);
      $('#team-config-result').textContent = teamConfigurationState(result);
    } catch (error) {
      $('#team-config-result').textContent = error.message;
    } finally { save.disabled = false; reset.disabled = false; }
  });
  $('#team-interaction-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!state.teamDetailName) return;
    const action = $('#team-interaction-action').value;
    const content = $('#team-interaction-content').value.trim();
    if (!content) return;
    const submit = $('#team-interaction-submit'); submit.disabled = true;
    $('#team-interaction-result').textContent = '正在提交…';
    try {
      const result = await api('/api/interactions', {
        method: 'POST',
        body: JSON.stringify({ target: `team:${state.teamDetailName}`, action, content }),
      });
      $('#team-interaction-content').value = '';
      $('#team-interaction-result').textContent = action === 'queue' ? `已创建可见任务 ${result.task_id}` : `已提交 ${action}`;
      await Promise.all([loadTeamAgentDetail(), loadTasks()]);
    } catch (error) {
      $('#team-interaction-result').textContent = error.message;
    } finally { submit.disabled = false; }
  });
  $('#team-detail-lifecycle').addEventListener('click', async () => {
    if (!state.teamDetailName) return;
    const button = $('#team-detail-lifecycle');
    const operation = button.dataset.operation;
    const verb = operation === 'stop' ? '关闭' : '重新启动';
    if (!window.confirm(`确认${verb} Team Agent「${state.teamDetailName}」吗？`)) return;
    button.disabled = true;
    try {
      if (operation === 'stop') {
        await api('/api/interactions', { method: 'POST', body: JSON.stringify({ target: `team:${state.teamDetailName}`, action: 'stop', content: '' }) });
      } else {
        await api(`/api/team/agents/${encodeURIComponent(state.teamDetailName)}/restart`, { method: 'POST', body: '{}' });
      }
      await Promise.all([loadTeamAgentDetail(), loadTasks()]);
    } catch (error) { $('#team-interaction-result').textContent = error.message; }
    finally { button.disabled = false; }
  });
  $('#team-detail-delete').addEventListener('click', async () => {
    if (!state.teamDetailName || !state.teamDetail) return;
    const agent = state.teamDetail.agent || {};
    if (agent.online || agent.status === 'stopping') return;
    if (!window.confirm(`确认永久删除 Team Agent「${state.teamDetailName}」吗？运行轨迹和审计记录会保留。`)) return;
    const button = $('#team-detail-delete'); button.disabled = true;
    try {
      await api(`/api/team/agents/${encodeURIComponent(state.teamDetailName)}`, { method: 'DELETE' });
      closeTeamAgentDetail();
      await loadTasks();
    } catch (error) {
      $('#team-interaction-result').textContent = error.message;
      button.disabled = false;
    }
  });

  async function init() {
    await loadStatus();
    await Promise.all([loadOverview(), loadSessions()]);
    await loadHistory();
    await loadPermissions();
    pollEvents();
    window.addEventListener('resize', () => requestAnimationFrame(drawTeamGraphEdges));
    setInterval(() => {
      if (!state.permission) return;
      state.permission.remaining_seconds = Math.max(0, Number(state.permission.remaining_seconds || 0) - 1);
      renderPermission();
      if (state.permission.remaining_seconds <= 0) loadPermissions();
    }, 1000);
    setInterval(() => { if (!state.turnRunning) loadOverview(); }, 5000);
    setInterval(() => { if (!state.turnRunning && state.page === 'tasks') loadTasks(); }, 15000);
    setInterval(() => { if (state.page === 'tasks') loadTeamCommunications(); }, 2500);
    setInterval(loadTeamAgentDetail, 2500);
  }

  init();
})();
