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
    pageScroll: { overview: 0, memory: 0, database: 0 },
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
    if (page === 'memory') loadMemories();
    if (page === 'database') loadTables();
  }

  $$('.nav-item').forEach((button) => button.addEventListener('click', () => setPage(button.dataset.page)));

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

  async function loadOverview(sessionId = state.viewingSessionId || state.currentSessionId) {
    try {
      const query = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
      const data = await api(`/api/overview${query}`);
      $('#metric-turn').textContent = `#${data.turn_count || 0}`;
      $('#metric-turn-time').textContent = data.last_turn_at ? `更新于 ${formatDate(data.last_turn_at)}` : '尚无记录';
      $('#metric-latency').textContent = Number.isFinite(data.last_latency_ms) ? `${(data.last_latency_ms / 1000).toFixed(1)}s` : '—';
      $('#metric-memory').textContent = data.memory_hits || 0;
      const memoryTotal = Number(data.memory?.facts || 0) + Number(data.memory?.episodes || 0);
      $('#metric-memory-total').textContent = `${memoryTotal} 条长期记忆`;
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
    input: '输入入口', retrieval_gate: 'Retrieval Gate', memory_injection: '记忆注入',
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
    if (stage === 'consolidation') {
      const consolidation = $('[data-stage="consolidation"]');
      consolidation?.classList.add(status === 'failed' ? 'is-error' : 'is-active');
      setTimeout(() => consolidation?.classList.remove('is-active'), 900);
    } else {
      const previous = state.currentStage ? $(`[data-stage="${state.currentStage}"]`) : null;
      if (previous && state.currentStage !== stage) {
        previous.classList.remove('is-active');
        if (!previous.classList.contains('is-error') && !previous.classList.contains('is-skipped')) previous.classList.add('is-complete');
      }
      const active = $(`[data-stage="${stage}"]`);
      if (active) {
        active.classList.remove('is-skipped');
        active.classList.toggle('is-active', status !== 'error');
        active.classList.toggle('is-error', status === 'error');
      }
      state.currentStage = stage;
    }
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
    let stage = event.stage;
    if (!stage) {
      if (event.type === 'turn_start') return null;
      else if (event.type === 'llm') {
        if (event.call_type === 'context_summary') return null;
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
        if (event.action === 'consolidate') stage = 'consolidation';
        else if (event.action === 'recall' && event.status === 'hit') return { action: 'memory-hit', memoryKinds: event.kinds || [] };
        else return null;
      }
      else if (event.type === 'turn_end') stage = 'reply';
      else if (event.type === 'turn_error') stage = 'error';
    }
    if (stage === 'error') return { stage: 'agent', status: 'error' };
    return stage ? { stage, status: event.status, memoryKinds: event.memory_kinds || [] } : null;
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
        if (!item.stage) continue;
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

  function addMessage(role, content, createdAt = new Date().toISOString()) {
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
    $('#send-button').disabled = historical || state.turnRunning;
    $('#chat-input').placeholder = historical ? '历史对话为只读，返回当前对话后可继续聊天' : 'Message gugugaga…';
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
      data.items.forEach((item) => addMessage(item.role, item.content, item.created_at));
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
    state.turnRunning = true; state.contextModeLocked = true; updateContextModeControl(); $('#send-button').disabled = true; $('#new-chat-button').disabled = true; $('#settings-button').disabled = true; $('#typing-state').hidden = false; addMessage('user', message);
    try {
      const data = await api('/api/chat', { method: 'POST', body: JSON.stringify({ message }) });
      if (data.session_id) {
        state.currentSessionId = data.session_id;
        state.viewingSessionId = data.session_id;
      }
      addMessage('assistant', data.reply || 'Agent 未返回文本。');
      await loadOverview();
      await loadSessions();
      if (state.page === 'memory') await loadMemories();
      if (state.page === 'database') await loadTables();
    } catch (error) {
      addMessage('assistant', `无法完成本次请求：${error.message}`);
      activateStage('agent', 'error');
    } finally {
      state.turnRunning = false; $('#new-chat-button').disabled = false; $('#settings-button').disabled = false; $('#typing-state').hidden = true; updateConversationMode(); if (!isViewingHistory()) $('#chat-input').focus();
    }
  }

  $('#chat-form').addEventListener('submit', (event) => {
    event.preventDefault(); if (state.turnRunning || isViewingHistory()) return;
    const input = $('#chat-input'); const message = input.value.trim(); if (!message) return;
    input.value = ''; sendMessage(message);
  });

  $('#chat-input').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); $('#chat-form').requestSubmit(); }
  });

  async function init() {
    await loadStatus();
    await Promise.all([loadOverview(), loadSessions()]);
    await loadHistory();
    pollEvents();
    setInterval(() => { if (!state.turnRunning) loadOverview(); }, 5000);
  }

  init();
})();
