const els = {
  badge: document.getElementById('memoryBadge'),
  refresh: document.getElementById('refreshMemory'),
  userId: document.getElementById('memoryUserId'),
  factCount: document.getElementById('factCount'),
  episodeCount: document.getElementById('episodeCount'),
  summaryCount: document.getElementById('summaryCount'),
  ftsState: document.getElementById('ftsState'),
  factsList: document.getElementById('factsList'),
  latestSummary: document.getElementById('latestSummary'),
  recentList: document.getElementById('recentList'),
  recallForm: document.getElementById('recallForm'),
  recallQuery: document.getElementById('recallQuery'),
  recallList: document.getElementById('recallList'),
  contextPreview: document.getElementById('contextPreview')
};

function setBadge(text, cls = '') {
  if (!els.badge) return;
  els.badge.className = `badge ${cls}`.trim();
  els.badge.textContent = text;
}

function escapeText(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
}

function formatTime(iso, ts) {
  if (iso) return iso;
  if (ts) return new Date(ts * 1000).toLocaleString();
  return 'unknown time';
}

function renderFacts(facts) {
  if (!Array.isArray(facts) || facts.length === 0) {
    els.factsList.className = 'memoryList empty';
    els.factsList.textContent = 'No durable facts stored yet.';
    return;
  }
  els.factsList.className = 'memoryList';
  els.factsList.innerHTML = facts.map(f => `
    <article class="memoryItem">
      <div class="memoryItemHead"><strong>${escapeText(f.key)}</strong><span>${Math.round((f.confidence || 0) * 100)}%</span></div>
      <div>${escapeText(f.value)}</div>
      <small>${escapeText(f.source || 'unknown source')} · updated ${escapeText(formatTime('', f.updated_at))}</small>
    </article>
  `).join('');
}

function renderEpisodes(target, episodes, emptyText) {
  if (!Array.isArray(episodes) || episodes.length === 0) {
    target.className = 'memoryList empty';
    target.textContent = emptyText;
    return;
  }
  target.className = 'memoryList';
  target.innerHTML = episodes.map(ep => `
    <article class="memoryItem">
      <div class="memoryItemHead"><strong>Episode #${escapeText(ep.id)}</strong><span>${escapeText(formatTime(ep.iso, ep.ts))}</span></div>
      <div><span class="roleChip">user</span> ${escapeText(ep.user_text)}</div>
      ${ep.assistant_text ? `<div><span class="roleChip assistantChip">assistant</span> ${escapeText(ep.assistant_text)}</div>` : ''}
    </article>
  `).join('');
}

async function loadMemory(query = '') {
  setBadge('memory: loading…', 'warn');
  const params = new URLSearchParams({ recent: '12' });
  if (query) params.set('q', query);
  const response = await fetch(`/memory/data?${params.toString()}`);
  const data = await response.json();
  if (!data.ok) {
    setBadge(`memory: ${data.error || 'unavailable'}`, 'danger');
    return;
  }
  setBadge('memory: enabled', 'ok');
  if (els.userId) els.userId.textContent = data.user_id || 'default';
  const stats = data.stats || {};
  els.factCount.textContent = stats.facts ?? '0';
  els.episodeCount.textContent = stats.episodes ?? '0';
  els.summaryCount.textContent = stats.summaries ?? '0';
  els.ftsState.textContent = stats.fts_enabled ? 'on' : 'off';
  renderFacts(data.facts || []);
  els.latestSummary.textContent = data.latest_summary || 'No rolling summary stored yet.';
  renderEpisodes(els.recentList, data.recent || [], 'No episodes stored yet.');
  renderEpisodes(els.recallList, data.recalled || [], query ? 'No matching memories found.' : 'No search yet.');
  els.contextPreview.textContent = data.context_preview || 'No prompt memory would be injected yet.';
}

els.refresh?.addEventListener('click', () => loadMemory(els.recallQuery?.value.trim() || '').catch(err => setBadge(`memory: ${String(err)}`, 'danger')));
els.recallForm?.addEventListener('submit', event => {
  event.preventDefault();
  loadMemory(els.recallQuery.value.trim()).catch(err => setBadge(`memory: ${String(err)}`, 'danger'));
});

loadMemory().catch(err => setBadge(`memory: ${String(err)}`, 'danger'));
