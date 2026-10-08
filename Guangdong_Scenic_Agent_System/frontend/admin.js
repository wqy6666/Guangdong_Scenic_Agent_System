const $ = (selector) => document.querySelector(selector);
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
}
async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    method: options.method || 'GET', credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'scenic-web' },
    ...(options.body === undefined ? {} : { body: JSON.stringify(options.body) })
  });
  const data = await response.json();
  if (!response.ok) { const error = new Error(data.message || '请求失败。'); error.status = response.status; throw error; }
  return data;
}
function showError(message = '') { $('#adminError').textContent = message; $('#adminError').hidden = !message; }
const formatNumber = (value) => Number(value || 0).toLocaleString('zh-CN');
function switchTab(tab) {
  for (const name of ['stats', 'system', 'errors', 'tokens', 'feedback', 'users']) {
    const active = name === tab;
    $(`#adminTab${name[0].toUpperCase()}${name.slice(1)}`).classList.toggle('is-active', active);
    $(`#admin${name[0].toUpperCase()}${name.slice(1)}Section`).classList.toggle('is-hidden', !active);
  }
  showError();
  if (tab === 'stats') loadSummary();
  if (tab === 'system') loadSystem();
  if (tab === 'errors') loadErrors();
  if (tab === 'tokens') loadUsage();
  if (tab === 'feedback') loadFeedback();
  if (tab === 'users') loadUsers();
}
function humanBytes(bytes) {
  const value = Number(bytes || 0);
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(0)} MB`;
  return `${(value / 1024 ** 3).toFixed(1)} GB`;
}
function humanDuration(seconds) {
  const days = Math.floor(seconds / 86400); const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return days ? `${days} 天 ${hours} 小时` : `${hours} 小时 ${minutes} 分钟`;
}
async function loadSystem() {
  try {
    const data = await api('/admin/system');
    $('#monitorEngine').textContent = data.engine_ready ? '就绪' : `未就绪${data.engine_error ? ` · ${data.engine_error}` : ''}`;
    $('#monitorCpu').textContent = `${Number(data.cpu_percent).toFixed(1)}%`;
    $('#monitorMemory').textContent = `${Number(data.memory_percent).toFixed(1)}% · ${humanBytes(data.memory_used_bytes)}`;
    $('#monitorProcessMemory').textContent = humanBytes(data.process_memory_bytes);
    $('#monitorDisk').textContent = `${Number(data.disk_percent).toFixed(1)}% · 剩余 ${humanBytes(data.disk_free_bytes)}`;
    $('#monitorQueue').textContent = data.queue_size;
    $('#monitorUptime').textContent = humanDuration(data.uptime_seconds);
    $('#monitorTimestamp').textContent = new Date(data.timestamp).toLocaleTimeString('zh-CN');
    $('#monitorRequests').textContent = formatNumber(data.requests_7d.total);
    $('#monitorOutcomes').textContent = `${formatNumber(data.requests_7d.succeeded)} / ${formatNumber(data.requests_7d.failed)}`;
    $('#monitorLatency').textContent = `${formatNumber(data.requests_7d.average_duration_ms)} ms`;
  } catch (error) { showError(error.message); }
}
async function loadErrors() {
  try {
    const days = $('#errorDays').value;
    const data = await api(`/admin/errors?days=${days}&limit=100`);
    $('#errorTotal').textContent = `最近 ${data.days} 天：${formatNumber(data.total)} 条`;
    $('#errorBreakdown').textContent = Object.entries(data.by_type).length
      ? Object.entries(data.by_type).map(([name, count]) => `${name}: ${formatNumber(count)}`).join(' · ')
      : '所选时间内暂无错误';
    $('#errorList').innerHTML = data.errors.length ? data.errors.map((item) =>
      `<article class="error-row"><header><strong>${escapeHtml(item.category)} · ${escapeHtml(item.error_type)} · HTTP ${escapeHtml(item.status)}</strong><time>${escapeHtml(new Date(item.timestamp).toLocaleString('zh-CN'))}</time></header><p>${escapeHtml(item.message)}</p><small>用户：${escapeHtml(item.username)} · 阶段：${escapeHtml(item.stage)} · 请求编号：${escapeHtml(item.request_id || '无')}</small></article>`
    ).join('') : '<p class="empty-state">所选时间内没有问答错误</p>';
  } catch (error) { showError(error.message); }
}
async function loadUsage() {
  try {
    const period = $('#usagePeriod').value; const data = await api(`/admin/usage?period=${period}`);
    const rows = Object.entries(data.users).sort((a, b) => (b[1].input_tokens + b[1].output_tokens) - (a[1].input_tokens + a[1].output_tokens));
    $('#usageRows').innerHTML = rows.length ? rows.map(([username, usage]) =>
      `<tr><td>${escapeHtml(username)}</td><td>${formatNumber(usage.input_tokens)}</td><td>${formatNumber(usage.output_tokens)}</td><td>${formatNumber(usage.calls)}</td><td>${formatNumber(usage.unreported_calls)}</td></tr>`
    ).join('') : '<tr><td colspan="5">所选周期暂无用量</td></tr>';
    $('#usagePeriodLabel').textContent = `统计日期：${data.start} 至 ${data.end}（北京时间）。Token 数量以模型服务实际返回的 usage 为准；未上报调用单独计数，不估算。`;
  } catch (error) { showError(error.message); }
}
async function loadSummary() {
  try {
    const data = await api('/admin/summary');
    $('#statOpens').textContent = data.opens; $('#statQuestions').textContent = data.questions; $('#statUsers').textContent = data.users;
  } catch (error) { showError(error.message); }
}
async function loadFeedback() {
  try {
    const data = await api('/admin/feedback');
    $('#feedbackList').innerHTML = data.feedback.length ? data.feedback.map((item) =>
      `<article class="feedback-row"><div class="feedback-copy"><p>${escapeHtml(item.content)}</p><small>${escapeHtml(item.username)} · ${escapeHtml(new Date(item.create_time).toLocaleString('zh-CN'))}</small></div></article>`
    ).join('') : '<p class="empty-state">暂无用户反馈</p>';
  } catch (error) { showError(error.message); }
}
async function loadUsers() {
  try {
    const data = await api('/admin/users');
    $('#userList').innerHTML = data.users.map((user) => {
      const admin = user.username.toLowerCase() === 'admin';
      const today = user.usage_today || {}; const month = user.usage_month || {};
      return `<article class="user-row"><div class="user-row-copy"><strong>${escapeHtml(user.username)}</strong><small>注册于 ${escapeHtml(user.create_time || '未知')} · ${user.sessions} 个会话 · 今日 ${formatNumber((today.input_tokens || 0) + (today.output_tokens || 0))} Token · 本月 ${formatNumber((month.input_tokens || 0) + (month.output_tokens || 0))} Token</small></div><span class="status-pill${user.disabled ? ' is-disabled' : ''}">${user.disabled ? '已关停' : '正常'}</span>${admin ? '' : `<button class="text-action user-toggle" type="button" data-user="${escapeHtml(user.username)}" data-action="${user.disabled ? 'activate' : 'disable'}">${user.disabled ? '激活用户' : '关停用户'}</button>`}</article>`;
    }).join('') || '<p class="empty-state">暂无用户</p>';
  } catch (error) { showError(error.message); }
}
document.addEventListener('click', async (event) => {
  const tab = event.target.closest('[id^="adminTab"]');
  if (tab) { switchTab(tab.id.replace('adminTab', '').toLowerCase()); return; }
  const button = event.target.closest('[data-user]');
  if (!button) return;
  button.disabled = true;
  try {
    await api(`/admin/users/${encodeURIComponent(button.dataset.user)}/${button.dataset.action}`, { method: 'POST' });
    await loadUsers();
  } catch (error) { showError(error.message); button.disabled = false; }
});
$('#adminLogout').addEventListener('click', async (event) => {
  event.preventDefault();
  try { await api('/auth/logout', { method: 'POST' }); } finally { location.replace('/'); }
});
$('#errorDays').addEventListener('change', loadErrors);
$('#refreshErrors').addEventListener('click', loadErrors);
$('#usagePeriod').addEventListener('change', loadUsage);
(async function boot() {
  try {
    const user = await api('/auth/me');
    if (user.role !== 'admin') { location.replace('/'); return; }
    $('#adminIdentity').textContent = `管理员：${user.username}`;
    await loadSummary();
    setInterval(() => {
      if (!$('#adminSystemSection').classList.contains('is-hidden')) loadSystem();
    }, 15000);
  } catch { location.replace('/'); }
})();
