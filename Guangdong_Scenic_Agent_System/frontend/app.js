const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));
const state = {
  messages: [], sessions: [], sessionId: null, username: '', loggedIn: false, role: '',
  settings: {}, ready: false, isTyping: false, navigating: false, booting: true,
  before: null, hasMore: false, epoch: 0
};
let authMode = 'login';
let toastTimer;
let deleteArmTimer;
let modalReturnFocus;
let featuredIndex = 0;
const mobileLayout = window.matchMedia('(max-width: 760px)');
const CHAT_REQUEST_TIMEOUT_MS = 30_000;
const CHAT_JOB_TIMEOUT_MS = 30 * 60 * 1000;
function syncSidebar() {
  const open = $('#sidebar').classList.contains('is-open');
  $('#sidebar').inert = mobileLayout.matches && !open;
  $('#mobileMenuBtn').setAttribute('aria-expanded', String(open));
}
function closeSidebar() { $('#sidebar').classList.remove('is-open'); syncSidebar(); }
mobileLayout.addEventListener('change', syncSidebar);
const conversationList = $('#conversationList');
let historyScrollbarDragging = false;
function updateHistoryScrollbar(event) {
  const bounds = conversationList.getBoundingClientRect();
  const nearEdge = event.pointerType !== 'touch' && event.clientX >= bounds.right - 14 &&
    event.clientX <= bounds.right && event.clientY >= bounds.top && event.clientY <= bounds.bottom;
  conversationList.classList.toggle('is-scrollbar-visible', historyScrollbarDragging || nearEdge);
}
conversationList.addEventListener('pointermove', updateHistoryScrollbar);
conversationList.addEventListener('pointerleave', updateHistoryScrollbar);
conversationList.addEventListener('pointerdown', (event) => {
  updateHistoryScrollbar(event);
  historyScrollbarDragging = conversationList.classList.contains('is-scrollbar-visible');
});
document.addEventListener('pointerup', (event) => { historyScrollbarDragging = false; updateHistoryScrollbar(event); });
conversationList.addEventListener('keydown', (event) => {
  if ((event.key === 'Enter' || event.key === ' ') && event.target.matches('[data-session]')) {
    event.preventDefault(); selectSession(event.target.dataset.session);
  }
});
document.addEventListener('pointercancel', () => { historyScrollbarDragging = false; conversationList.classList.remove('is-scrollbar-visible'); });
window.addEventListener('blur', () => { historyScrollbarDragging = false; conversationList.classList.remove('is-scrollbar-visible'); });
const featuredSpots = [
  { name: '广东第一峰', location: '清远 · 云海', badge: '山野风光', description: '在山野与云海之间，放慢一个周末。', tags: ['自然风光', '登山'], prompt: '介绍一下广东第一峰，适合什么时候去？', visual: 'scene-1' },
  { name: '丹霞山', location: '韶关 · 丹霞地貌', badge: '自然风光', description: '走近丹霞地貌，了解山川的故事。', tags: ['丹霞地貌', '徒步'], prompt: '介绍一下丹霞山景区，有什么值得看的？', visual: 'scene-2' },
  { name: '长隆野生动物世界', location: '广州 · 番禺', badge: '亲子出游', description: '和孩子一起，走进自然课堂。', tags: ['亲子出游', '动物'], prompt: '长隆野生动物世界适合亲子游吗？', visual: 'scene-3' },
  { name: '西樵山', location: '佛山 · 岭南名山', badge: '山水人文', description: '在山水之间，寻找岭南文化的印记。', tags: ['山水人文', '周末游'], prompt: '介绍一下西樵山景区，适合一天游吗？', visual: 'scene-4' },
  { name: '惠州西湖', location: '惠州 · 城市山水', badge: '城市漫游', description: '沿着湖畔慢慢走，享受半天闲暇。', tags: ['湖光山色', '休闲'], prompt: '介绍一下惠州西湖，有哪些游玩建议？', visual: 'scene-5' }
];

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
}
function createRequestId() {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID();
  // LAN HTTP pages lack randomUUID; getRandomValues still supplies random bytes.
  return Array.from(crypto.getRandomValues(new Uint8Array(16)),
    (byte) => byte.toString(16).padStart(2, '0')).join('');
}
function showToast(text) {
  $('#toast').textContent = text; $('#toast').classList.add('show');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => $('#toast').classList.remove('show'), 4000);
}
function setError(id, message = '') { $(id).textContent = message; $(id).hidden = !message; }
function lastSession(value) {
  const key = `scenic:last-session:${state.username}`;
  try { if (value !== undefined) localStorage.setItem(key, value); return localStorage.getItem(key); }
  catch { return null; }
}
async function api(path, { method = 'GET', body, timeout = 30000, allowAnonymous = false } = {}) {
  const controller = new AbortController(); const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(`/api${path}`, {
      method, credentials: 'same-origin', signal: controller.signal,
      headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'scenic-web' },
      ...(body === undefined ? {} : { body: JSON.stringify(body) })
    });
    const data = await response.json();
    if (!response.ok) {
      if (response.status === 401 && !allowAnonymous) { clearAccount(); openLoginModal(); }
      const error = new Error(data.message || '请求失败，请稍后重试。'); error.status = response.status; throw error;
    }
    return data;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('等待超时。服务可能仍在处理，请稍后加载历史，或点击重试。');
    if (error instanceof TypeError) throw new Error('无法连接服务，请检查网络和后端是否启动。');
    throw error;
  } finally { clearTimeout(timer); }
}
function updateControls() {
  const busy = state.isTyping || state.navigating || state.booting;
  for (const id of ['#sendBtn', '#newChatBtn', '#loadOlderBtn']) $(id).disabled = busy;
  $$('.conversation-item, [data-prompt], [data-retry], [data-open-settings]').forEach((node) => {
    node.disabled = busy;
    if (node.classList.contains('conversation-item')) node.setAttribute('aria-disabled', String(busy));
  });
  $('#messages').setAttribute('aria-busy', String(state.isTyping));
}
function updateAccountState() {
  $('#profileTitle').textContent = state.loggedIn ? state.username : '未登录';
  $('#profileAvatar').textContent = state.loggedIn ? state.username.slice(0, 1) : '游';
  $('#profileSubtitle').textContent = state.loggedIn ? (state.settings.configured ? `${state.settings.model_name} · 已配置` : '尚未配置模型') : '登录后配置 API';
  $('#loginBtn').classList.toggle('is-hidden', state.loggedIn);
  for (const id of ['#openKeyBtn', '#changePasswordBtn', '#feedbackBtn', '#logoutBtn', '#accountMenuDivider']) $(id).classList.toggle('is-hidden', !state.loggedIn);
  $('#adminBtn').classList.toggle('is-hidden', !(state.loggedIn && state.role === 'admin'));
  updateControls();
}
function clearAccount() {
  state.epoch += 1;
  Object.assign(state, { loggedIn: false, username: '', role: '', settings: {}, sessions: [], messages: [], sessionId: null,
    isTyping: false, navigating: false, hasMore: false, before: null });
  $('#messageInput').value = ''; closeModal('#keyModal'); closeModal('#feedbackModal'); closeModal('#recoveryModal'); closeModal('#changePasswordModal'); closeAccountMenu();
  renderSessions(); renderMessages(); updateAccountState();
}
function renderSessions() {
  $('#conversationList').innerHTML = state.sessions.length ? state.sessions.map((session) =>
    `<div class="conversation-item${session.session_id === state.sessionId ? ' is-active' : ''}" role="button" tabindex="0" data-session="${escapeHtml(session.session_id)}" aria-current="${session.session_id === state.sessionId}"><span class="conversation-copy"><strong>${escapeHtml(session.title)}</strong><small>${escapeHtml(new Date(session.updated_at).toLocaleString('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' }))}</small></span><button class="conversation-delete" type="button" data-delete-session="${escapeHtml(session.session_id)}" aria-label="删除对话：${escapeHtml(session.title)}">×</button></div>`
  ).join('') : `<p class="empty-state">${state.loggedIn ? '暂无对话，点击「开启新对话」开始' : '登录后查看历史对话'}</p>`;
  updateControls();
}
function sourceMarkup(sources = []) {
  const links = sources.flatMap((source) => {
    try {
      const url = new URL(source.source_url); if (!['http:', 'https:'].includes(url.protocol)) return [];
      return [`<a href="${escapeHtml(url.href)}" target="_blank" rel="noopener noreferrer">${escapeHtml(source.spot_name || source.source_domain || url.hostname)}</a>`];
    } catch { return []; }
  });
  return links.length ? `<div class="answer-sources">参考资料：${[...new Set(links)].join(' · ')}</div>` : '';
}
function spotMarkup(spots = []) {
  if (!spots.length) return '';
  return `<details class="answer-details"><summary>相关景区资料（${spots.length}）</summary>${spots.map((spot) =>
    `<div class="result-spot"><div class="result-spot-copy"><strong>${escapeHtml(spot.spot_name)}</strong><small>${escapeHtml([spot.city, spot.level, spot.ticket && `门票：${spot.ticket}`, spot.open_time && `开放时间：${spot.open_time}`, spot.address && `地址：${spot.address}`, spot.telephone && `电话：${spot.telephone}`].filter(Boolean).join(' · '))}</small></div></div>`
  ).join('')}</details>`;
}
function renderMessages(scroll = true) {
  $('#welcomePanel').hidden = state.messages.length > 0; $('#loadOlderBtn').hidden = !state.hasMore;
  $('#messages').innerHTML = state.messages.map((message, index) => {
    const user = message.role === 'user';
    return `<div class="message-row${user ? ' user' : ''}"><div class="message-content"><div class="message-meta"><span>${user ? '你' : '粤游智答'}</span></div><div class="message-bubble${message.error ? ' message-error' : ''}"><p>${escapeHtml(message.text)}</p>${user || message.error ? '' : spotMarkup(message.spots) + sourceMarkup(message.sources)}</div>${user ? '' : `<div class="message-actions">${message.error ? `<button type="button" data-retry="${index}">重试本次提问</button>${message.modelError ? '<button type="button" data-open-settings>检查模型配置</button>' : ''}` : `<button type="button" data-copy="${index}">复制回答</button>`}</div>`}</div></div>`;
  }).join('') + (state.isTyping ? '<div class="loading-answer" role="status">正在查询资料并整理回答，请稍候……</div>' : '');
  updateControls(); if (scroll) $('#chatScroll').scrollTop = $('#chatScroll').scrollHeight;
}
async function refreshSessions() {
  const epoch = state.epoch; const data = await api('/sessions');
  if (epoch !== state.epoch) return;
  state.sessions = data.sessions; renderSessions();
}
async function selectSession(sessionId) {
  if (state.isTyping || state.navigating) return;
  state.navigating = true; updateControls(); setError('#historyError'); const epoch = state.epoch;
  try {
    const data = await api(`/sessions/${encodeURIComponent(sessionId)}/messages`);
    if (epoch !== state.epoch) return;
    state.sessionId = sessionId; state.messages = data.messages; state.before = data.before; state.hasMore = data.has_more;
    lastSession(sessionId); renderSessions(); renderMessages(); closeSidebar();
  } catch (error) { setError('#historyError', error.message); }
  finally { state.navigating = false; updateControls(); }
}
function wait(ms) { return new Promise((resolve) => setTimeout(resolve, ms)); }
async function waitForChatJob(sessionId, requestId, epoch) {
  const deadline = Date.now() + CHAT_JOB_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const job = await api(`/chat/jobs/${encodeURIComponent(sessionId)}/${encodeURIComponent(requestId)}`, { timeout: CHAT_REQUEST_TIMEOUT_MS });
    if (epoch !== state.epoch) throw new Error('账户已切换，请重新提问。');
    if (job.status === 'succeeded') return job.result;
    if (job.status === 'failed' || job.status === 'interrupted') {
      const error = new Error(job.error?.message || '问答任务未完成，请重试。');
      error.status = job.error?.status;
      throw error;
    }
    await wait(1500);
  }
  throw new Error('等待回答超过 30 分钟。任务可能仍在后台处理，请稍后查看历史或点击重试。');
}
async function createSession() {
  const epoch = state.epoch; const data = await api('/sessions', { method: 'POST' });
  if (epoch !== state.epoch) throw new Error('账户已切换，请重新提问。');
  state.sessionId = data.session_id; state.messages = []; state.before = null; state.hasMore = false;
  lastSession(state.sessionId); renderMessages();
  try { await refreshSessions(); } catch (error) { showToast(error.message); }
}
async function newChat() {
  if (state.isTyping || state.navigating || state.booting) return;
  if (!state.loggedIn) return openLoginModal();
  state.navigating = true; updateControls();
  try { await createSession(); closeSidebar(); $('#messageInput').focus(); showToast('已开启新对话'); }
  catch (error) { showToast(error.message); }
  finally { state.navigating = false; updateControls(); }
}
async function submitMessage(rawText, retry = null) {
  const text = rawText.trim();
  if (!text || state.isTyping || state.navigating || state.booting) return;
  if (!state.loggedIn) { $('#messageInput').value = text; return openLoginModal(); }
  if (!state.settings.configured) { $('#messageInput').value = text; return openKeyModal(); }
  if (!state.ready) return showToast('问答服务未就绪，请点击顶部状态重试连接。');
  if (text.length > 4000) return showToast('问题请控制在 4000 字以内。');
  state.isTyping = true; updateControls(); const epoch = state.epoch;
  let requestId = retry?.requestId;
  try {
    requestId ||= createRequestId();
    if (!state.sessionId) await createSession();
    if (retry) state.messages = state.messages.filter((message) => message !== retry);
    else state.messages.push({ role: 'user', text });
    $('#messageInput').value = ''; $('#messageInput').style.height = 'auto'; renderMessages();
    const submitted = await api('/chat/jobs', { method: 'POST', timeout: CHAT_REQUEST_TIMEOUT_MS,
      body: { session_id: state.sessionId, message: text, request_id: requestId } });
    const result = submitted.result || await waitForChatJob(state.sessionId, requestId, epoch);
    if (epoch !== state.epoch) return;
    state.messages.push({ role: 'assistant', text: result.data.answer, spots: result.data.spots, sources: result.data.sources });
    try { await refreshSessions(); } catch (error) { showToast(`回答已收到，历史列表更新失败：${error.message}`); }
  } catch (error) {
    if (epoch === state.epoch) state.messages.push({ role: 'assistant', text: error.message, error: true, modelError: error.status === 502, question: text, requestId });
  } finally { if (epoch === state.epoch) { state.isTyping = false; renderMessages(); } }
}
function resetArmedDelete() {
  $$('.conversation-delete.is-armed').forEach((el) => { el.classList.remove('is-armed'); el.textContent = '×'; });
}
function armOrDeleteConversation(button) {
  if (state.isTyping || state.navigating) return showToast('正在处理对话，请稍后再删除。');
  if (button.classList.contains('is-armed')) {
    clearTimeout(deleteArmTimer);
    return deleteConversation(button.dataset.deleteSession);
  }
  resetArmedDelete();
  button.classList.add('is-armed'); button.textContent = '确认';
  deleteArmTimer = setTimeout(() => { button.classList.remove('is-armed'); button.textContent = '×'; }, 3000);
}
async function deleteConversation(sessionId) {
  try {
    await api(`/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
    state.sessions = state.sessions.filter((session) => session.session_id !== sessionId);
    if (state.sessionId === sessionId) {
      state.sessionId = null; state.messages = []; state.before = null; state.hasMore = false;
      lastSession(''); renderMessages();
    }
    renderSessions(); showToast('对话已删除');
  } catch (error) { showToast(error.message); }
}
async function loadOlder() {
  if (!state.hasMore || state.isTyping || state.navigating) return;
  state.navigating = true; updateControls(); setError('#historyError'); const epoch = state.epoch;
  const scroll = $('#chatScroll'); const oldHeight = scroll.scrollHeight; const oldTop = scroll.scrollTop;
  try {
    const data = await api(`/sessions/${encodeURIComponent(state.sessionId)}/messages?before=${state.before}`);
    if (epoch !== state.epoch) return;
    state.messages.unshift(...data.messages); state.before = data.before; state.hasMore = data.has_more;
    renderMessages(false); scroll.scrollTop = oldTop + scroll.scrollHeight - oldHeight;
  } catch (error) { setError('#historyError', error.message); }
  finally { state.navigating = false; updateControls(); }
}
function closeAccountMenu() {
  $('#accountMenu').classList.remove('is-open'); $('#accountMenu').setAttribute('aria-hidden', 'true');
  $('#accountMenuBtn').setAttribute('aria-expanded', 'false'); $('#accountMenu').inert = true;
}
function openModal(id, focusId) {
  closeAccountMenu(); modalReturnFocus = document.activeElement;
  $(id).classList.add('is-open'); $(id).setAttribute('aria-hidden', 'false'); $(id).inert = false;
  $('.app-shell').inert = true; $(focusId).focus();
}
function closeModal(id) {
  if (!$(id).classList.contains('is-open')) return;
  $(id).classList.remove('is-open'); $(id).setAttribute('aria-hidden', 'true'); $(id).inert = true;
  $('.app-shell').inert = false; if (modalReturnFocus?.isConnected) modalReturnFocus.focus();
}
function openLoginModal() { setAuthMode('login'); setError('#authError'); openModal('#loginModal', '#accountInput'); }
function setAuthMode(mode) {
  authMode = mode;
  const register = mode === 'register';
  $('#loginModalTitle').textContent = register ? '创建账户' : '登录账户';
  $('#loginModalCopy').textContent = '登录后可以保存你的服务配置，并使用自己的 API 进行提问。';
  $('#loginFields').classList.remove('is-hidden');
  $('#registerPasswordConfirmRow').classList.toggle('is-hidden', !register);
  $('#passwordInput').autocomplete = register ? 'new-password' : 'current-password';
  $('#loginNoteText').textContent = '使用已有账户登录，或注册新账户。密码 8–16 位、仅限字母和数字、不能以数字开头。';
  $('#submitLoginBtn').textContent = register ? '注册并继续' : '登录并继续';
  $('#authSwitchBtn').textContent = register ? '已有账号？返回登录' : '没有账号？立即注册';
  $('#authForgotBtn').classList.toggle('is-hidden', register);
  setError('#authError');
}
function openKeyModal() {
  if (!state.loggedIn) return openLoginModal();
  $('#providerUrlInput').value = state.settings.base_url || ''; $('#modelNameInput').value = state.settings.model_name || '';
  $('#apiKeyInput').value = ''; $('#apiKeyInput').type = 'password'; $('#toggleKeyVisibilityBtn').textContent = '显示';
  $('#apiKeyInput').placeholder = state.settings.configured ? '已保存，留空保留原密钥' : '请输入你的 API 密钥';
  $('#keyHint').textContent = state.settings.configured ? '密钥已保存；留空保留，填写新值则替换。' : '首次配置请填写密钥。';
  const url = state.settings.base_url || '';
  $('#providerSelect').value = url.includes('api.deepseek.com') ? 'deepseek' : url.includes('api.openai.com') ? 'openai' : 'custom';
  setError('#settingsError'); setError('#identityError');
  $('#identityNameInput').value = ''; $('#identityBirthInput').value = ''; $('#identityPhraseInput').value = '';
  loadIdentityStatus(); openModal('#keyModal', '#modelNameInput');
}
async function restoreAccount(data) {
  Object.assign(state, { loggedIn: true, username: data.username, role: data.role || 'user', settings: data.settings });
  updateAccountState(); await refreshSessions();
  const remembered = lastSession(); const selected = state.sessions.find((session) => session.session_id === remembered) || state.sessions[0];
  if (selected) await selectSession(selected.session_id);
}
async function authenticate() {
  if ($('#submitLoginBtn').disabled) return;
  const username = $('#accountInput').value.trim(); const password = $('#passwordInput').value;
  if (!username || !password) { setError('#authError', '请输入用户名和密码。'); return (!username ? $('#accountInput') : $('#passwordInput')).focus(); }
  if (authMode === 'register') {
    if (password.length < 8 || password.length > 16) { setError('#authError', '注册密码长度需要 8 到 16 位。'); return $('#passwordInput').focus(); }
    if (!/^[A-Za-z0-9]+$/.test(password)) { setError('#authError', '密码只能由字母和数字组成，不能包含空格或其他特殊字符。'); return $('#passwordInput').focus(); }
    if (/^[0-9]/.test(password)) { setError('#authError', '密码不能以数字开头。'); return $('#passwordInput').focus(); }
    if (password !== $('#passwordConfirmInput').value) { setError('#authError', '两次输入的密码不一致。'); return $('#passwordInput').focus(); }
  }
  $('#submitLoginBtn').disabled = true; $('#authSwitchBtn').disabled = true; $('#submitLoginBtn').textContent = '正在验证…'; setError('#authError');
  try {
    const data = await api(`/auth/${authMode}`, { method: 'POST', body: { username, password }, allowAnonymous: true });
    $('#passwordInput').value = ''; $('#passwordConfirmInput').value = ''; closeModal('#loginModal');
    await restoreAccount(data); showToast('已登录'); if (!state.settings.configured) openKeyModal();
  } catch (error) { setError('#authError', error.message); showToast(error.message); }
  finally { $('#submitLoginBtn').disabled = false; $('#authSwitchBtn').disabled = false; $('#submitLoginBtn').textContent = authMode === 'register' ? '注册并继续' : '登录并继续'; }
}
async function saveSettings() {
  if ($('#saveKeyBtn').disabled) return;
  const base_url = $('#providerUrlInput').value.trim(); const model_name = $('#modelNameInput').value.trim(); const api_key = $('#apiKeyInput').value.trim();
  const missing = !base_url ? '#providerUrlInput' : !model_name ? '#modelNameInput' : !api_key && !state.settings.configured ? '#apiKeyInput' : null;
  if (missing) { setError('#settingsError', '请填写服务地址、模型名称和首次使用的密钥。'); return $(missing).focus(); }
  $('#saveKeyBtn').disabled = true; $('#saveKeyBtn').textContent = '正在保存…'; setError('#settingsError');
  try {
    state.settings = await api('/settings/llm', { method: 'PUT', body: { base_url, model_name, api_key: api_key || null } });
    $('#apiKeyInput').value = ''; closeModal('#keyModal'); updateAccountState(); showToast('配置已保存');
  } catch (error) { setError('#settingsError', error.message); }
  finally { $('#saveKeyBtn').disabled = false; $('#saveKeyBtn').textContent = '保存并继续'; }
}
async function loadIdentityStatus() {
  try {
    const data = await api('/settings/security-questions');
    $('#identityStatus').textContent = data.configured ? '已设置密码找回答案。重新填写三项可更新，全部留空可关闭。' : '尚未设置密码找回答案。';
  } catch (error) { $('#identityStatus').textContent = error.message; }
}
async function saveIdentityAnswers() {
  if ($('#saveIdentityBtn').disabled) return;
  const values = ['#identityNameInput', '#identityBirthInput', '#identityPhraseInput'].map((id) => $(id).value.trim());
  if (values.some(Boolean) && !values.every(Boolean)) {
    setError('#identityError', '请填写全部三项，或全部留空以关闭密码找回。'); return;
  }
  $('#saveIdentityBtn').disabled = true; $('#saveIdentityBtn').textContent = '正在保存…'; setError('#identityError');
  try {
    const result = await api('/settings/security-questions', { method: 'PUT', body: {
      full_name: values[0], birth_date: values[1], verification_phrase: values[2]
    } });
    $('#identityNameInput').value = ''; $('#identityBirthInput').value = ''; $('#identityPhraseInput').value = '';
    $('#identityStatus').textContent = result.configured ? '密码找回答案已保存。' : '密码找回验证已关闭。';
    showToast(result.configured ? '验证信息已保存' : '密码找回验证已关闭');
  } catch (error) { setError('#identityError', error.message); }
  finally { $('#saveIdentityBtn').disabled = false; $('#saveIdentityBtn').textContent = '保存验证信息'; }
}
function openChangePasswordModal() {
  for (const id of ['#currentPasswordInput', '#newPasswordInput', '#confirmNewPasswordInput']) $(id).value = '';
  setError('#changePasswordError'); closeAccountMenu(); openModal('#changePasswordModal', '#currentPasswordInput');
}
async function submitChangePassword() {
  if ($('#submitChangePasswordBtn').disabled) return;
  const oldPassword = $('#currentPasswordInput').value;
  const newPassword = $('#newPasswordInput').value;
  if (!oldPassword) { setError('#changePasswordError', '请输入当前密码。'); return $('#currentPasswordInput').focus(); }
  if (newPassword.length < 8 || newPassword.length > 16 || !/^[A-Za-z0-9]+$/.test(newPassword) || /^[0-9]/.test(newPassword)) {
    setError('#changePasswordError', '密码需为 8–16 位字母和数字，且不能以数字开头。'); return $('#newPasswordInput').focus();
  }
  if (newPassword !== $('#confirmNewPasswordInput').value) {
    setError('#changePasswordError', '两次输入的新密码不一致。'); return $('#confirmNewPasswordInput').focus();
  }
  $('#submitChangePasswordBtn').disabled = true; $('#submitChangePasswordBtn').textContent = '正在验证…'; setError('#changePasswordError');
  try {
    await api('/auth/reset-password', { method: 'POST', body: {
      username: state.username, old_password: oldPassword, new_password: newPassword
    }, allowAnonymous: true });
    $('#accountInput').value = state.username; closeModal('#changePasswordModal'); clearAccount();
    openLoginModal(); showToast('密码已修改，请使用新密码重新登录');
  } catch (error) { setError('#changePasswordError', error.message); }
  finally { $('#submitChangePasswordBtn').disabled = false; $('#submitChangePasswordBtn').textContent = '验证并修改'; }
}
function openRecoveryModal() {
  $('#recoveryUsernameInput').value = $('#accountInput').value.trim();
  for (const id of ['#recoveryNameInput', '#recoveryBirthInput', '#recoveryPhraseInput', '#recoveryNewPasswordInput', '#recoveryConfirmInput']) $(id).value = '';
  setError('#recoveryError'); closeModal('#loginModal'); openModal('#recoveryModal', '#recoveryUsernameInput');
}
async function submitRecovery() {
  if ($('#submitRecoveryBtn').disabled) return;
  const username = $('#recoveryUsernameInput').value.trim();
  const newPassword = $('#recoveryNewPasswordInput').value;
  if (!username) { setError('#recoveryError', '请输入用户名。'); return $('#recoveryUsernameInput').focus(); }
  if (newPassword.length < 8 || newPassword.length > 16 || !/^[A-Za-z0-9]+$/.test(newPassword) || /^[0-9]/.test(newPassword)) {
    setError('#recoveryError', '密码需为 8–16 位字母和数字，且不能以数字开头。'); return $('#recoveryNewPasswordInput').focus();
  }
  if (newPassword !== $('#recoveryConfirmInput').value) { setError('#recoveryError', '两次输入的新密码不一致。'); return $('#recoveryConfirmInput').focus(); }
  $('#submitRecoveryBtn').disabled = true; $('#submitRecoveryBtn').textContent = '正在验证…'; setError('#recoveryError');
  try {
    await api('/auth/reset-password-by-answers', { method: 'POST', body: {
      username, full_name: $('#recoveryNameInput').value.trim(), birth_date: $('#recoveryBirthInput').value.trim(),
      verification_phrase: $('#recoveryPhraseInput').value.trim(), new_password: newPassword
    }, allowAnonymous: true });
    $('#accountInput').value = username; $('#passwordInput').value = ''; closeModal('#recoveryModal');
    showToast('密码修改成功，请使用新密码登录'); openLoginModal();
  } catch (error) { setError('#recoveryError', error.message); }
  finally { $('#submitRecoveryBtn').disabled = false; $('#submitRecoveryBtn').textContent = '验证并修改'; }
}
async function submitFeedback() {
  if ($('#submitFeedbackBtn').disabled) return;
  const content = $('#feedbackInput').value.trim();
  if (!content) { setError('#feedbackError', '请填写反馈内容。'); return $('#feedbackInput').focus(); }
  $('#submitFeedbackBtn').disabled = true; $('#submitFeedbackBtn').textContent = '正在提交…'; setError('#feedbackError');
  try {
    await api('/feedback', { method: 'POST', body: { content } });
    $('#feedbackInput').value = ''; closeModal('#feedbackModal'); showToast('感谢你的反馈！');
  } catch (error) { setError('#feedbackError', error.message); }
  finally { $('#submitFeedbackBtn').disabled = false; $('#submitFeedbackBtn').textContent = '提交反馈'; }
}
async function checkHealth() {
  try {
    const data = await api('/health', { allowAnonymous: true }); state.ready = data.ready;
    $$('[data-service-status]').forEach((node) => { node.textContent = data.ready ? '问答服务就绪' : '服务未就绪'; });
  } catch { state.ready = false; $$('[data-service-status]').forEach((node) => { node.textContent = '连接失败'; }); }
  $$('.service-status, .online-pill').forEach((node) => node.classList.toggle('is-offline', !state.ready));
}
function renderFeaturedSlide(index) {
  featuredIndex = (index + featuredSpots.length) % featuredSpots.length; const spot = featuredSpots[featuredIndex];
  $('#featuredImage').className = `featured-image ${spot.visual}`; $('#featuredImage').setAttribute('aria-label', `${spot.name}景观示意图`);
  $('#featuredImageLabel').textContent = spot.location; $('#featuredBadge').textContent = spot.badge;
  $('#featuredTitle').textContent = spot.name; $('#featuredDescription').textContent = spot.description;
  $('#featuredTags').innerHTML = spot.tags.map((tag) => `<span>${escapeHtml(tag)}</span>`).join('');
  $('#featuredPrompt').dataset.prompt = spot.prompt; $('#carouselCounter').textContent = `${featuredIndex + 1} / ${featuredSpots.length}`;
  $('#carouselDots').innerHTML = featuredSpots.map((_, index) => `<button class="carousel-dot${index === featuredIndex ? ' is-active' : ''}" data-carousel-index="${index}" aria-label="查看第 ${index + 1} 个景区" aria-current="${index === featuredIndex}" type="button"></button>`).join('');
}

$('#chatForm').addEventListener('submit', (event) => { event.preventDefault(); submitMessage($('#messageInput').value); });
$('#messageInput').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); submitMessage(event.currentTarget.value); }
});
$('#messageInput').addEventListener('input', (event) => { event.currentTarget.style.height = 'auto'; event.currentTarget.style.height = `${Math.min(event.currentTarget.scrollHeight, 120)}px`; });
document.addEventListener('click', async (event) => {
  const deleteBtn = event.target.closest('[data-delete-session]');
  if (deleteBtn) { armOrDeleteConversation(deleteBtn); return; }
  resetArmedDelete();
  const sessionEl = event.target.closest('[data-session]');
  if (sessionEl) selectSession(sessionEl.dataset.session);
  const target = event.target.closest('button'); if (!target || target.disabled) return;
  if (target.dataset.prompt) submitMessage(target.dataset.prompt);
  if (target.dataset.carouselIndex !== undefined) renderFeaturedSlide(Number(target.dataset.carouselIndex));
  if (target.dataset.retry !== undefined) { const message = state.messages[Number(target.dataset.retry)]; if (message?.error) submitMessage(message.question, message); }
  if (target.hasAttribute('data-open-settings')) openKeyModal();
  if (target.dataset.copy !== undefined) {
    try { await navigator.clipboard.writeText(state.messages[Number(target.dataset.copy)].text); showToast('回答已复制'); }
    catch { showToast('浏览器未允许复制，请选中回答文本手动复制。'); }
  }
});
$('#newChatBtn').addEventListener('click', newChat); $('#loadOlderBtn').addEventListener('click', loadOlder);
$('#serviceStatusBtn').addEventListener('click', checkHealth);
$('#mobileMenuBtn').addEventListener('click', () => { $('#sidebar').classList.toggle('is-open'); syncSidebar(); });
$('#sidebarToggleBtn').addEventListener('click', () => {
  const collapsed = $('.app-shell').classList.toggle('sidebar-collapsed');
  $('#sidebarToggleBtn').setAttribute('aria-expanded', String(!collapsed)); $('#sidebarToggleBtn').setAttribute('aria-label', collapsed ? '展开侧边栏' : '收起侧边栏');
});
$('#carouselPrevBtn').addEventListener('click', () => renderFeaturedSlide(featuredIndex - 1));
$('#carouselNextBtn').addEventListener('click', () => renderFeaturedSlide(featuredIndex + 1));
$('#refreshBtn').addEventListener('click', () => renderFeaturedSlide(featuredIndex + 1));
$('#accountMenuBtn').addEventListener('click', () => {
  const open = !$('#accountMenu').classList.contains('is-open'); $('#accountMenu').classList.toggle('is-open', open); $('#accountMenu').inert = !open;
  $('#accountMenu').setAttribute('aria-hidden', String(!open)); $('#accountMenuBtn').setAttribute('aria-expanded', String(open));
});
$('#loginBtn').addEventListener('click', openLoginModal); $('#openKeyBtn').addEventListener('click', openKeyModal);
$('#changePasswordBtn').addEventListener('click', openChangePasswordModal);
$('#feedbackBtn').addEventListener('click', () => { $('#feedbackInput').value = ''; setError('#feedbackError'); openModal('#feedbackModal', '#feedbackInput'); });
$('#submitFeedbackBtn').addEventListener('click', submitFeedback);
for (const id of ['#closeFeedbackModalBtn', '#cancelFeedbackBtn']) $(id).addEventListener('click', () => closeModal('#feedbackModal'));
const help = () => { closeAccountMenu(); showToast('登录并配置模型后即可提问。可以查景区详情、按城市推荐，或点击左侧恢复历史对话。'); };
$('#helpBtn').addEventListener('click', help); $('#helpAccountBtn').addEventListener('click', help);
$('#logoutBtn').addEventListener('click', async () => {
  $('#logoutBtn').disabled = true;
  try { await api('/auth/logout', { method: 'POST' }); clearAccount(); showToast('已退出账号'); }
  catch (error) { showToast(error.message); } finally { $('#logoutBtn').disabled = false; }
});
$('#authSwitchBtn').addEventListener('click', () => setAuthMode(authMode === 'login' ? 'register' : 'login'));
$('#authForgotBtn').addEventListener('click', openRecoveryModal);
$('#submitLoginBtn').addEventListener('click', authenticate); $('#saveKeyBtn').addEventListener('click', saveSettings);
$('#saveIdentityBtn').addEventListener('click', saveIdentityAnswers);
$('#submitRecoveryBtn').addEventListener('click', submitRecovery);
$('#submitChangePasswordBtn').addEventListener('click', submitChangePassword);
for (const id of ['#closeChangePasswordBtn', '#cancelChangePasswordBtn']) $(id).addEventListener('click', () => closeModal('#changePasswordModal'));
for (const id of ['#closeRecoveryModalBtn', '#cancelRecoveryBtn']) $(id).addEventListener('click', () => {
  closeModal('#recoveryModal'); openLoginModal();
});
$('#providerSelect').addEventListener('change', (event) => {
  const provider = event.target.value; $('#providerUrlInput').value = provider === 'deepseek' ? 'https://api.deepseek.com' : provider === 'openai' ? 'https://api.openai.com/v1' : '';
  $('#modelNameInput').value = ''; $('#apiKeyInput').value = '';
});
$('#toggleKeyVisibilityBtn').addEventListener('click', () => {
  const show = $('#apiKeyInput').type === 'password'; $('#apiKeyInput').type = show ? 'text' : 'password'; $('#toggleKeyVisibilityBtn').textContent = show ? '隐藏' : '显示';
});
for (const id of ['#closeKeyModalBtn', '#cancelKeyBtn']) $(id).addEventListener('click', () => closeModal('#keyModal'));
for (const id of ['#closeLoginModalBtn', '#cancelLoginBtn']) $(id).addEventListener('click', () => closeModal('#loginModal'));
for (const id of ['#loginModal', '#keyModal', '#feedbackModal', '#recoveryModal', '#changePasswordModal']) {
  $(id).inert = true; $(id).addEventListener('click', (event) => { if (event.target === $(id)) closeModal(id); });
  $(id).addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && event.target.tagName === 'INPUT' && !event.isComposing) {
      event.preventDefault();
      if (id === '#loginModal') authenticate();
      else if (id === '#keyModal' && event.target.id.startsWith('identity')) saveIdentityAnswers();
      else if (id === '#keyModal') saveSettings();
      else if (id === '#recoveryModal') submitRecovery();
      else if (id === '#changePasswordModal') submitChangePassword();
    }
  });
}
document.addEventListener('click', (event) => {
  if (!event.target.closest('#accountMenu, #accountMenuBtn')) closeAccountMenu();
  if (mobileLayout.matches && !event.target.closest('#sidebar, #mobileMenuBtn')) closeSidebar();
});
document.addEventListener('keydown', (event) => {
  const modal = $('.modal-backdrop.is-open');
  if (event.key === 'Escape') { if (modal) closeModal(`#${modal.id}`); closeAccountMenu(); closeSidebar(); resetArmedDelete(); }
  if (event.key === 'Tab' && modal) {
    const elements = Array.from(modal.querySelectorAll('button, input, select')).filter((node) => !node.disabled && node.getClientRects().length);
    const first = elements[0], last = elements[elements.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  }
});
async function boot() {
  closeAccountMenu(); syncSidebar(); renderMessages(); renderSessions(); updateAccountState(); renderFeaturedSlide(0); await checkHealth();
  try { await restoreAccount(await api('/auth/me', { allowAnonymous: true })); }
  catch (error) { if (error.status !== 401) showToast(error.message); }
  finally { state.booting = false; updateControls(); }
}
boot();
