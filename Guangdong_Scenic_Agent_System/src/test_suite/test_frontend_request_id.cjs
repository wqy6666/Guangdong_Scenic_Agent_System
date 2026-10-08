// Run with node src/test_suite/test_frontend_request_id.cjs; no server or model required.
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const { webcrypto } = require('node:crypto');
const vm = require('node:vm');
const { test } = require('node:test');

const source = readFileSync(path.resolve(__dirname, '../../frontend/app.js'), 'utf8');
function functionSource(start, end) {
  const begin = source.indexOf(start);
  const finish = source.indexOf(end, begin);
  assert.ok(begin >= 0 && finish > begin);
  return source.slice(begin, finish);
}
const functions = functionSource('function createRequestId()', 'function showToast(') +
  functionSource('async function submitMessage(', 'function resetArmedDelete(');

function setup(crypto = { getRandomValues: webcrypto.getRandomValues.bind(webcrypto) }) {
  const state = {
    loggedIn: true, settings: { configured: true }, ready: true,
    isTyping: false, navigating: false, booting: false, epoch: 0,
    sessionId: 'lan-session', messages: []
  };
  const requests = [];
  const input = { value: '', style: {} };
  const context = vm.createContext({
    crypto, Uint8Array, state,
    $: () => input,
    updateControls() {}, renderMessages() {},
    async refreshSessions() {},
    async api(url, options) {
      assert.equal(url, '/chat');
      requests.push(options.body);
      return { data: { answer: '测试回答', spots: [], sources: [] } };
    }
  });
  vm.runInContext(functions, context);
  return { context, state, requests };
}

test('secure contexts keep using native randomUUID', () => {
  const id = 'a329e20c-3744-48b5-b890-d49802de69bc';
  const { context } = setup({ randomUUID: () => id });
  assert.equal(context.createRequestId(), id);
});

test('LAN HTTP without randomUUID generates valid independent request IDs', async () => {
  const { context, requests, state } = setup();
  await context.submitMessage('手机第一次提问');
  await context.submitMessage('手机第二次提问');
  assert.equal(requests.length, 2);
  for (const request of requests) {
    assert.match(request.request_id, /^[a-f0-9]{32}$/);
    assert.equal(request.session_id, 'lan-session');
  }
  assert.notEqual(requests[0].request_id, requests[1].request_id);
  assert.equal(state.isTyping, false);
  assert.equal(state.messages.filter((item) => item.role === 'assistant').length, 2);
});

test('retry reuses the failed request ID to prevent duplicate submissions', async () => {
  const { context, requests, state } = setup();
  const success = context.api;
  context.api = async (url, options) => {
    requests.push(options.body);
    throw new Error('模拟网络中断');
  };
  await context.submitMessage('手机提问');
  const retry = state.messages.at(-1);
  assert.equal(retry.error, true);
  assert.equal(state.isTyping, false);
  context.api = success;
  await context.submitMessage(retry.question, retry);
  assert.equal(requests.length, 2);
  assert.equal(requests[0].request_id, requests[1].request_id);
  assert.equal(state.messages.filter((item) => item.role === 'user').length, 1);
  assert.equal(state.messages.some((item) => item.error), false);
  assert.equal(state.isTyping, false);
});

test('an ID generation failure restores the send controls', async () => {
  const { context, state, requests } = setup({
    getRandomValues() { throw new Error('模拟浏览器随机数异常'); }
  });
  await context.submitMessage('手机提问');
  assert.equal(requests.length, 0);
  assert.equal(state.isTyping, false);
  assert.equal(state.messages.at(-1).error, true);
});
