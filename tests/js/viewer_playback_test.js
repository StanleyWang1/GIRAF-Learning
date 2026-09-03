const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const appPath = process.argv[2];
if (!appPath) {
  throw new Error('expected the path to app.js');
}

const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
const elements = new Map();
const displayedUrls = [];
const revokedUrls = [];

class DummyElement {
  constructor(id = '') {
    this.id = id;
    this.className = '';
    this.classList = { add() {}, remove() {}, toggle() {} };
    this.dataset = {};
    this.style = {};
    this.textContent = '';
    this.value = '1';
    this.clientWidth = 800;
  }

  addEventListener() {}
  append() {}
  appendChild() {}
  querySelectorAll() { return []; }
  replaceChildren() {}
}

function element(id) {
  if (!elements.has(id)) {
    const value = new DummyElement(id);
    if (id === 'cameraFrame') {
      Object.defineProperty(value, 'src', {
        set(url) { displayedUrls.push(url); },
      });
    }
    elements.set(id, value);
  }
  return elements.get(id);
}

let fetchFrame = () => new Promise(() => {});
let objectUrlIndex = 0;
class TestImage {
  async decode() {
    await delay(25);
  }
}

const context = vm.createContext({
  URL,
  URLSearchParams,
  Image: TestImage,
  HTMLInputElement: class {},
  HTMLSelectElement: class {},
  cancelAnimationFrame() {},
  clearTimeout,
  console,
  document: {
    createElement: () => new DummyElement(),
    getElementById: element,
    querySelector: () => new DummyElement(),
  },
  fetch: (...args) => fetchFrame(...args),
  performance,
  requestAnimationFrame: () => 1,
  setTimeout,
  window: {
    addEventListener() {},
    clearTimeout,
    devicePixelRatio: 1,
    history: { replaceState() {} },
    location: { href: 'http://127.0.0.1:8080/', search: '' },
    setTimeout,
  },
});
context.URL.createObjectURL = () => `blob:frame-${++objectUrlIndex}`;
context.URL.revokeObjectURL = (url) => revokedUrls.push(url);

const source = fs.readFileSync(appPath, 'utf8');
vm.runInContext(
  `${source}\nglobalThis.__viewerTest = { state, queueFrame, playbackTick };`,
  context,
  { filename: appPath },
);

async function main() {
  const { state, queueFrame, playbackTick } = context.__viewerTest;
  state.currentEpisode = 0;
  state.currentFrame = 0;
  state.desiredFrame = 1;
  state.loadToken = 4;
  state.frameGeneration = 9;
  state.playing = true;
  state.schema = null;

  fetchFrame = async (url) => {
    const index = Number(new URL(url, 'http://127.0.0.1').searchParams.get('idx'));
    await delay(35);
    return {
      ok: true,
      blob: async () => ({ index }),
    };
  };

  queueFrame(1, 'scrub');
  await delay(5);
  state.desiredFrame = 2;
  queueFrame(2, 'scrub');
  await delay(180);

  assert.deepEqual(
    displayedUrls,
    ['blob:frame-1', 'blob:frame-2'],
    'a late same-session frame should display before the newest pending frame',
  );
  assert.equal(state.currentFrame, 2);
  assert.deepEqual(revokedUrls, ['blob:frame-1']);

  const requestedFrames = [];
  fetchFrame = (url) => {
    requestedFrames.push(Number(new URL(url, 'http://127.0.0.1').searchParams.get('idx')));
    return new Promise(() => {});
  };
  state.schema = { length: 445 };
  state.currentFrame = 0;
  state.desiredFrame = 0;
  state.playing = true;
  state.playAnchorFrame = 0;
  state.playAnchorTime = 100;
  playbackTick(99.5);
  assert.deepEqual(
    requestedFrames,
    [],
    'a pre-anchor animation timestamp must not request frame -1',
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
