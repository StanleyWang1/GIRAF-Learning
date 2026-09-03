const DEFAULT_FPS_CAP = 30;
const DEFAULT_SPEED = 1.0;
const DEFAULT_INFERENCE_SERVER_HOST = 'localhost';
const DEFAULT_INFERENCE_SERVER_PORT = 8766;
const DEFAULT_INFERENCE_WARMUP_STEPS = 1;
const DEFAULT_INFERENCE_BATCH_SIZE = 1;
const DEFAULT_INFERENCE_MODE = 'train_time';
const MAX_INFERENCE_BATCH_SIZE = 32;
const INFERENCE_SETTINGS_CACHE_KEY = 'dataset_visualization.inference_settings.v1';
const DEFAULT_PROGRESS_SERVER_HOST = 'localhost';
const DEFAULT_PROGRESS_SERVER_PORT = 8767;
const DEFAULT_PROGRESS_EVAL_EVERY = 10;
const PROGRESS_SETTINGS_CACHE_KEY = 'dataset_visualization.progress_settings.v1';
const PROGRESS_CURSOR_COLOR = '#ff9f1c';
const PERFORMANCE_SETTINGS_CACHE_KEY = 'dataset_visualization.performance_settings.v1';
const DEFAULT_PREVIEW_MODE = 'all';
const DEFAULT_IMAGE_MODE = 'full';
const DEFAULT_PLOT_RATE = 'full';
const FULL_FRAME_PROFILE = 'full';
const SCRUB_FRAME_PROFILE = 'scrub';
const PREVIEW_FRAME_PROFILE = 'preview';

const PLAYBACK_TARGET_MS = 33;
const PLOT_TARGET_MS = 33;
const LOW_RATE_PLOT_TARGET_MS = 200;
const MAIN_LOOKAHEAD = 10;
const MAIN_MAX_INFLIGHT = 3;
const MAX_FRAME_LAG = 2;
const MAIN_FRAME_TIMEOUT_MS = 2500;
const HIGH_QUALITY_REFRESH_DELAY_MS = 200;

const SIGNAL_CHUNK_SIZE = 512;
const SIGNAL_MAX_CHUNKS = 24;
const TRAJECTORY_CHUNK_SIZE = 512;
const TRAJECTORY_MAX_CHUNKS = 24;

const PREVIEW_BACKLOG_HIGH = 10;
const PREVIEW_BACKLOG_LOW = 4;
const PREVIEW_MAX_STRIDE = 6;

const FEET_TO_M = 0.3048;
const ROBOT_HALF_SEPARATION_FT = 1.5;
const ROBOT_HALF_SEPARATION_M = ROBOT_HALF_SEPARATION_FT * FEET_TO_M;

const LEGEND_ROW_HEIGHT_PX = 24;
const LEGEND_MIN_ITEM_PX = 120;
const LEGEND_MAX_ITEM_PX = 260;
const LEGEND_BASE_BOTTOM_PX = 36;
const LEGEND_GAP_FROM_PLOT_PX = 14;
const LEGEND_COMPACT_BOTTOM_PX = 28;
const LEGEND_ITEM_PADDING_PX = 12;
const PREDICTION_MEMBER_COLORS = { robot0: '#f4a261', robot1: '#52b788' };
const PREDICTION_MEMBER_COLOR_PALETTES = {
  robot0: ['#f4a261', '#ffb26b', '#f7c66f', '#f08a5d', '#ff8c42', '#f6bd60'],
  robot1: ['#52b788', '#74c69d', '#95d5b2', '#40916c', '#2d6a4f', '#1b4332'],
};
const PREDICTION_AVERAGE_COLORS = { robot0: '#e76f51', robot1: '#277da1' };
const PREDICTION_MEMBER_TRACE_OPACITY = 0.9;
const PREDICTION_MEMBER_LINE_WIDTH = 5;
const PREDICTION_AVERAGE_LINE_WIDTH = 11;
const PREDICTION_AVERAGE_MARKER_SIZE = 12;

const state = {
  summary: null,
  episodes: [],
  schema: null,
  events: [],
  daggerSegments: [],
  timing: null,
  currentEpisode: 0,
  currentFrame: 0,
  targetFrame: 0,
  isPlaying: false,
  playHandle: null,
  playAnchorMs: 0,
  playAnchorFrame: 0,
  timeline: {
    dragging: false,
  },
  performanceControls: {
    previewMode: DEFAULT_PREVIEW_MODE,
    imageMode: DEFAULT_IMAGE_MODE,
    plotRate: DEFAULT_PLOT_RATE,
  },
  selectedKeys: new Set(),
  primaryStream: null,
  historyFrames: 180,
  lastSignalsTs: [],
  lastSignalsWindowStart: 0,
  playMode: 'timestamp',
  fpsCap: DEFAULT_FPS_CAP,
  speedMultiplier: DEFAULT_SPEED,
  isDeletingEpisode: false,
  previewTiles: new Map(),
  previewAdaptive: false,
  previewStride: 1,
  episodeLoadToken: 0,
  mainFrameDisplay: {
    objectUrl: '',
    highQualityTimer: null,
    highQualityController: null,
  },
  perf: {
    main_frame_requested: 0,
    main_frame_displayed: 0,
    main_frame_dropped: 0,
    preview_frame_displayed: 0,
    inflight_requests: 0,
    plot2d_updates: 0,
    plot3d_updates: 0,
    effective_main_fps: 0,
    effective_plot2d_fps: 0,
    effective_plot3d_fps: 0,
    frame_lag: 0,
    cache_hit_rate: 0,
    samples: {
      main: { lastTs: 0, lastCount: 0 },
      plot2d: { lastTs: 0, lastCount: 0 },
      plot3d: { lastTs: 0, lastCount: 0 },
    },
  },
  mainPipeline: {
    key: '',
    token: 0,
    queue: new Map(),
    pendingOrder: [],
    inflight: new Set(),
    lookahead: MAIN_LOOKAHEAD,
    maxInflight: MAIN_MAX_INFLIGHT,
    cacheHits: 0,
    fetches: 0,
  },
  signalStore: createChunkStore(SIGNAL_CHUNK_SIZE, SIGNAL_MAX_CHUNKS),
  trajectoryStore: createChunkStore(TRAJECTORY_CHUNK_SIZE, TRAJECTORY_MAX_CHUNKS),
  plot: {
    pending: false,
    force: false,
    running: false,
    lastRunMs: 0,
  },
  signalsPlot: {
    initialized: false,
    structureSig: '',
    traceMeta: [],
    inFlight: false,
    pending: false,
    pendingForce: false,
    lastUpdateMs: 0,
    lastFrame: -1,
  },
  trajectoryPlot: {
    available: false,
    keys: [],
    initialized: false,
    inFlight: false,
    pending: false,
    pendingForce: false,
    lastUpdateMs: 0,
    lastFrame: -1,
    axisRadius: 0.8,
    staticTraceCount: 0,
    dynamicTraceIndices: {},
    overlaySig: '',
  },
  trajectoryCamera: null,
  trajectoryRelayoutBound: false,
  inference: {
    history: [],
    running: false,
    yamlPath: '',
    serverHost: DEFAULT_INFERENCE_SERVER_HOST,
    serverPort: DEFAULT_INFERENCE_SERVER_PORT,
    inferenceMode: DEFAULT_INFERENCE_MODE,
    warmupSteps: DEFAULT_INFERENCE_WARMUP_STEPS,
    batchSize: DEFAULT_INFERENCE_BATCH_SIZE,
    noGripper: false,
    statusText: 'Inference overlay idle.',
    statusKind: 'idle',
  },
  progressGraph: {
    history: {},      // key: `${episode_index}::${yaml}::${evalEvery}` -> { frames, progress, episode_length }
    currentKey: null,
    rendered: false,
    running: false,
    yamlPath: '',
    serverHost: DEFAULT_PROGRESS_SERVER_HOST,
    serverPort: DEFAULT_PROGRESS_SERVER_PORT,
    evalEvery: DEFAULT_PROGRESS_EVAL_EVERY,
    statusText: 'Progress graph idle.',
    statusKind: 'idle',
  },
};

const el = {
  summaryBadge: document.getElementById('summaryBadge'),
  perfBadge: document.getElementById('perfBadge'),
  unsupportedBanner: document.getElementById('unsupportedBanner'),
  episodeSelect: document.getElementById('episodeSelect'),
  deleteEpisodeBtn: document.getElementById('deleteEpisodeBtn'),
  frameInput: document.getElementById('frameInput'),
  timeInput: document.getElementById('timeInput'),
  playBtn: document.getElementById('playBtn'),
  previewModeSelect: document.getElementById('previewModeSelect'),
  imageModeSelect: document.getElementById('imageModeSelect'),
  plotRateSelect: document.getElementById('plotRateSelect'),
  playModeSelect: document.getElementById('playModeSelect'),
  fpsCapInput: document.getElementById('fpsCapInput'),
  speedInput: document.getElementById('speedInput'),
  historyInput: document.getElementById('historyInput'),
  primaryStreamSelect: document.getElementById('primaryStreamSelect'),
  depthToggle: document.getElementById('depthToggle'),
  colormapSelect: document.getElementById('colormapSelect'),
  mainFrame: document.getElementById('mainFrame'),
  cameraGrid: document.getElementById('cameraGrid'),
  timelineSlider: document.getElementById('timelineSlider'),
  timelineInfo: document.getElementById('timelineInfo'),
  eventMarkers: document.getElementById('eventMarkers'),
  keyList: document.getElementById('keyList'),
  eventList: document.getElementById('eventList'),
  signalsPlot: document.getElementById('signalsPlot'),
  trajectory3d: document.getElementById('trajectory3d'),
  inferenceYamlPath: document.getElementById('inferenceYamlPath'),
  inferenceServerHost: document.getElementById('inferenceServerHost'),
  inferenceServerPort: document.getElementById('inferenceServerPort'),
  inferenceModeSelect: document.getElementById('inferenceModeSelect'),
  inferenceWarmupSteps: document.getElementById('inferenceWarmupSteps'),
  inferenceBatchSize: document.getElementById('inferenceBatchSize'),
  inferenceNoGripper: document.getElementById('inferenceNoGripper'),
  runInferenceBtn: document.getElementById('runInferenceBtn'),
  clearInferenceBtn: document.getElementById('clearInferenceBtn'),
  inferenceStatus: document.getElementById('inferenceStatus'),
  progressYamlPath: document.getElementById('progressYamlPath'),
  progressServerHost: document.getElementById('progressServerHost'),
  progressServerPort: document.getElementById('progressServerPort'),
  progressEvalEvery: document.getElementById('progressEvalEvery'),
  runProgressBtn: document.getElementById('runProgressBtn'),
  clearProgressBtn: document.getElementById('clearProgressBtn'),
  progressStatus: document.getElementById('progressStatus'),
  progressPlot: document.getElementById('progressPlot'),
  uncheckSignalsBtn: document.getElementById('uncheckSignalsBtn'),
};

const presetButtons = Array.from(document.querySelectorAll('.preset-buttons button[data-preset]'));

function createChunkStore(chunkSize, maxChunks) {
  return {
    episode: null,
    keys: [],
    keysSig: '',
    token: 0,
    chunkSize,
    maxChunks,
    cache: new Map(),
    pending: new Map(),
  };
}

async function apiGet(path) {
  const res = await fetch(path);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

async function apiDelete(path) {
  const res = await fetch(path, { method: 'DELETE' });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const payload = await res.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

function revokeObjectUrl(url) {
  if (!url || typeof url !== 'string' || !url.startsWith('blob:')) {
    return;
  }
  URL.revokeObjectURL(url);
}

function isAbortError(err) {
  return Boolean(err && typeof err === 'object' && err.name === 'AbortError');
}

async function fetchImageObjectUrl(url, controller, timeoutMs) {
  const timeout = window.setTimeout(() => {
    controller.abort();
  }, timeoutMs);

  try {
    const res = await fetch(url, {
      signal: controller.signal,
      cache: 'no-store',
    });
    if (!res.ok) {
      throw new Error(`Request failed (${res.status})`);
    }
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  } finally {
    window.clearTimeout(timeout);
  }
}

function setUnsupported(reason) {
  el.unsupportedBanner.classList.remove('hidden');
  el.unsupportedBanner.textContent = reason;
}

function clearUnsupported() {
  el.unsupportedBanner.classList.add('hidden');
  el.unsupportedBanner.textContent = '';
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function toFiniteNumber(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function previewModeIsHideOthers() {
  return state.performanceControls.previewMode === 'hide-others';
}

function imageModeIsDynamic() {
  return state.performanceControls.imageMode === 'dynamic';
}

function computePlotRefreshIntervalMs() {
  if (state.performanceControls.plotRate === 'low' && (state.timeline.dragging || state.isPlaying)) {
    return LOW_RATE_PLOT_TARGET_MS;
  }
  return PLOT_TARGET_MS;
}

function sanitizeInferencePort(value) {
  return clamp(Math.round(toFiniteNumber(value, DEFAULT_INFERENCE_SERVER_PORT)), 1, 65535);
}

function sanitizeWarmupSteps(value) {
  return Math.max(1, Math.round(toFiniteNumber(value, DEFAULT_INFERENCE_WARMUP_STEPS)));
}

function sanitizeBatchSize(value) {
  return clamp(Math.round(toFiniteNumber(value, DEFAULT_INFERENCE_BATCH_SIZE)), 1, MAX_INFERENCE_BATCH_SIZE);
}

function sanitizeInferenceMode(value) {
  return DEFAULT_INFERENCE_MODE;
}

function inferenceModeLabel(value) {
  return 'Train-Time';
}

function persistInferenceSettings() {
  try {
    const payload = {
      yamlPath: String(state.inference.yamlPath || ''),
      serverHost: String(state.inference.serverHost || DEFAULT_INFERENCE_SERVER_HOST),
      serverPort: sanitizeInferencePort(state.inference.serverPort),
      inferenceMode: sanitizeInferenceMode(state.inference.inferenceMode),
      warmupSteps: sanitizeWarmupSteps(state.inference.warmupSteps),
      batchSize: sanitizeBatchSize(state.inference.batchSize),
      noGripper: !!state.inference.noGripper,
    };
    localStorage.setItem(INFERENCE_SETTINGS_CACHE_KEY, JSON.stringify(payload));
  } catch (_) {
    // localStorage may be unavailable in some browser contexts.
  }
}

function loadCachedInferenceSettings() {
  try {
    const raw = localStorage.getItem(INFERENCE_SETTINGS_CACHE_KEY);
    if (!raw) {
      return;
    }
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') {
      return;
    }
    state.inference.yamlPath = String(parsed.yamlPath || '').trim();
    state.inference.serverHost = String(parsed.serverHost || DEFAULT_INFERENCE_SERVER_HOST).trim() || DEFAULT_INFERENCE_SERVER_HOST;
    state.inference.serverPort = sanitizeInferencePort(parsed.serverPort);
    state.inference.inferenceMode = sanitizeInferenceMode(parsed.inferenceMode);
    state.inference.warmupSteps = sanitizeWarmupSteps(parsed.warmupSteps);
    state.inference.batchSize = sanitizeBatchSize(parsed.batchSize);
    state.inference.noGripper = !!parsed.noGripper;
  } catch (_) {
    // Ignore malformed cache.
  }
}

function sanitizeProgressPort(value) {
  return clamp(Math.round(toFiniteNumber(value, DEFAULT_PROGRESS_SERVER_PORT)), 1, 65535);
}

function sanitizeProgressEvalEvery(value) {
  return Math.max(1, Math.round(toFiniteNumber(value, DEFAULT_PROGRESS_EVAL_EVERY)));
}

function persistProgressSettings() {
  try {
    const payload = {
      yamlPath: String(state.progressGraph.yamlPath || ''),
      serverHost: String(state.progressGraph.serverHost || DEFAULT_PROGRESS_SERVER_HOST),
      serverPort: sanitizeProgressPort(state.progressGraph.serverPort),
      evalEvery: sanitizeProgressEvalEvery(state.progressGraph.evalEvery),
    };
    localStorage.setItem(PROGRESS_SETTINGS_CACHE_KEY, JSON.stringify(payload));
  } catch (_) {
    // localStorage may be unavailable in some browser contexts.
  }
}

function loadCachedProgressSettings() {
  try {
    const raw = localStorage.getItem(PROGRESS_SETTINGS_CACHE_KEY);
    if (!raw) {
      return;
    }
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') {
      return;
    }
    state.progressGraph.yamlPath = String(parsed.yamlPath || '').trim();
    state.progressGraph.serverHost = String(parsed.serverHost || DEFAULT_PROGRESS_SERVER_HOST).trim() || DEFAULT_PROGRESS_SERVER_HOST;
    state.progressGraph.serverPort = sanitizeProgressPort(parsed.serverPort);
    state.progressGraph.evalEvery = sanitizeProgressEvalEvery(parsed.evalEvery);
  } catch (_) {
    // Ignore malformed cache.
  }
}

function persistPerformanceSettings() {
  try {
    localStorage.setItem(PERFORMANCE_SETTINGS_CACHE_KEY, JSON.stringify({
      previewMode: state.performanceControls.previewMode,
      imageMode: state.performanceControls.imageMode,
      plotRate: state.performanceControls.plotRate,
    }));
  } catch (_) {
    // localStorage may be unavailable in some browser contexts.
  }
}

function loadCachedPerformanceSettings() {
  try {
    const raw = localStorage.getItem(PERFORMANCE_SETTINGS_CACHE_KEY);
    if (!raw) {
      return;
    }
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') {
      return;
    }
    if (parsed.previewMode === 'hide-others' || parsed.previewMode === 'all') {
      state.performanceControls.previewMode = parsed.previewMode;
    }
    if (parsed.imageMode === 'dynamic' || parsed.imageMode === 'full') {
      state.performanceControls.imageMode = parsed.imageMode;
    }
    if (parsed.plotRate === 'low' || parsed.plotRate === 'full') {
      state.performanceControls.plotRate = parsed.plotRate;
    }
  } catch (_) {
    // Ignore malformed cache.
  }
}

function syncPerformanceControls() {
  if (el.previewModeSelect) {
    el.previewModeSelect.value = state.performanceControls.previewMode;
  }
  if (el.imageModeSelect) {
    el.imageModeSelect.value = state.performanceControls.imageMode;
  }
  if (el.plotRateSelect) {
    el.plotRateSelect.value = state.performanceControls.plotRate;
  }
}

function setInferenceStatus(message, kind = 'idle') {
  state.inference.statusText = String(message || '');
  state.inference.statusKind = kind || 'idle';
  if (!el.inferenceStatus) {
    return;
  }
  el.inferenceStatus.textContent = state.inference.statusText || 'Inference overlay idle.';
  el.inferenceStatus.dataset.kind = state.inference.statusKind;
}

function syncInferenceControls() {
  if (!el.inferenceYamlPath) {
    return;
  }

  el.inferenceYamlPath.value = state.inference.yamlPath;
  el.inferenceServerHost.value = state.inference.serverHost;
  el.inferenceServerPort.value = String(state.inference.serverPort);
  if (el.inferenceModeSelect) {
    el.inferenceModeSelect.value = sanitizeInferenceMode(state.inference.inferenceMode);
  }
  el.inferenceWarmupSteps.value = String(state.inference.warmupSteps);
  el.inferenceWarmupSteps.disabled = true;
  if (el.inferenceBatchSize) {
    el.inferenceBatchSize.value = String(state.inference.batchSize);
  }
  if (el.inferenceNoGripper) {
    el.inferenceNoGripper.checked = !!state.inference.noGripper;
  }

  const canRun = hasEpisodes() && !state.inference.running;
  if (el.runInferenceBtn) {
    el.runInferenceBtn.disabled = !canRun;
    el.runInferenceBtn.textContent = state.inference.running ? 'Running…' : 'Run Inference';
  }
  if (el.clearInferenceBtn) {
    el.clearInferenceBtn.disabled = state.inference.running || state.inference.history.length === 0;
  }
  if (el.inferenceStatus) {
    el.inferenceStatus.textContent = state.inference.statusText || 'Inference overlay idle.';
    el.inferenceStatus.dataset.kind = state.inference.statusKind || 'idle';
  }
}

function setProgressStatus(message, kind = 'idle') {
  state.progressGraph.statusText = String(message || '');
  state.progressGraph.statusKind = kind || 'idle';
  if (!el.progressStatus) {
    return;
  }
  el.progressStatus.textContent = state.progressGraph.statusText || 'Progress graph idle.';
  el.progressStatus.dataset.kind = state.progressGraph.statusKind;
}

function progressGraphCacheKey(episodeIndex, yamlPath, evalEvery) {
  return `${Number(episodeIndex)}::${String(yamlPath || '')}::${Math.max(1, Number(evalEvery) || 1)}`;
}

function currentProgressGraphEntry() {
  const key = state.progressGraph.currentKey;
  if (!key) {
    return null;
  }
  const entry = state.progressGraph.history[key];
  return entry && entry.episode_index === state.currentEpisode ? entry : null;
}

function syncProgressControls() {
  if (!el.progressYamlPath) {
    return;
  }
  el.progressYamlPath.value = state.progressGraph.yamlPath;
  el.progressServerHost.value = state.progressGraph.serverHost;
  el.progressServerPort.value = String(state.progressGraph.serverPort);
  el.progressEvalEvery.value = String(state.progressGraph.evalEvery);

  const canRun = hasEpisodes() && !state.progressGraph.running;
  if (el.runProgressBtn) {
    el.runProgressBtn.disabled = !canRun;
    el.runProgressBtn.textContent = state.progressGraph.running ? 'Running…' : 'Generate Graph';
  }
  const hasEntry = currentProgressGraphEntry() !== null;
  if (el.clearProgressBtn) {
    el.clearProgressBtn.disabled = state.progressGraph.running || !hasEntry;
  }
  if (el.progressStatus) {
    el.progressStatus.textContent = state.progressGraph.statusText || 'Progress graph idle.';
    el.progressStatus.dataset.kind = state.progressGraph.statusKind || 'idle';
  }
}

function progressCursorShape(frame, episodeLength) {
  const total = Math.max(1, Number(episodeLength) || 1);
  const x = clamp(Number(frame) || 0, 0, total - 1);
  return {
    type: 'line',
    x0: x,
    x1: x,
    y0: 0,
    y1: 1,
    yref: 'paper',
    line: { color: PROGRESS_CURSOR_COLOR, width: 2, dash: 'dot' },
  };
}

function renderProgressPlot(entry) {
  if (!entry || !el.progressPlot || typeof Plotly === 'undefined') {
    return;
  }
  const episodeLength = Math.max(1, Number(entry.episode_length) || 1);
  const data = [{
    x: entry.frames,
    y: entry.progress,
    mode: 'lines+markers',
    type: 'scatter',
    line: { width: 2, color: '#4c9aff' },
    marker: { size: 6, color: '#4c9aff' },
    name: 'progress',
    hovertemplate: 'frame %{x}<br>progress %{y:.3f}<extra></extra>',
  }];
  const layout = {
    xaxis: { title: 'Frame', range: [0, episodeLength - 1], zeroline: false },
    yaxis: { title: 'Progress', range: [0, 1], zeroline: false },
    shapes: [progressCursorShape(state.currentFrame, episodeLength)],
    margin: { l: 50, r: 20, t: 20, b: 40 },
    showlegend: false,
  };
  try {
    Plotly.react('progressPlot', data, layout, { responsive: true, displaylogo: false });
    state.progressGraph.rendered = true;
  } catch (err) {
    console.warn('progress plot render failed', err);
  }
}

function updateProgressCursor(frameIndex) {
  if (!state.progressGraph.rendered || !el.progressPlot || typeof Plotly === 'undefined') {
    return;
  }
  const entry = currentProgressGraphEntry();
  if (!entry) {
    return;
  }
  try {
    Plotly.relayout('progressPlot', { shapes: [progressCursorShape(frameIndex, entry.episode_length)] });
  } catch (_) {
    // If the plot div was removed or not yet initialized, ignore.
  }
}

function clearProgressPlot() {
  if (!el.progressPlot || typeof Plotly === 'undefined') {
    state.progressGraph.rendered = false;
    return;
  }
  try {
    Plotly.purge('progressPlot');
  } catch (_) {
    // Ignore errors from an uninitialized plot.
  }
  state.progressGraph.rendered = false;
}

function showProgressPlotForCurrentEpisode() {
  // Look for a cached entry matching the current (episode, yaml, evalEvery) tuple.
  const key = progressGraphCacheKey(
    state.currentEpisode,
    state.progressGraph.yamlPath,
    state.progressGraph.evalEvery,
  );
  const entry = state.progressGraph.history[key];
  if (entry && entry.episode_index === state.currentEpisode) {
    state.progressGraph.currentKey = key;
    renderProgressPlot(entry);
  } else {
    state.progressGraph.currentKey = null;
    clearProgressPlot();
  }
  syncProgressControls();
}

async function runProgressGraph() {
  if (!hasEpisodes() || !state.schema) {
    return;
  }

  state.progressGraph.yamlPath = (el.progressYamlPath?.value || '').trim();
  state.progressGraph.serverHost = (el.progressServerHost?.value || DEFAULT_PROGRESS_SERVER_HOST).trim() || DEFAULT_PROGRESS_SERVER_HOST;
  state.progressGraph.serverPort = sanitizeProgressPort(el.progressServerPort?.value);
  state.progressGraph.evalEvery = sanitizeProgressEvalEvery(el.progressEvalEvery?.value);
  syncProgressControls();

  if (!state.progressGraph.yamlPath) {
    setProgressStatus('Provide a YAML path before generating the graph.', 'error');
    syncProgressControls();
    return;
  }

  const key = progressGraphCacheKey(
    state.currentEpisode,
    state.progressGraph.yamlPath,
    state.progressGraph.evalEvery,
  );

  state.progressGraph.running = true;
  setProgressStatus(
    `Running progress inference across episode ${state.currentEpisode} (eval every ${state.progressGraph.evalEvery})…`,
    'pending',
  );
  syncProgressControls();

  try {
    const payload = await apiPost('/api/inference/progress_graph', {
      episode_index: state.currentEpisode,
      yaml_path: state.progressGraph.yamlPath,
      server_host: state.progressGraph.serverHost,
      server_port: state.progressGraph.serverPort,
      eval_every: state.progressGraph.evalEvery,
    });

    const frames = Array.isArray(payload.frames) ? payload.frames.map(Number) : [];
    const progress = Array.isArray(payload.progress) ? payload.progress.map(Number) : [];
    const episodeLength = Math.max(1, Number(payload.episode_length) || (state.schema?.length || frames[frames.length - 1] + 1 || 1));

    const entry = {
      episode_index: Number(payload.episode_index),
      episode_length: episodeLength,
      eval_every: Number(payload.eval_every) || state.progressGraph.evalEvery,
      yaml_path: String(payload.yaml_path || state.progressGraph.yamlPath),
      frames,
      progress,
      saved_at_ms: Date.now(),
    };
    state.progressGraph.history[key] = entry;
    state.progressGraph.currentKey = key;
    renderProgressPlot(entry);
    setProgressStatus(
      `Generated ${frames.length} point(s) across ${episodeLength} frames (eval every ${entry.eval_every}).`,
      'success',
    );
  } catch (err) {
    setProgressStatus(String(err), 'error');
  } finally {
    state.progressGraph.running = false;
    syncProgressControls();
  }
}

function clearProgressGraph(opts = {}) {
  const { clearStatus = true } = opts;
  state.progressGraph.history = {};
  state.progressGraph.currentKey = null;
  clearProgressPlot();
  if (clearStatus) {
    setProgressStatus('Progress graph idle.', 'idle');
  }
  syncProgressControls();
}

function currentInferenceOverlay() {
  const candidates = state.inference.history
    .filter((overlay) => overlay.episode_index === state.currentEpisode)
    .filter((overlay) => {
      const durationFrames = inferenceOverlayPredictionDurationFrames(overlay);
      if (durationFrames <= 0) {
        return false;
      }
      const frameDelta = state.currentFrame - overlay.frame_index;
      return frameDelta >= 0 && frameDelta < durationFrames;
    })
    .sort((a, b) => {
      if (a.frame_index !== b.frame_index) {
        return b.frame_index - a.frame_index;
      }
      return Number(b.saved_at_ms || 0) - Number(a.saved_at_ms || 0);
    });

  return candidates.length > 0 ? candidates[0] : null;
}

function inferenceOverlaySignature(overlay) {
  if (!overlay) {
    return '';
  }
  const batches = Array.isArray(overlay.batches) && overlay.batches.length > 0
    ? overlay.batches
    : [{ batch_index: 0, robots: overlay.robots || {} }];
  const batchParts = batches.map((batch, batchListIndex) => {
    const batchIndex = Number.isFinite(Number(batch.batch_index)) ? Number(batch.batch_index) : batchListIndex;
    const robotParts = ['robot0', 'robot1']
      .map((robot) => {
        const points = batch.robots && batch.robots[robot] && Array.isArray(batch.robots[robot].predicted_pos)
          ? batch.robots[robot].predicted_pos.length
          : 0;
        return points > 0 ? `${robot}:${points}` : '';
      })
      .filter(Boolean)
      .join(',');
    return `b${batchIndex}:${robotParts}`;
  }).join('|');
  const actionShape = Array.isArray(overlay.action_shape) ? overlay.action_shape.join('x') : '';
  const inferenceMode = sanitizeInferenceMode(overlay.inference_mode);
  return `${overlay.episode_index}|${overlay.frame_index}|${inferenceMode}|${actionShape}|${batchParts}`;
}

function inferenceOverlayPredictionLength(overlay) {
  if (!overlay) {
    return 0;
  }
  const fromBatchShapes = Array.isArray(overlay.batch_action_shapes)
    ? overlay.batch_action_shapes
        .map((shape) => (Array.isArray(shape) && shape.length > 0 ? Number(shape[0]) : 0))
        .filter((value) => Number.isFinite(value) && value > 0)
    : [];
  if (fromBatchShapes.length > 0) {
    return Math.max(...fromBatchShapes);
  }

  const batches = Array.isArray(overlay.batches) && overlay.batches.length > 0
    ? overlay.batches
    : [{ robots: overlay.robots || {} }];
  let maxLen = 0;
  for (const batch of batches) {
    for (const robot of ['robot0', 'robot1']) {
      const predicted = batch.robots && batch.robots[robot] && Array.isArray(batch.robots[robot].predicted_pos)
        ? batch.robots[robot].predicted_pos
        : [];
      maxLen = Math.max(maxLen, predicted.length);
    }
  }
  return maxLen;
}

function inferenceOverlayActionStride(overlay) {
  const stride = Math.floor(Number(overlay?.action_downsample_steps || 1));
  return Number.isFinite(stride) && stride > 0 ? stride : 1;
}

function inferenceOverlayPredictionDurationFrames(overlay) {
  const predictionLength = inferenceOverlayPredictionLength(overlay);
  if (predictionLength <= 0) {
    return 0;
  }
  return predictionLength * inferenceOverlayActionStride(overlay);
}

function saveInferenceOverlay(payload) {
  const saved = {
    ...payload,
    inference_mode: sanitizeInferenceMode(payload?.inference_mode),
    saved_at_ms: Date.now(),
  };
  const nextHistory = state.inference.history.filter(
    (overlay) => !(
      overlay.episode_index === saved.episode_index &&
      overlay.frame_index === saved.frame_index &&
      sanitizeInferenceMode(overlay.inference_mode) === saved.inference_mode
    ),
  );
  nextHistory.push(saved);
  nextHistory.sort((a, b) => {
    if (a.episode_index !== b.episode_index) {
      return a.episode_index - b.episode_index;
    }
    if (a.frame_index !== b.frame_index) {
      return a.frame_index - b.frame_index;
    }
    return Number(a.saved_at_ms || 0) - Number(b.saved_at_ms || 0);
  });
  state.inference.history = nextHistory;
}

function clearInferenceOverlay(opts = {}) {
  const { clearStatus = true, refreshPlot = true } = opts;
  state.inference.history = [];
  if (clearStatus) {
    setInferenceStatus('Inference overlay idle.', 'idle');
  }
  syncInferenceControls();
  if (refreshPlot) {
    requestPlotRefresh(true);
    flushPlotRefresh(performance.now(), true).catch(() => {});
  }
}

function buildInferenceOverlayData(overlay, frameIndex = state.currentFrame) {
  if (!overlay) {
    return { batches: [], averages: {} };
  }

  const rawBatches = Array.isArray(overlay.batches) && overlay.batches.length > 0
    ? overlay.batches
    : [{ batch_index: 0, robots: overlay.robots || {} }];
  const yOffsets = { robot0: ROBOT_HALF_SEPARATION_M, robot1: -ROBOT_HALF_SEPARATION_M };
  const batches = [];
  const actionStride = inferenceOverlayActionStride(overlay);
  const activeOffset = Math.floor(Math.max(0, frameIndex - overlay.frame_index) / actionStride);

  for (let batchListIndex = 0; batchListIndex < rawBatches.length; batchListIndex += 1) {
    const rawBatch = rawBatches[batchListIndex];
    const robots = {};

    for (const robot of ['robot0', 'robot1']) {
      const entry = rawBatch.robots ? rawBatch.robots[robot] : null;
      if (!entry || !Array.isArray(entry.predicted_pos) || entry.predicted_pos.length === 0) {
        continue;
      }

      const offsetY = Number(yOffsets[robot] || 0);
      const predicted = entry.predicted_pos
        .filter((point) => Array.isArray(point) && point.length >= 3)
        .map((point) => [Number(point[0]), Number(point[1]) + offsetY, Number(point[2])])
        .filter((point) => point.every((v) => Number.isFinite(v)));

      if (!predicted.length) {
        continue;
      }

      const currentPredicted = predicted[Math.min(activeOffset, predicted.length - 1)];
      robots[robot] = {
        predicted,
        currentPredicted,
      };
    }

    if (Object.keys(robots).length > 0) {
      batches.push({
        batchIndex: Number.isFinite(Number(rawBatch.batch_index)) ? Number(rawBatch.batch_index) : batchListIndex,
        robots,
      });
    }
  }

  const averages = {};
  for (const robot of ['robot0', 'robot1']) {
    const robotPredictions = batches
      .map((batch) => batch.robots[robot])
      .filter(Boolean)
      .map((entry) => entry.predicted);
    if (robotPredictions.length === 0) {
      continue;
    }

    const maxLen = Math.max(...robotPredictions.map((points) => points.length));
    const averagePredicted = [];
    for (let step = 0; step < maxLen; step += 1) {
      const stepPoints = robotPredictions
        .map((points) => (step < points.length ? points[step] : null))
        .filter(Boolean);
      if (stepPoints.length === 0) {
        continue;
      }
      const sum = stepPoints.reduce((acc, point) => {
        acc[0] += point[0];
        acc[1] += point[1];
        acc[2] += point[2];
        return acc;
      }, [0, 0, 0]);
      averagePredicted.push([
        sum[0] / stepPoints.length,
        sum[1] / stepPoints.length,
        sum[2] / stepPoints.length,
      ]);
    }

    if (averagePredicted.length > 0) {
      averages[robot] = {
        predicted: averagePredicted,
        currentPredicted: averagePredicted[Math.min(activeOffset, averagePredicted.length - 1)],
      };
    }
  }

  return { batches, averages };
}

function estimateLegendItemWidthPx(label) {
  const text = String(label || '');
  return clamp(44 + text.length * 7, LEGEND_MIN_ITEM_PX, LEGEND_MAX_ITEM_PX);
}

function getContainerWidthPx(graphEl, fallback = 760) {
  if (graphEl && Number.isFinite(graphEl.clientWidth) && graphEl.clientWidth > 0) {
    return graphEl.clientWidth;
  }
  if (graphEl && typeof graphEl.getBoundingClientRect === 'function') {
    const w = Number(graphEl.getBoundingClientRect().width);
    if (Number.isFinite(w) && w > 0) {
      return w;
    }
  }
  return fallback;
}

function countLegendRows(traces, graphEl) {
  const visible = (traces || []).filter((trace) => trace && trace.showlegend !== false);
  if (visible.length === 0) {
    return 0;
  }

  const width = Math.max(320, getContainerWidthPx(graphEl));
  let rows = 1;
  let rowWidth = 0;

  for (const trace of visible) {
    const needed = estimateLegendItemWidthPx(trace.name) + LEGEND_ITEM_PADDING_PX;
    if (rowWidth > 0 && rowWidth + needed > width) {
      rows += 1;
      rowWidth = needed;
    } else {
      rowWidth += needed;
    }
  }

  return rows;
}

function buildLegendLayout(traces, graphEl, compactBottom = LEGEND_COMPACT_BOTTOM_PX) {
  const rows = countLegendRows(traces, graphEl);
  if (rows <= 0) {
    return {
      legend: {
        orientation: 'h',
        x: 0,
        xanchor: 'left',
        y: -0.14,
        yanchor: 'top',
      },
      bottomMargin: compactBottom,
    };
  }

  const bottomMargin = LEGEND_BASE_BOTTOM_PX + LEGEND_GAP_FROM_PLOT_PX + rows * LEGEND_ROW_HEIGHT_PX;
  return {
    legend: {
      orientation: 'h',
      x: 0,
      xanchor: 'left',
      y: -0.14,
      yanchor: 'top',
    },
    bottomMargin,
  };
}

function predictionMemberColor(robot, batchIndex = 0) {
  const palette = PREDICTION_MEMBER_COLOR_PALETTES[robot];
  if (Array.isArray(palette) && palette.length > 0) {
    const normalizedIndex = Math.abs(Math.trunc(Number(batchIndex) || 0));
    return palette[normalizedIndex % palette.length];
  }
  return PREDICTION_MEMBER_COLORS[robot] || '#f4a261';
}

function niceStep(value) {
  const v = Math.max(1e-9, Number(value) || 0);
  const exp = Math.floor(Math.log10(v));
  const base = 10 ** exp;
  const frac = v / base;
  let niceFrac = 1;
  if (frac <= 1.0) {
    niceFrac = 1.0;
  } else if (frac <= 2.0) {
    niceFrac = 2.0;
  } else if (frac <= 2.5) {
    niceFrac = 2.5;
  } else if (frac <= 5.0) {
    niceFrac = 5.0;
  } else {
    niceFrac = 10.0;
  }
  return niceFrac * base;
}

function buildFloorGridTraces(axisRadius) {
  const radius = Math.max(0.1, Number(axisRadius) || 0);
  let step = niceStep(radius / 3.0);
  let halfLines = Math.floor(radius / step);
  halfLines = clamp(halfLines, 2, 4);
  step = radius / halfLines;

  const traces = [];
  const color = '#385168';
  for (let i = -halfLines; i <= halfLines; i += 1) {
    if (i === 0) {
      continue;
    }
    const pos = i * step;
    traces.push({
      x: [pos, pos],
      y: [-radius, radius],
      z: [0, 0],
      type: 'scatter3d',
      mode: 'lines',
      showlegend: false,
      hoverinfo: 'skip',
      line: { color, width: 1 },
    });
    traces.push({
      x: [-radius, radius],
      y: [pos, pos],
      z: [0, 0],
      type: 'scatter3d',
      mode: 'lines',
      showlegend: false,
      hoverinfo: 'skip',
      line: { color, width: 1 },
    });
  }
  return traces;
}

function buildAxisEndLabelTraces(axisRadius) {
  const r = Math.max(0.1, Number(axisRadius) || 0);
  const lowZ = Math.max(0.015, r * 0.02);
  return [
    {
      x: [r],
      y: [0],
      z: [lowZ],
      type: 'scatter3d',
      mode: 'markers+text',
      text: ['+X'],
      textposition: 'top center',
      textfont: { color: '#ffb7b7', size: 10 },
      marker: { size: 4, color: '#ff6b6b' },
      showlegend: false,
      hoverinfo: 'skip',
    },
    {
      x: [0],
      y: [r],
      z: [lowZ],
      type: 'scatter3d',
      mode: 'markers+text',
      text: ['+Y'],
      textposition: 'top center',
      textfont: { color: '#b7f1ce', size: 10 },
      marker: { size: 4, color: '#6bd49a' },
      showlegend: false,
      hoverinfo: 'skip',
    },
    {
      x: [0],
      y: [0],
      z: [r],
      type: 'scatter3d',
      mode: 'markers+text',
      text: ['+Z'],
      textposition: 'top center',
      textfont: { color: '#c7ddff', size: 10 },
      marker: { size: 4, color: '#6ca8ff' },
      showlegend: false,
      hoverinfo: 'skip',
    },
  ];
}

function currentEpisodeMeta() {
  return state.episodes.find((e) => e.episode_index === state.currentEpisode) || null;
}

function hasEpisodes() {
  return Array.isArray(state.episodes) && state.episodes.length > 0;
}

function renderEpisodeOptions() {
  el.episodeSelect.innerHTML = '';
  for (const ep of state.episodes) {
    const option = document.createElement('option');
    option.value = String(ep.episode_index);
    option.textContent = `Ep ${ep.episode_index} (${ep.length} frames)`;
    el.episodeSelect.appendChild(option);
  }
}

function updateDeleteButtonState() {
  if (!el.deleteEpisodeBtn) {
    return;
  }
  const canDelete = hasEpisodes() && !state.isDeletingEpisode;
  el.deleteEpisodeBtn.disabled = !canDelete;
  el.deleteEpisodeBtn.textContent = state.isDeletingEpisode ? 'Deleting...' : 'Delete Episode';
}

async function renderNoEpisodesState() {
  stopPlayback();
  state.episodeLoadToken += 1;
  state.schema = null;
  state.events = [];
  state.daggerSegments = [];
  state.timing = null;
  state.currentEpisode = 0;
  state.currentFrame = 0;
  state.targetFrame = 0;
  state.timeline.dragging = false;

  el.episodeSelect.value = '';
  el.episodeSelect.disabled = true;
  el.frameInput.value = '0';
  el.frameInput.max = '0';
  el.timelineSlider.value = '0';
  el.timelineSlider.max = '0';
  el.timeInput.value = '0';
  setMainFrameSource('');
  for (const tileState of state.previewTiles.values()) {
    cancelPreviewRequest(tileState);
    setPreviewTileSource(tileState, '');
  }
  state.previewTiles = new Map();
  el.cameraGrid.innerHTML = '<div class="camera-label">No episodes available.</div>';
  el.keyList.innerHTML = '';
  el.eventList.innerHTML = '';
  el.eventMarkers.innerHTML = '';
  updateTimelineInfo();

  resetMainPipeline('');
  resetChunkStore(state.signalStore, null, []);
  resetChunkStore(state.trajectoryStore, null, []);
  state.signalsPlot.initialized = false;
  state.signalsPlot.structureSig = '';
  state.signalsPlot.traceMeta = [];
  state.trajectoryPlot.initialized = false;
  state.trajectoryPlot.keys = [];
  state.trajectoryPlot.dynamicTraceIndices = {};
  state.trajectoryPlot.overlaySig = '';
  state.trajectoryCamera = null;
  state.trajectoryRelayoutBound = false;
  clearInferenceOverlay({ clearStatus: true, refreshPlot: false });

  if (typeof Plotly !== 'undefined') {
    await Plotly.react('signalsPlot', [], {
      paper_bgcolor: '#1b263b',
      plot_bgcolor: '#0a131f',
      font: { color: '#e0e1dd' },
      margin: { l: 40, r: 20, t: 20, b: 30 },
      annotations: [{ text: 'No episodes available', showarrow: false }],
      uirevision: 'signals-empty',
    }, { responsive: true });

    await Plotly.react('trajectory3d', [], {
      paper_bgcolor: '#1b263b',
      plot_bgcolor: '#0a131f',
      font: { color: '#e0e1dd' },
      margin: { l: 10, r: 10, t: 10, b: 10 },
      annotations: [{ text: 'No episodes available', showarrow: false }],
      uirevision: 'trajectory-empty',
    }, { responsive: true });
  }

  updateDeleteButtonState();
  updatePerfBadge();
}

function clampFrame(frame) {
  const length = Number(state.schema?.length || 0);
  if (length <= 0) {
    return 0;
  }
  return clamp(Math.round(Number(frame) || 0), 0, length - 1);
}

function resolvePlaybackMode() {
  if (state.playMode === 'fixed') {
    return 'fixed';
  }
  if (state.timing && state.timing.has_timestamps) {
    return 'timestamp';
  }
  return 'fixed';
}

function computeTargetFps() {
  const fpsCap = clamp(state.fpsCap, 1, 240);
  const speed = clamp(state.speedMultiplier, 0.1, 8.0);
  if (resolvePlaybackMode() === 'timestamp' && state.timing && state.timing.has_timestamps) {
    const medianDt = Number(state.timing.median_dt_sec || 0);
    if (medianDt > 0) {
      const datasetFps = (1.0 / medianDt) * speed;
      return clamp(Math.min(fpsCap, datasetFps), 1, 240);
    }
  }
  return clamp(fpsCap * speed, 1, 240);
}

function computeFrameIntervalMs() {
  return Math.max(1, 1000.0 / computeTargetFps());
}

function updateRateFromSamples(sample, countValue, nowMs) {
  if (sample.lastTs <= 0) {
    sample.lastTs = nowMs;
    sample.lastCount = countValue;
    return 0;
  }
  const elapsed = nowMs - sample.lastTs;
  if (elapsed < 500) {
    return null;
  }
  const delta = countValue - sample.lastCount;
  sample.lastTs = nowMs;
  sample.lastCount = countValue;
  return Math.max(0, (delta * 1000.0) / elapsed);
}

function refreshPerfRates(nowMs = performance.now()) {
  const m = updateRateFromSamples(state.perf.samples.main, state.perf.main_frame_displayed, nowMs);
  if (m !== null) {
    state.perf.effective_main_fps = m;
  }
  const p2 = updateRateFromSamples(state.perf.samples.plot2d, state.perf.plot2d_updates, nowMs);
  if (p2 !== null) {
    state.perf.effective_plot2d_fps = p2;
  }
  const p3 = updateRateFromSamples(state.perf.samples.plot3d, state.perf.plot3d_updates, nowMs);
  if (p3 !== null) {
    state.perf.effective_plot3d_fps = p3;
  }
}

function updateSummaryBadge() {
  if (!state.summary) {
    return;
  }
  const modeName = resolvePlaybackMode() === 'timestamp' ? 'Timestamp' : 'Fixed';
  const fps = computeTargetFps();
  const shown = state.perf.effective_main_fps || 0;
  el.summaryBadge.textContent = `${state.summary.format} • ${state.summary.episode_count} eps • ${modeName} • target ${fps.toFixed(1)} fps • shown ${shown.toFixed(1)} fps`;
}

function updatePerfBadge() {
  refreshPerfRates();
  state.perf.frame_lag = Math.max(0, state.targetFrame - state.currentFrame);
  const denom = state.mainPipeline.cacheHits + state.mainPipeline.fetches;
  state.perf.cache_hit_rate = denom > 0 ? state.mainPipeline.cacheHits / denom : 0;

  const modeShort = resolvePlaybackMode() === 'timestamp' ? 'TS' : 'FIX';
  const previewState = previewModeIsHideOthers()
    ? 'hide'
    : (state.previewAdaptive ? `x${state.previewStride}` : 'x1');
  const text = [
    `${modeShort}`,
    `m ${state.perf.effective_main_fps.toFixed(1)}`,
    `2d ${state.perf.effective_plot2d_fps.toFixed(1)}`,
    `3d ${state.perf.effective_plot3d_fps.toFixed(1)}`,
    `lag ${state.perf.frame_lag}`,
    `q ${state.perf.inflight_requests}`,
    `cache ${(state.perf.cache_hit_rate * 100).toFixed(0)}%`,
    `drop ${state.perf.main_frame_dropped}`,
    `prev ${previewState}`,
  ];
  el.perfBadge.textContent = text.join(' | ');
  updateSummaryBadge();
}

function updatePreviewPolicyFromBacklog() {
  const backlog = state.perf.inflight_requests;
  if (backlog >= PREVIEW_BACKLOG_HIGH) {
    state.previewAdaptive = true;
    state.previewStride = clamp(state.previewStride + 1, 1, PREVIEW_MAX_STRIDE);
    return;
  }

  if (backlog <= PREVIEW_BACKLOG_LOW && state.previewAdaptive) {
    state.previewStride = clamp(state.previewStride - 1, 1, PREVIEW_MAX_STRIDE);
    if (state.previewStride <= 1) {
      state.previewAdaptive = false;
      state.previewStride = 1;
    }
  }
}

function shouldUpdatePreviews(frameIndex, policy) {
  if (policy === 'full') {
    return true;
  }
  if (state.mainPipeline.inflight.size >= state.mainPipeline.maxInflight) {
    return false;
  }
  updatePreviewPolicyFromBacklog();
  if (!state.previewAdaptive) {
    return true;
  }
  return frameIndex % state.previewStride === 0;
}

function resolveMainFrameProfile() {
  if (!imageModeIsDynamic()) {
    return FULL_FRAME_PROFILE;
  }
  if (state.timeline.dragging || state.isPlaying) {
    return SCRUB_FRAME_PROFILE;
  }
  return FULL_FRAME_PROFILE;
}

function frameUrl(streamId, idx, modality, profile = FULL_FRAME_PROFILE) {
  const cmap = encodeURIComponent(el.colormapSelect.value || 'turbo');
  return `/api/episode/${state.currentEpisode}/frame?camera=${encodeURIComponent(streamId)}&idx=${idx}&modality=${modality}&colormap=${cmap}&profile=${encodeURIComponent(profile)}`;
}

function isDaggerActiveAtFrame(frameIndex) {
  if (!Array.isArray(state.daggerSegments) || state.daggerSegments.length === 0) {
    return false;
  }

  const idx = Math.round(Number(frameIndex) || 0);
  for (const seg of state.daggerSegments) {
    const start = Number(seg?.start);
    const end = Number(seg?.end);
    if (!Number.isFinite(start) || !Number.isFinite(end)) {
      continue;
    }
    if (idx < start) {
      break;
    }
    if (idx <= end) {
      return true;
    }
  }
  return false;
}

function updateTimelineInfo() {
  const length = Number(state.schema?.length || 0);
  const eventsNear = state.events.filter((e) => Math.abs(e.idx - state.currentFrame) <= 2).slice(0, 4);
  const marker = eventsNear.map((e) => e.label).join(' | ');
  const status = [];
  if (isDaggerActiveAtFrame(state.currentFrame)) {
    status.push('Dagger ACTIVE');
  }
  if (marker) {
    status.push(marker);
  }
  el.timelineInfo.textContent = `Frame ${state.currentFrame + 1}/${Math.max(1, length)}${status.length ? ` • ${status.join(' | ')}` : ''}`;
}

function setFrameUI(frameIndex) {
  const idx = clampFrame(frameIndex);
  state.currentFrame = idx;
  el.frameInput.value = String(idx);
  el.timelineSlider.value = String(idx);
  updateTimelineInfo();
  updateProgressCursor(idx);
}

function updatePreviewSelectionClasses() {
  const hideOthers = previewModeIsHideOthers();
  for (const [streamId, tileState] of state.previewTiles.entries()) {
    tileState.tile.classList.toggle('active', streamId === state.primaryStream);
    tileState.tile.classList.toggle('hidden-preview', hideOthers && streamId !== state.primaryStream);
    if (hideOthers && streamId !== state.primaryStream) {
      cancelPreviewRequest(tileState);
      setPreviewTileSource(tileState, '');
    }
    if (tileState.placeholder) {
      if (hideOthers && streamId !== state.primaryStream) {
        tileState.placeholder.textContent = `Click to switch to ${streamId}`;
      } else {
        tileState.placeholder.textContent = '';
      }
    }
  }
}

function cancelPreviewRequest(tileState) {
  if (tileState && tileState.controller) {
    tileState.controller.abort();
    tileState.controller = null;
  }
}

function setPreviewTileSource(tileState, src, ownsObjectUrl = false) {
  if (!tileState) {
    return;
  }
  const prevOwnedUrl = tileState.objectUrl;
  tileState.objectUrl = ownsObjectUrl ? (src || '') : '';
  tileState.img.src = src || '';
  if (prevOwnedUrl && prevOwnedUrl !== src) {
    revokeObjectUrl(prevOwnedUrl);
  }
}

function syncPrimaryPreviewToMain() {
  if (!previewModeIsHideOthers()) {
    return;
  }
  const tileState = state.previewTiles.get(state.primaryStream);
  if (!tileState) {
    return;
  }
  cancelPreviewRequest(tileState);
  setPreviewTileSource(tileState, state.mainFrameDisplay.objectUrl || '', false);
}

function requestPreviewFrame(tileState, url) {
  tileState.requestToken += 1;
  const token = tileState.requestToken;
  const timeoutMs = Math.max(180, Math.round(computeFrameIntervalMs() * 2.0));
  cancelPreviewRequest(tileState);
  const controller = new AbortController();
  tileState.controller = controller;

  state.perf.inflight_requests += 1;
  updatePerfBadge();

  let settled = false;
  const finish = (displayed, objectUrl = '') => {
    if (settled) {
      return;
    }
    settled = true;
    if (tileState.controller === controller) {
      tileState.controller = null;
    }
    state.perf.inflight_requests = Math.max(0, state.perf.inflight_requests - 1);
    if (displayed) {
      state.perf.preview_frame_displayed += 1;
      setPreviewTileSource(tileState, objectUrl, true);
    } else if (objectUrl) {
      revokeObjectUrl(objectUrl);
    }
    updatePreviewPolicyFromBacklog();
    updatePerfBadge();
  };

  fetchImageObjectUrl(url, controller, timeoutMs)
    .then((objectUrl) => {
      if (token !== tileState.requestToken || controller.signal.aborted) {
        finish(false, objectUrl);
        return;
      }
      finish(true, objectUrl);
    })
    .catch((err) => {
      if (!isAbortError(err)) {
        console.debug('Preview frame request failed:', err);
      }
      finish(false);
    });
}

function updatePreviewFrames(frameIndex, policy = 'auto') {
  if (!state.schema || !state.schema.cameras || state.schema.cameras.length === 0) {
    el.cameraGrid.innerHTML = '<div class="camera-label">No camera streams available.</div>';
    return;
  }

  if (!shouldUpdatePreviews(frameIndex, policy)) {
    updatePerfBadge();
    return;
  }

  if (previewModeIsHideOthers()) {
    updatePreviewSelectionClasses();
    syncPrimaryPreviewToMain();
    updatePerfBadge();
    return;
  }

  for (const stream of state.schema.cameras) {
    const tileState = state.previewTiles.get(stream.stream_id);
    if (!tileState) {
      continue;
    }
    const url = frameUrl(stream.stream_id, frameIndex, 'rgb', PREVIEW_FRAME_PROFILE);
    requestPreviewFrame(tileState, url);
  }

  updatePreviewSelectionClasses();
}

function getPrimarySchema() {
  const streams = state.schema?.cameras || [];
  if (!streams.length) {
    return null;
  }
  let schema = streams.find((s) => s.stream_id === state.primaryStream) || streams[0];
  if (!schema) {
    return null;
  }
  if (!state.primaryStream || !streams.some((s) => s.stream_id === state.primaryStream)) {
    state.primaryStream = schema.stream_id;
    el.primaryStreamSelect.value = state.primaryStream;
  }
  return schema;
}

function getMainModality(primarySchema) {
  const wantsDepth = Boolean(el.depthToggle.checked);
  if (wantsDepth && primarySchema && primarySchema.depth_source) {
    return 'depth';
  }
  return 'rgb';
}

function cancelHighQualityRefresh() {
  if (state.mainFrameDisplay.highQualityTimer) {
    window.clearTimeout(state.mainFrameDisplay.highQualityTimer);
    state.mainFrameDisplay.highQualityTimer = null;
  }
  if (state.mainFrameDisplay.highQualityController) {
    state.mainFrameDisplay.highQualityController.abort();
    state.mainFrameDisplay.highQualityController = null;
  }
}

function setMainFrameSource(src) {
  const nextSrc = src || '';
  const prevSrc = state.mainFrameDisplay.objectUrl;
  state.mainFrameDisplay.objectUrl = nextSrc;
  el.mainFrame.src = nextSrc;
  if (previewModeIsHideOthers()) {
    syncPrimaryPreviewToMain();
  }
  if (prevSrc && prevSrc !== nextSrc) {
    revokeObjectUrl(prevSrc);
  }
}

function scheduleHighQualityRefresh(frameIndex, streamId, modality) {
  cancelHighQualityRefresh();
  if (!imageModeIsDynamic()) {
    return;
  }
  if (state.timeline.dragging || state.isPlaying) {
    return;
  }

  state.mainFrameDisplay.highQualityTimer = window.setTimeout(() => {
    state.mainFrameDisplay.highQualityTimer = null;
    const primarySchema = getPrimarySchema();
    if (state.currentFrame !== frameIndex || state.primaryStream !== streamId) {
      return;
    }
    if (!primarySchema || getMainModality(primarySchema) !== modality) {
      return;
    }
    if (resolveMainFrameProfile() !== FULL_FRAME_PROFILE) {
      return;
    }

    const controller = new AbortController();
    state.mainFrameDisplay.highQualityController = controller;
    const url = frameUrl(streamId, frameIndex, modality, FULL_FRAME_PROFILE);
    fetchImageObjectUrl(url, controller, MAIN_FRAME_TIMEOUT_MS)
      .then((objectUrl) => {
        if (controller.signal.aborted) {
          revokeObjectUrl(objectUrl);
          return;
        }
        const latestPrimarySchema = getPrimarySchema();
        if (
          state.currentFrame !== frameIndex
          || state.primaryStream !== streamId
          || state.timeline.dragging
          || state.isPlaying
          || !latestPrimarySchema
          || getMainModality(latestPrimarySchema) !== modality
        ) {
          revokeObjectUrl(objectUrl);
          return;
        }
        setMainFrameSource(objectUrl);
      })
      .catch((err) => {
        if (!isAbortError(err)) {
          console.debug('High-quality frame refresh failed:', err);
        }
      })
      .finally(() => {
        if (state.mainFrameDisplay.highQualityController === controller) {
          state.mainFrameDisplay.highQualityController = null;
        }
      });
  }, HIGH_QUALITY_REFRESH_DELAY_MS);
}

function computeMainPipelineKey() {
  const primarySchema = getPrimarySchema();
  if (!primarySchema) {
    return '';
  }
  const modality = getMainModality(primarySchema);
  const profile = resolveMainFrameProfile();
  const cmap = modality === 'depth' ? (el.colormapSelect.value || 'turbo') : '-';
  return `${state.currentEpisode}|${primarySchema.stream_id}|${modality}|${profile}|${cmap}`;
}

function clearMainPipelineQueue() {
  for (const entry of state.mainPipeline.queue.values()) {
    if (entry.controller) {
      entry.controller.abort();
    }
    if (entry.blobUrl && entry.blobUrl !== state.mainFrameDisplay.objectUrl) {
      revokeObjectUrl(entry.blobUrl);
    }
  }
  state.mainPipeline.queue.clear();
  state.mainPipeline.pendingOrder = [];
  state.mainPipeline.inflight.clear();
}

function resetMainPipeline(newKey = null) {
  cancelHighQualityRefresh();
  state.mainPipeline.token += 1;
  if (newKey !== null) {
    state.mainPipeline.key = newKey;
  }
  clearMainPipelineQueue();
}

function ensureMainPipelineContext() {
  const nextKey = computeMainPipelineKey();
  if (nextKey !== state.mainPipeline.key) {
    resetMainPipeline(nextKey);
  }
}

function trimMainPipelineQueue() {
  const length = Number(state.schema?.length || 0);
  if (length <= 0) {
    return;
  }

  const low = Math.max(0, state.currentFrame - 4);
  const high = Math.min(length - 1, Math.max(state.currentFrame, state.targetFrame) + state.mainPipeline.lookahead + 4);

  for (const [idx, entry] of state.mainPipeline.queue.entries()) {
    if (idx < low || idx > high) {
      if (entry.controller) {
        entry.controller.abort();
      }
      if (entry.blobUrl && entry.blobUrl !== state.mainFrameDisplay.objectUrl) {
        revokeObjectUrl(entry.blobUrl);
      }
      state.mainPipeline.queue.delete(idx);
    }
  }

  state.mainPipeline.pendingOrder = state.mainPipeline.pendingOrder.filter((idx) => idx >= low && idx <= high);
}

function queuePendingIndex(idx, priority = false) {
  if (priority) {
    state.mainPipeline.pendingOrder = [idx, ...state.mainPipeline.pendingOrder.filter((v) => v !== idx)];
  } else if (!state.mainPipeline.pendingOrder.includes(idx)) {
    state.mainPipeline.pendingOrder.push(idx);
  }
}

function makeMainFrameEntry(idx, url, streamId, modality, profile) {
  return {
    idx,
    url,
    streamId,
    modality,
    profile,
    status: 'queued',
    promise: null,
    controller: null,
    blobUrl: '',
  };
}

function startMainFetch(entry) {
  const token = state.mainPipeline.token;
  entry.status = 'pending';
  const controller = new AbortController();
  entry.controller = controller;
  state.mainPipeline.inflight.add(entry.idx);
  state.mainPipeline.fetches += 1;
  state.perf.main_frame_requested += 1;
  state.perf.inflight_requests += 1;
  updatePerfBadge();

  entry.promise = new Promise((resolve) => {
    let settled = false;

    const finish = (ok) => {
      if (settled) {
        return;
      }
      settled = true;
      state.mainPipeline.inflight.delete(entry.idx);
      if (entry.controller === controller) {
        entry.controller = null;
      }
      state.perf.inflight_requests = Math.max(0, state.perf.inflight_requests - 1);
      if (token !== state.mainPipeline.token) {
        resolve(false);
      } else {
        if (ok) {
          entry.status = 'ready';
        } else {
          entry.status = 'error';
        }
        resolve(ok);
      }
      updatePreviewPolicyFromBacklog();
      updatePerfBadge();
      drainMainPipeline();
    };

    fetchImageObjectUrl(entry.url, controller, MAIN_FRAME_TIMEOUT_MS)
      .then((objectUrl) => {
        if (token !== state.mainPipeline.token || controller.signal.aborted) {
          revokeObjectUrl(objectUrl);
          finish(false);
          return;
        }
        entry.blobUrl = objectUrl;
        finish(true);
      })
      .catch((err) => {
        if (!isAbortError(err)) {
          console.debug('Main frame request failed:', err);
        }
        finish(false);
      });
  });

  return entry.promise;
}

function drainMainPipeline() {
  while (
    state.mainPipeline.inflight.size < state.mainPipeline.maxInflight
    && state.mainPipeline.pendingOrder.length > 0
  ) {
    const idx = state.mainPipeline.pendingOrder.shift();
    const entry = state.mainPipeline.queue.get(idx);
    if (!entry) {
      continue;
    }
    if (entry.status !== 'queued') {
      continue;
    }
    startMainFetch(entry);
  }
}

function queueMainFrame(idx, priority = false) {
  const length = Number(state.schema?.length || 0);
  if (length <= 0) {
    return null;
  }

  const frameIdx = clampFrame(idx);
  ensureMainPipelineContext();
  const entryExisting = state.mainPipeline.queue.get(frameIdx);
  if (entryExisting) {
    if (entryExisting.status === 'ready' || entryExisting.status === 'pending') {
      state.mainPipeline.cacheHits += 1;
      updatePerfBadge();
      return entryExisting;
    }
    queuePendingIndex(frameIdx, priority);
    drainMainPipeline();
    return entryExisting;
  }

  const primarySchema = getPrimarySchema();
  if (!primarySchema) {
    return null;
  }
  const modality = getMainModality(primarySchema);
  const profile = resolveMainFrameProfile();
  const entry = makeMainFrameEntry(
    frameIdx,
    frameUrl(primarySchema.stream_id, frameIdx, modality, profile),
    primarySchema.stream_id,
    modality,
    profile,
  );
  state.mainPipeline.queue.set(frameIdx, entry);
  queuePendingIndex(frameIdx, priority);
  drainMainPipeline();
  return entry;
}

async function waitForMainFrameReady(idx, timeoutMs = 1800) {
  const entry = queueMainFrame(idx, true);
  if (!entry) {
    return false;
  }
  if (entry.status === 'ready') {
    return true;
  }
  if (!entry.promise) {
    drainMainPipeline();
  }
  const p = entry.promise || Promise.resolve(false);
  const timeout = new Promise((resolve) => {
    window.setTimeout(() => resolve(false), timeoutMs);
  });
  await Promise.race([p, timeout]);

  const latest = state.mainPipeline.queue.get(clampFrame(idx));
  return Boolean(latest && latest.status === 'ready');
}

function pickReadyMainCandidate() {
  const length = Number(state.schema?.length || 0);
  if (length <= 0) {
    return null;
  }
  const maxIdx = clamp(state.targetFrame, 0, length - 1);
  const minIdx = clamp(state.currentFrame + 1, 0, length - 1);
  if (maxIdx < minIdx) {
    return null;
  }

  for (let idx = maxIdx; idx >= minIdx; idx -= 1) {
    const entry = state.mainPipeline.queue.get(idx);
    if (entry && entry.status === 'ready') {
      return idx;
    }
  }
  return null;
}

function scheduleMainLookahead() {
  const length = Number(state.schema?.length || 0);
  if (length <= 0) {
    return;
  }

  ensureMainPipelineContext();

  const base = Math.max(state.currentFrame, state.targetFrame);
  const start = clamp(state.currentFrame + 1, 0, length - 1);
  const end = clamp(base + state.mainPipeline.lookahead, 0, length - 1);

  if (start > end) {
    return;
  }

  for (let idx = start; idx <= end; idx += 1) {
    queueMainFrame(idx, idx <= start + 1);
  }
  trimMainPipelineQueue();
}

function applyDisplayedFrame(frameIndex, opts = { refreshPlots: true, previewPolicy: 'auto', allowLookahead: true }) {
  const idx = clampFrame(frameIndex);
  const entry = state.mainPipeline.queue.get(idx);
  if (!entry || entry.status !== 'ready' || !entry.blobUrl) {
    return false;
  }

  if (idx > state.currentFrame + 1) {
    state.perf.main_frame_dropped += (idx - state.currentFrame - 1);
  }

  setMainFrameSource(entry.blobUrl);

  state.perf.main_frame_displayed += 1;
  setFrameUI(idx);
  state.targetFrame = opts.latestOnly ? idx : Math.max(state.targetFrame, idx);

  updatePreviewFrames(idx, opts.previewPolicy || 'auto');

  if (opts.refreshPlots !== false) {
    requestPlotRefresh(false);
  }

  if (opts.allowLookahead) {
    scheduleMainLookahead();
  }
  trimMainPipelineQueue();
  if (imageModeIsDynamic() && entry.profile === SCRUB_FRAME_PROFILE) {
    scheduleHighQualityRefresh(idx, entry.streamId, entry.modality);
  } else {
    cancelHighQualityRefresh();
  }
  updatePerfBadge();
  return true;
}

async function showFrameNow(frameIndex, opts = { refreshPlots: true, previewPolicy: 'full', latestOnly: false, allowLookahead: false, forcePlotRefresh: true }) {
  const idx = clampFrame(frameIndex);
  state.targetFrame = idx;
  if (opts.latestOnly) {
    resetMainPipeline(computeMainPipelineKey());
  } else {
    ensureMainPipelineContext();
  }

  queueMainFrame(idx, true);
  if (opts.allowLookahead) {
    scheduleMainLookahead();
  }

  const ready = await waitForMainFrameReady(idx);
  if (!ready) {
    return false;
  }

  const applied = applyDisplayedFrame(idx, {
    refreshPlots: opts.refreshPlots !== false,
    previewPolicy: opts.previewPolicy || 'full',
    latestOnly: opts.latestOnly === true,
    allowLookahead: opts.allowLookahead === true,
  });
  if (applied && opts.refreshPlots !== false) {
    await flushPlotRefresh(performance.now(), opts.forcePlotRefresh !== false);
  }
  return applied;
}

function stopPlayback() {
  const wasPlaying = state.isPlaying;
  state.isPlaying = false;
  el.playBtn.textContent = 'Play';
  if (state.playHandle) {
    cancelAnimationFrame(state.playHandle);
    state.playHandle = null;
  }
  resetMainPipeline(computeMainPipelineKey());
  if (wasPlaying) {
    requestPlotRefresh(true);
    flushPlotRefresh(performance.now(), true).catch(() => {});
    const primarySchema = getPrimarySchema();
    if (primarySchema && state.primaryStream) {
      scheduleHighQualityRefresh(state.currentFrame, state.primaryStream, getMainModality(primarySchema));
    }
  }
  updatePerfBadge();
}

function startPlayback() {
  if (state.isPlaying) {
    return;
  }
  const episode = currentEpisodeMeta();
  if (!episode || episode.length <= 1) {
    return;
  }

  state.isPlaying = true;
  state.playAnchorFrame = state.currentFrame;
  state.playAnchorMs = performance.now();
  state.targetFrame = state.currentFrame;
  cancelHighQualityRefresh();

  ensureMainPipelineContext();
  scheduleMainLookahead();

  el.playBtn.textContent = 'Pause';
  state.playHandle = requestAnimationFrame(playbackStep);
  updatePerfBadge();
}

function updateTargetFrameFromClock(nowMs) {
  const episode = currentEpisodeMeta();
  if (!episode) {
    return;
  }
  const computedInterval = computeFrameIntervalMs();
  const intervalMs = Number.isFinite(computedInterval) && computedInterval > 0
    ? computedInterval
    : PLAYBACK_TARGET_MS;
  const elapsedFrames = Math.floor((nowMs - state.playAnchorMs) / intervalMs);
  const desired = clamp(state.playAnchorFrame + elapsedFrames, 0, episode.length - 1);
  if (desired > state.targetFrame) {
    state.targetFrame = desired;
  }
}

function playbackStep(nowMs) {
  if (!state.isPlaying) {
    return;
  }

  const episode = currentEpisodeMeta();
  if (!episode) {
    stopPlayback();
    return;
  }

  if (state.currentFrame >= episode.length - 1) {
    stopPlayback();
    return;
  }

  updateTargetFrameFromClock(nowMs);
  scheduleMainLookahead();

  const candidate = pickReadyMainCandidate();
  if (candidate !== null) {
    applyDisplayedFrame(candidate, {
      refreshPlots: false,
      previewPolicy: 'auto',
      latestOnly: false,
      allowLookahead: true,
    });
    requestPlotRefresh(false);
  } else {
    if (state.targetFrame - state.currentFrame > MAX_FRAME_LAG) {
      queueMainFrame(state.targetFrame, true);
    }
    queueMainFrame(state.currentFrame + 1, true);
  }

  flushPlotRefresh(nowMs, false).catch(() => {});

  if (state.currentFrame >= episode.length - 1) {
    stopPlayback();
    return;
  }

  state.playHandle = requestAnimationFrame(playbackStep);
}

async function setFrame(frame, opts = { refreshPlots: true, previewPolicy: 'full', latestOnly: false, allowLookahead: false, forcePlotRefresh: true }) {
  const idx = clampFrame(frame);
  state.targetFrame = idx;

  if (state.isPlaying) {
    state.playAnchorFrame = idx;
    state.playAnchorMs = performance.now();
  }

  const ok = await showFrameNow(idx, {
    refreshPlots: opts.refreshPlots !== false,
    previewPolicy: opts.previewPolicy || 'full',
    latestOnly: opts.latestOnly === true,
    allowLookahead: opts.allowLookahead === true,
    forcePlotRefresh: opts.forcePlotRefresh !== false,
  });

  if (!ok && opts.refreshPlots !== false && opts.latestOnly !== true) {
    requestPlotRefresh(true);
    await flushPlotRefresh(performance.now(), opts.forcePlotRefresh !== false);
  }
}

function onTimeJump() {
  if (!state.lastSignalsTs || state.lastSignalsTs.length === 0) {
    return;
  }

  const t = Number(el.timeInput.value);
  if (!Number.isFinite(t)) {
    return;
  }

  let bestLocal = 0;
  let bestDist = Infinity;
  for (let i = 0; i < state.lastSignalsTs.length; i += 1) {
    const d = Math.abs(state.lastSignalsTs[i] - t);
    if (d < bestDist) {
      bestDist = d;
      bestLocal = i;
    }
  }

  const frame = clampFrame(state.lastSignalsWindowStart + bestLocal);
  setFrame(frame, {
    refreshPlots: true,
    previewPolicy: 'full',
    latestOnly: true,
    allowLookahead: false,
    forcePlotRefresh: true,
  }).catch(() => {});
}

function frameToTimelinePercent(frameIndex, length) {
  const total = Math.max(0, Math.floor(Number(length) || 0));
  if (total <= 1) {
    return 0;
  }
  const idx = clamp(Math.round(Number(frameIndex) || 0), 0, total - 1);
  return (idx / (total - 1)) * 100;
}

function renderEventMarkers() {
  el.eventMarkers.innerHTML = '';
  const length = Number(state.schema?.length || 0);
  if (length <= 1) {
    return;
  }

  for (const seg of state.daggerSegments) {
    const start = clamp(Number(seg.start ?? 0), 0, length - 1);
    const end = clamp(Number(seg.end ?? start), start, length - 1);
    const startPct = frameToTimelinePercent(start, length);
    const endPct = frameToTimelinePercent(end, length);

    const block = document.createElement('div');
    block.className = 'event-block dagger';
    block.style.left = `${startPct}%`;
    block.style.width = `${Math.max(0, endPct - startPct)}%`;
    block.title = `dagger @ [${start}, ${end}]`;
    el.eventMarkers.appendChild(block);
  }

  for (const evt of state.events.slice(0, 500)) {
    const dot = document.createElement('div');
    dot.className = `event-dot ${evt.type || 'info'}`;
    dot.style.left = `${frameToTimelinePercent(evt.idx, length)}%`;
    dot.title = `${evt.label} @ ${evt.idx}`;
    dot.addEventListener('click', () => {
      setFrame(evt.idx, {
        refreshPlots: true,
        previewPolicy: 'full',
        latestOnly: true,
        allowLookahead: false,
        forcePlotRefresh: true,
      }).catch(() => {});
    });
    el.eventMarkers.appendChild(dot);
  }
}

function buildBinarySegments(values, threshold = 0.5) {
  const segments = [];
  let start = -1;

  for (let i = 0; i < values.length; i += 1) {
    const v = Number(values[i]);
    const active = Number.isFinite(v) && v > threshold;
    if (active && start < 0) {
      start = i;
    } else if (!active && start >= 0) {
      segments.push({ start, end: i - 1 });
      start = -1;
    }
  }

  if (start >= 0) {
    segments.push({ start, end: values.length - 1 });
  }

  return segments;
}

async function fetchDaggerSegments(episodeIndex, length, schema) {
  if (length <= 0) {
    return [];
  }

  const hasDaggerKey = (schema?.keys || []).some((k) => k?.key === 'dagger');
  if (!hasDaggerKey) {
    return [];
  }

  try {
    const payload = await apiGet(
      `/api/episode/${episodeIndex}/signals?keys=dagger&start=0&end=${length}&stride=1`,
    );
    const channels = payload?.series?.dagger;
    if (!Array.isArray(channels) || channels.length === 0) {
      return [];
    }
    const values = channels[0]?.values;
    if (!Array.isArray(values) || values.length === 0) {
      return [];
    }
    return buildBinarySegments(values, 0.5);
  } catch (err) {
    console.warn('Failed to load dagger timeline segments:', err);
    return [];
  }
}

function renderEventList() {
  el.eventList.innerHTML = '';
  for (const evt of state.events.slice(0, 200)) {
    const li = document.createElement('li');
    li.textContent = `[${evt.type}] #${evt.idx} ${evt.label}`;
    li.addEventListener('click', () => {
      setFrame(evt.idx, {
        refreshPlots: true,
        previewPolicy: 'full',
        latestOnly: true,
        allowLookahead: false,
        forcePlotRefresh: true,
      }).catch(() => {});
    });
    el.eventList.appendChild(li);
  }
}

function buildStreamSelectors() {
  const streams = state.schema?.cameras || [];
  el.primaryStreamSelect.innerHTML = '';

  if (streams.length === 0) {
    state.primaryStream = null;
    return;
  }

  for (const stream of streams) {
    const option = document.createElement('option');
    option.value = stream.stream_id;
    option.textContent = stream.stream_id;
    el.primaryStreamSelect.appendChild(option);
  }

  state.primaryStream = streams[0].stream_id;
  el.primaryStreamSelect.value = state.primaryStream;
}

function buildCameraGrid() {
  for (const tileState of state.previewTiles.values()) {
    cancelPreviewRequest(tileState);
    setPreviewTileSource(tileState, '');
  }
  state.previewTiles = new Map();
  el.cameraGrid.innerHTML = '';

  const streams = state.schema?.cameras || [];
  if (streams.length === 0) {
    el.cameraGrid.innerHTML = '<div class="camera-label">No camera streams available.</div>';
    return;
  }

  for (const stream of streams) {
    const tile = document.createElement('div');
    tile.className = 'camera-tile' + (stream.stream_id === state.primaryStream ? ' active' : '');

    const img = document.createElement('img');
    img.alt = stream.stream_id;

    const placeholder = document.createElement('div');
    placeholder.className = 'camera-placeholder';

    const lbl = document.createElement('div');
    lbl.className = 'camera-label';
    lbl.textContent = stream.stream_id + (stream.depth_source ? ' • depth' : '');

    tile.appendChild(img);
    tile.appendChild(placeholder);
    tile.appendChild(lbl);
    tile.addEventListener('click', () => {
      state.primaryStream = stream.stream_id;
      el.primaryStreamSelect.value = stream.stream_id;
      updatePreviewSelectionClasses();
      resetMainPipeline(computeMainPipelineKey());
      setFrame(state.currentFrame, {
        refreshPlots: false,
        previewPolicy: 'full',
        latestOnly: true,
        allowLookahead: false,
        forcePlotRefresh: false,
      }).catch(() => {});
    });

    el.cameraGrid.appendChild(tile);
    state.previewTiles.set(stream.stream_id, {
      streamId: stream.stream_id,
      tile,
      img,
      placeholder,
      requestToken: 0,
      controller: null,
      objectUrl: '',
    });
  }

  updatePreviewSelectionClasses();
}

function buildKeySelector() {
  el.keyList.innerHTML = '';
  state.selectedKeys.clear();

  const keys = (state.schema?.keys || []).filter((k) => k.graphable);
  for (const keyInfo of keys) {
    const row = document.createElement('label');
    row.className = 'key-item';

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = keyInfo.key;
    cb.addEventListener('change', () => {
      if (cb.checked) {
        state.selectedKeys.add(cb.value);
      } else {
        state.selectedKeys.delete(cb.value);
      }
      onSelectedKeysChanged();
    });

    const text = document.createElement('span');
    text.textContent = keyInfo.key;

    row.appendChild(cb);
    row.appendChild(text);
    el.keyList.appendChild(row);
  }
}

function selectedKeysArray() {
  return Array.from(state.selectedKeys).sort();
}

function keysSignature(keys) {
  return keys.join('|');
}

function resetChunkStore(store, episodeIndex, keys) {
  store.episode = episodeIndex;
  store.keys = keys.slice();
  store.keysSig = keysSignature(store.keys);
  store.token += 1;
  store.cache.clear();
  store.pending.clear();
}

function ensureChunkStoreConfig(store, episodeIndex, keys) {
  const sig = keysSignature(keys);
  if (store.episode !== episodeIndex || store.keysSig !== sig) {
    resetChunkStore(store, episodeIndex, keys);
  }
}

function chunkStartsForRange(start, end, chunkSize) {
  if (end <= start) {
    return [];
  }
  const starts = [];
  const first = Math.floor(start / chunkSize) * chunkSize;
  for (let s = first; s < end; s += chunkSize) {
    starts.push(s);
  }
  return starts;
}

function touchChunk(store, chunkStart) {
  const value = store.cache.get(chunkStart);
  if (!value) {
    return null;
  }
  store.cache.delete(chunkStart);
  store.cache.set(chunkStart, value);
  return value;
}

function setChunk(store, chunkStart, payload) {
  store.cache.set(chunkStart, payload);
  while (store.cache.size > store.maxChunks) {
    const oldest = store.cache.keys().next();
    if (oldest.done) {
      break;
    }
    store.cache.delete(oldest.value);
  }
}

function hasChunkRange(store, start, end) {
  const starts = chunkStartsForRange(start, end, store.chunkSize);
  return starts.every((s) => store.cache.has(s));
}

function getChunk(store, chunkStart) {
  return touchChunk(store, chunkStart);
}

function scheduleChunkFetch(store, chunkStart) {
  if (store.cache.has(chunkStart) || store.pending.has(chunkStart) || store.keys.length === 0) {
    return;
  }

  const episodeAtRequest = store.episode;
  const tokenAtRequest = store.token;
  const length = Number(state.schema?.length || 0);
  const end = Math.min(length, chunkStart + store.chunkSize);
  if (end <= chunkStart) {
    return;
  }

  const query = new URLSearchParams({
    keys: store.keys.join(','),
    start: String(chunkStart),
    end: String(end),
    stride: '1',
  });

  const promise = apiGet(`/api/episode/${episodeAtRequest}/signals?${query.toString()}`)
    .then((payload) => {
      if (store.token !== tokenAtRequest) {
        return;
      }
      if (store.episode !== episodeAtRequest || episodeAtRequest !== state.currentEpisode) {
        return;
      }
      setChunk(store, chunkStart, payload);
    })
    .catch(() => {
      // keep previous data on fetch failures
    })
    .finally(() => {
      store.pending.delete(chunkStart);
      requestPlotRefresh(false);
      if (!state.isPlaying) {
        flushPlotRefresh(performance.now(), false).catch(() => {});
      }
    });

  store.pending.set(chunkStart, promise);
}

function ensureChunksForRange(store, start, end) {
  if (end <= start) {
    return;
  }
  for (const chunkStart of chunkStartsForRange(start, end, store.chunkSize)) {
    scheduleChunkFetch(store, chunkStart);
  }
}

function buildSignalTraceMeta(keys) {
  const keyMeta = new Map((state.schema?.keys || []).map((k) => [k.key, k]));
  const traces = [];
  for (const key of keys) {
    const info = keyMeta.get(key);
    if (!info) {
      continue;
    }
    const shape = Array.isArray(info.shape) ? info.shape : [];
    const channels = shape.length >= 2 ? Math.max(1, Number(shape[1]) || 1) : 1;
    for (let i = 0; i < channels; i += 1) {
      traces.push({
        key,
        channel: i,
        name: channels === 1 ? key : `${key}[${i}]`,
      });
    }
  }
  return traces;
}

function buildSignalsWindowFromCache(frameIndex) {
  const end = frameIndex + 1;
  const start = Math.max(0, end - state.historyFrames);
  const keys = selectedKeysArray();
  const meta = state.signalsPlot.traceMeta;

  if (meta.length === 0) {
    return {
      ready: true,
      start,
      end,
      timestamps: [],
      valuesByTrace: [],
    };
  }

  if (!hasChunkRange(state.signalStore, start, end)) {
    return { ready: false, start, end };
  }

  const valuesByTrace = meta.map(() => []);
  const timestamps = [];

  for (let globalIdx = start; globalIdx < end; globalIdx += 1) {
    const chunkStart = Math.floor(globalIdx / state.signalStore.chunkSize) * state.signalStore.chunkSize;
    const chunk = getChunk(state.signalStore, chunkStart);
    if (!chunk) {
      return { ready: false, start, end };
    }
    const localIdx = globalIdx - Number(chunk.start || chunkStart);
    if (localIdx < 0) {
      continue;
    }

    const t = Array.isArray(chunk.timestamps) && localIdx < chunk.timestamps.length
      ? Number(chunk.timestamps[localIdx])
      : Number(globalIdx);
    timestamps.push(t);

    for (let ti = 0; ti < meta.length; ti += 1) {
      const trace = meta[ti];
      const channels = chunk.series ? chunk.series[trace.key] : null;
      const channel = Array.isArray(channels) ? channels[trace.channel] : null;
      const arr = channel && Array.isArray(channel.values) ? channel.values : null;
      const v = arr && localIdx < arr.length ? Number(arr[localIdx]) : null;
      valuesByTrace[ti].push(Number.isFinite(v) ? v : null);
    }
  }

  return {
    ready: true,
    start,
    end,
    keys,
    timestamps,
    valuesByTrace,
  };
}

function buildSignalsShapes(timestamps) {
  if (!timestamps || timestamps.length === 0) {
    return [];
  }
  const xCurrent = timestamps[timestamps.length - 1];
  return [{
    type: 'line',
    x0: xCurrent,
    x1: xCurrent,
    y0: 0,
    y1: 1,
    yref: 'paper',
    line: { color: '#ff9f1c', width: 1, dash: 'dot' },
  }];
}

async function reactSignalsPlot(windowData) {
  const meta = state.signalsPlot.traceMeta;
  const traces = meta.map((trace, i) => ({
    x: windowData.timestamps,
    y: windowData.valuesByTrace[i],
    name: trace.name,
    mode: 'lines',
    type: 'scattergl',
    line: { width: 1.7 },
  }));

  const shapes = buildSignalsShapes(windowData.timestamps);
  const legendCfg = buildLegendLayout(traces, el.signalsPlot, LEGEND_COMPACT_BOTTOM_PX);

  await Plotly.react('signalsPlot', traces, {
    paper_bgcolor: '#1b263b',
    plot_bgcolor: '#0a131f',
    font: { color: '#e0e1dd', size: 11 },
    margin: { l: 42, r: 20, t: 20, b: legendCfg.bottomMargin },
    xaxis: { title: 'timestamp', gridcolor: '#24354b' },
    yaxis: { title: 'value', gridcolor: '#24354b' },
    legend: legendCfg.legend,
    shapes,
    uirevision: `signals-${state.currentEpisode}-${state.signalStore.keysSig}`,
  }, { responsive: true });

  state.perf.plot2d_updates += 1;
  state.signalsPlot.initialized = true;
}

async function restyleSignalsPlot(windowData) {
  const traceCount = state.signalsPlot.traceMeta.length;
  const indices = Array.from({ length: traceCount }, (_, i) => i);
  const xSeries = state.signalsPlot.traceMeta.map(() => windowData.timestamps);
  const ySeries = windowData.valuesByTrace;

  await Plotly.restyle('signalsPlot', {
    x: xSeries,
    y: ySeries,
  }, indices);

  await Plotly.relayout('signalsPlot', {
    shapes: buildSignalsShapes(windowData.timestamps),
  });

  state.perf.plot2d_updates += 1;
}

async function updateSignalsPlot(frameIndex, force = false) {
  if (typeof Plotly === 'undefined') {
    return;
  }

  if (state.signalsPlot.inFlight) {
    state.signalsPlot.pending = true;
    state.signalsPlot.pendingForce = state.signalsPlot.pendingForce || force;
    return;
  }

  const nowMs = performance.now();
  if (!force && nowMs - state.signalsPlot.lastUpdateMs < computePlotRefreshIntervalMs()) {
    return;
  }

  state.signalsPlot.inFlight = true;

  try {
    const keys = selectedKeysArray();
    ensureChunkStoreConfig(state.signalStore, state.currentEpisode, keys);

    const structureSig = `${state.currentEpisode}|${state.signalStore.keysSig}|${state.historyFrames}`;
    if (state.signalsPlot.structureSig !== structureSig) {
      state.signalsPlot.traceMeta = buildSignalTraceMeta(keys);
      state.signalsPlot.structureSig = structureSig;
      state.signalsPlot.initialized = false;
    }

    if (keys.length === 0 || state.signalsPlot.traceMeta.length === 0) {
      if (!state.signalsPlot.initialized) {
        await Plotly.react('signalsPlot', [], {
          paper_bgcolor: '#1b263b',
          plot_bgcolor: '#0a131f',
          font: { color: '#e0e1dd' },
          xaxis: { title: 'time' },
          yaxis: { title: 'value' },
          margin: { l: 40, r: 20, t: 20, b: 30 },
          uirevision: `signals-empty-${state.currentEpisode}`,
        }, { responsive: true });
        state.perf.plot2d_updates += 1;
        state.signalsPlot.initialized = true;
      }
      return;
    }

    const length = Number(state.schema?.length || 0);
    const windowEnd = Math.min(length, frameIndex + 1);
    const windowStart = Math.max(0, windowEnd - state.historyFrames);

    const prefetchEnd = Math.min(length, Math.max(windowEnd, state.targetFrame + 1) + state.signalStore.chunkSize);
    const prefetchStart = Math.max(0, windowStart - state.signalStore.chunkSize);
    ensureChunksForRange(state.signalStore, prefetchStart, prefetchEnd);

    const windowData = buildSignalsWindowFromCache(frameIndex);
    if (!windowData.ready) {
      return;
    }

    state.lastSignalsTs = windowData.timestamps;
    state.lastSignalsWindowStart = windowData.start;

    if (!state.signalsPlot.initialized || force) {
      await reactSignalsPlot(windowData);
    } else {
      await restyleSignalsPlot(windowData);
    }

    if (windowData.timestamps.length > 0) {
      el.timeInput.value = String(windowData.timestamps[windowData.timestamps.length - 1]);
    }

    state.signalsPlot.lastFrame = frameIndex;
  } finally {
    state.signalsPlot.lastUpdateMs = performance.now();
    state.signalsPlot.inFlight = false;

    if (state.signalsPlot.pending) {
      const pendingForce = state.signalsPlot.pendingForce;
      state.signalsPlot.pending = false;
      state.signalsPlot.pendingForce = false;
      updateSignalsPlot(state.currentFrame, pendingForce).catch(() => {});
    }
  }
}

function defaultTrajectoryCamera() {
  return {
    eye: { x: 1.35, y: 1.35, z: 0.95 },
    up: { x: 0, y: 0, z: 1 },
  };
}

function cloneCamera(cam) {
  if (!cam || typeof cam !== 'object') {
    return null;
  }
  return JSON.parse(JSON.stringify(cam));
}

function captureTrajectoryCamera(relayoutData) {
  if (!relayoutData || typeof relayoutData !== 'object') {
    return;
  }

  if (relayoutData['scene.camera']) {
    state.trajectoryCamera = cloneCamera(relayoutData['scene.camera']);
    return;
  }

  const keys = [
    'scene.camera.eye.x',
    'scene.camera.eye.y',
    'scene.camera.eye.z',
    'scene.camera.center.x',
    'scene.camera.center.y',
    'scene.camera.center.z',
    'scene.camera.up.x',
    'scene.camera.up.y',
    'scene.camera.up.z',
  ];

  if (!keys.some((k) => Object.prototype.hasOwnProperty.call(relayoutData, k))) {
    return;
  }

  const camera = cloneCamera(state.trajectoryCamera || defaultTrajectoryCamera()) || defaultTrajectoryCamera();
  for (const key of keys) {
    if (!Object.prototype.hasOwnProperty.call(relayoutData, key)) {
      continue;
    }
    const value = relayoutData[key];
    if (!Number.isFinite(value)) {
      continue;
    }
    const parts = key.split('.');
    if (parts.length !== 4) {
      continue;
    }
    const section = parts[2];
    const axis = parts[3];
    if (!camera[section]) {
      camera[section] = {};
    }
    camera[section][axis] = Number(value);
  }
  state.trajectoryCamera = camera;
}

function bindTrajectoryRelayoutIfNeeded(graphDiv) {
  if (!graphDiv || state.trajectoryRelayoutBound || typeof graphDiv.on !== 'function') {
    return;
  }
  graphDiv.on('plotly_relayout', (relayoutData) => {
    captureTrajectoryCamera(relayoutData);
  });
  state.trajectoryRelayoutBound = true;
}

function trajectoryAvailableKeys() {
  const keySet = new Set((state.schema?.keys || []).map((k) => k.key));
  const keys = [];
  if (keySet.has('robot0_eef_pos')) {
    keys.push('robot0_eef_pos');
  }
  if (keySet.has('robot1_eef_pos')) {
    keys.push('robot1_eef_pos');
  }
  return keys;
}

function getTrajectoryPointAt(store, key, globalIdx) {
  const chunkStart = Math.floor(globalIdx / store.chunkSize) * store.chunkSize;
  const chunk = getChunk(store, chunkStart);
  if (!chunk) {
    return null;
  }
  const localIdx = globalIdx - Number(chunk.start || chunkStart);
  if (localIdx < 0) {
    return null;
  }

  const channels = chunk.series ? chunk.series[key] : null;
  if (!Array.isArray(channels) || channels.length < 3) {
    return null;
  }

  const vals = [];
  for (let i = 0; i < 3; i += 1) {
    const arr = channels[i] && Array.isArray(channels[i].values) ? channels[i].values : null;
    const v = arr && localIdx < arr.length ? Number(arr[localIdx]) : null;
    if (!Number.isFinite(v)) {
      return null;
    }
    vals.push(v);
  }

  return vals;
}

function buildTrajectoryFromCache(frameIndex) {
  const end = frameIndex + 1;
  const start = Math.max(0, end - state.historyFrames);

  if (!hasChunkRange(state.trajectoryStore, start, end)) {
    return { ready: false, start, end };
  }

  const robots = {};
  const extents = { x: 0, y: 0, z: 0 };
  const yOffsets = { robot0: ROBOT_HALF_SEPARATION_M, robot1: -ROBOT_HALF_SEPARATION_M };

  const updateExtents = (x, y, z) => {
    extents.x = Math.max(extents.x, Math.abs(x));
    extents.y = Math.max(extents.y, Math.abs(y));
    extents.z = Math.max(extents.z, Math.abs(z));
  };

  for (const robot of ['robot0', 'robot1']) {
    const key = `${robot}_eef_pos`;
    if (!state.trajectoryStore.keys.includes(key)) {
      continue;
    }

    const offsetY = Number(yOffsets[robot] || 0);
    const history = [];

    for (let idx = start; idx < end; idx += 1) {
      const p = getTrajectoryPointAt(state.trajectoryStore, key, idx);
      if (!p) {
        continue;
      }
      const point = [Number(p[0]), Number(p[1]) + offsetY, Number(p[2])];
      history.push(point);
      updateExtents(point[0], point[1], point[2]);
    }

    if (!history.length) {
      continue;
    }

    const current = history[history.length - 1];
    updateExtents(0, offsetY, 0);
    robots[robot] = {
      history,
      current,
      origin: [0, offsetY, 0],
    };
  }

  if (Object.keys(robots).length === 0) {
    return { ready: true, start, end, robots: {}, extents };
  }

  return { ready: true, start, end, robots, extents };
}

function computeTrajectoryExtents(trajectoryData, overlayData = { batches: [], averages: {} }) {
  const extents = {
    x: Number(trajectoryData?.extents?.x || 0),
    y: Number(trajectoryData?.extents?.y || 0),
    z: Number(trajectoryData?.extents?.z || 0),
  };

  const updateExtents = (point) => {
    if (!Array.isArray(point) || point.length < 3) {
      return;
    }
    const [x, y, z] = point;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      return;
    }
    extents.x = Math.max(extents.x, Math.abs(x));
    extents.y = Math.max(extents.y, Math.abs(y));
    extents.z = Math.max(extents.z, Math.abs(z));
  };

  for (const batch of overlayData.batches || []) {
    for (const robot of ['robot0', 'robot1']) {
      const overlay = batch.robots ? batch.robots[robot] : null;
      if (!overlay) {
        continue;
      }
      for (const point of overlay.predicted || []) {
        updateExtents(point);
      }
      updateExtents(overlay.currentPredicted);
    }
  }

  for (const robot of ['robot0', 'robot1']) {
    const overlay = overlayData.averages ? overlayData.averages[robot] : null;
    if (!overlay) {
      continue;
    }
    for (const point of overlay.predicted || []) {
      updateExtents(point);
    }
    updateExtents(overlay.currentPredicted);
  }

  return extents;
}

function computeTrajectoryRequiredRadius(trajectoryData, overlayData = { batches: [], averages: {} }) {
  const extents = computeTrajectoryExtents(trajectoryData, overlayData);
  const requiredRadius = Math.max(
    0.25,
    extents.x * 1.15,
    extents.y * 1.15,
    extents.z * 1.15,
    ROBOT_HALF_SEPARATION_M + 0.25,
  );
  return { extents, requiredRadius };
}

function buildTrajectoryStaticTraces(axisRadius) {
  const planeExtent = Math.max(axisRadius, 0.6);
  const traces = [
    {
      type: 'surface',
      x: [[-planeExtent, planeExtent], [-planeExtent, planeExtent]],
      y: [[-planeExtent, -planeExtent], [planeExtent, planeExtent]],
      z: [[0, 0], [0, 0]],
      opacity: 0.28,
      showscale: false,
      showlegend: false,
      hoverinfo: 'skip',
      colorscale: [[0, '#2d435a'], [1, '#2d435a']],
    },
    {
      x: [-axisRadius, axisRadius],
      y: [0, 0],
      z: [0, 0],
      type: 'scatter3d',
      mode: 'lines',
      showlegend: false,
      hoverinfo: 'skip',
      line: { color: '#ff6b6b', width: 4 },
    },
    {
      x: [0, 0],
      y: [-axisRadius, axisRadius],
      z: [0, 0],
      type: 'scatter3d',
      mode: 'lines',
      showlegend: false,
      hoverinfo: 'skip',
      line: { color: '#6bd49a', width: 4 },
    },
    {
      x: [0, 0],
      y: [0, 0],
      z: [-axisRadius, axisRadius],
      type: 'scatter3d',
      mode: 'lines',
      showlegend: false,
      hoverinfo: 'skip',
      line: { color: '#6ca8ff', width: 4 },
    },
  ];

  return [
    ...traces,
    ...buildFloorGridTraces(axisRadius),
    ...buildAxisEndLabelTraces(axisRadius),
  ];
}

function buildTrajectoryDynamicTraces(robots, overlayData = { batches: [], averages: {} }) {
  const colors = { robot0: '#3a86ff', robot1: '#2ec4b6' };
  const traces = [];
  const indices = {};

  for (const robot of ['robot0', 'robot1']) {
    const data = robots[robot];
    if (!data) {
      continue;
    }

    const historyIdx = traces.length;
    traces.push({
      x: data.history.map((p) => p[0]),
      y: data.history.map((p) => p[1]),
      z: data.history.map((p) => p[2]),
      type: 'scatter3d',
      mode: 'lines',
      name: `${robot} history`,
      line: { width: 5, color: colors[robot] || '#ccc' },
      hovertemplate: `${robot} history<br>x=%{x:.3f} m<br>y=%{y:.3f} m<br>z=%{z:.3f} m<extra></extra>`,
    });

    const currentIdx = traces.length;
    traces.push({
      x: [data.current[0]],
      y: [data.current[1]],
      z: [data.current[2]],
      type: 'scatter3d',
      mode: 'markers',
      name: `${robot} current`,
      marker: { size: 7, color: colors[robot] || '#fff', line: { color: '#e0e1dd', width: 1 } },
      hovertemplate: `${robot} current<br>x=%{x:.3f} m<br>y=%{y:.3f} m<br>z=%{z:.3f} m<extra></extra>`,
    });

    traces.push({
      x: [data.origin[0]],
      y: [data.origin[1]],
      z: [data.origin[2]],
      type: 'scatter3d',
      mode: 'markers+text',
      name: `${robot} origin`,
      marker: { size: 6, color: colors[robot] || '#fff', symbol: 'diamond' },
      text: [`${robot} origin`],
      textposition: 'top center',
      textfont: { color: '#c7d5ea', size: 10 },
      hovertemplate: `${robot} origin<br>x=%{x:.3f} m<br>y=%{y:.3f} m<br>z=%{z:.3f} m<extra></extra>`,
    });

    indices[robot] = {
      history: historyIdx,
      current: currentIdx,
    };

  }

  for (const robot of ['robot0', 'robot1']) {
    const batchRobots = (overlayData.batches || [])
      .map((batch) => ({
        batchIndex: batch.batchIndex,
        robot: batch.robots ? batch.robots[robot] : null,
      }))
      .filter((entry) => entry.robot);

    for (let idx = 0; idx < batchRobots.length; idx += 1) {
      const batchEntry = batchRobots[idx];
      const overlay = batchEntry.robot;
      const memberColor = predictionMemberColor(robot, batchEntry.batchIndex);
      traces.push({
        x: overlay.predicted.map((p) => p[0]),
        y: overlay.predicted.map((p) => p[1]),
        z: overlay.predicted.map((p) => p[2]),
        type: 'scatter3d',
        mode: 'lines',
        name: `${robot} samples`,
        showlegend: idx === 0,
        opacity: PREDICTION_MEMBER_TRACE_OPACITY,
        line: {
          width: PREDICTION_MEMBER_LINE_WIDTH,
          color: memberColor,
          dash: 'dash',
        },
        hovertemplate: `${robot} sample ${idx + 1}<br>x=%{x:.3f} m<br>y=%{y:.3f} m<br>z=%{z:.3f} m<extra></extra>`,
      });
    }

    const average = overlayData.averages ? overlayData.averages[robot] : null;
    if (!average) {
      continue;
    }

    const averageIdx = traces.length;
    traces.push({
      x: average.predicted.map((p) => p[0]),
      y: average.predicted.map((p) => p[1]),
      z: average.predicted.map((p) => p[2]),
      type: 'scatter3d',
      mode: 'lines',
      name: `${robot} average`,
      opacity: 1,
      line: {
        width: PREDICTION_AVERAGE_LINE_WIDTH,
        color: PREDICTION_AVERAGE_COLORS[robot] || '#e76f51',
      },
      hovertemplate: `${robot} average<br>x=%{x:.3f} m<br>y=%{y:.3f} m<br>z=%{z:.3f} m<extra></extra>`,
    });

    const averageCurrentIdx = traces.length;
    traces.push({
      x: [average.currentPredicted[0]],
      y: [average.currentPredicted[1]],
      z: [average.currentPredicted[2]],
      type: 'scatter3d',
      mode: 'markers',
      name: `${robot} average current`,
      showlegend: false,
      marker: {
        size: PREDICTION_AVERAGE_MARKER_SIZE,
        color: PREDICTION_AVERAGE_COLORS[robot] || '#e76f51',
        line: { color: '#ffffff', width: 2.5 },
      },
      hovertemplate: `${robot} average current<br>x=%{x:.3f} m<br>y=%{y:.3f} m<br>z=%{z:.3f} m<extra></extra>`,
    });

    indices[robot].averageLine = averageIdx;
    indices[robot].averageCurrent = averageCurrentIdx;
  }

  return { traces, indices };
}

async function renderTrajectoryNoData(reason) {
  await Plotly.react('trajectory3d', [], {
    paper_bgcolor: '#1b263b',
    plot_bgcolor: '#0a131f',
    font: { color: '#e0e1dd' },
    annotations: [{ text: reason || 'No 3D trajectory', showarrow: false }],
    margin: { l: 10, r: 10, t: 10, b: 10 },
    uirevision: `trajectory-${state.currentEpisode}`,
  }, { responsive: true });
  state.trajectoryPlot.initialized = true;
  state.trajectoryPlot.dynamicTraceIndices = {};
  state.trajectoryPlot.staticTraceCount = 0;
  state.trajectoryPlot.overlaySig = '';
  state.perf.plot3d_updates += 1;
}

async function reactTrajectoryPlot(trajectoryData, overlay = null) {
  const overlayData = buildInferenceOverlayData(overlay, state.currentFrame);
  const overlaySig = inferenceOverlaySignature(overlay);
  const { requiredRadius } = computeTrajectoryRequiredRadius(trajectoryData, overlayData);

  state.trajectoryPlot.axisRadius = Math.max(state.trajectoryPlot.axisRadius, requiredRadius);
  const staticTraces = buildTrajectoryStaticTraces(state.trajectoryPlot.axisRadius);
  const dynamic = buildTrajectoryDynamicTraces(trajectoryData.robots, overlayData);
  const allTraces = [...staticTraces, ...dynamic.traces];

  const legendCfg = buildLegendLayout(dynamic.traces, el.trajectory3d, LEGEND_COMPACT_BOTTOM_PX);
  const axisRange = [-state.trajectoryPlot.axisRadius, state.trajectoryPlot.axisRadius];
  const camera = cloneCamera(state.trajectoryCamera) || defaultTrajectoryCamera();

  await Plotly.react('trajectory3d', allTraces, {
    paper_bgcolor: '#1b263b',
    scene: {
      bgcolor: '#0a131f',
      xaxis: {
        title: 'X (forward/back, m)',
        range: axisRange,
        gridcolor: '#23364a',
        zeroline: true,
        zerolinecolor: '#ff6b6b',
        zerolinewidth: 1,
      },
      yaxis: {
        title: 'Y (table/lateral, m)',
        range: axisRange,
        gridcolor: '#23364a',
        zeroline: true,
        zerolinecolor: '#6bd49a',
        zerolinewidth: 1,
      },
      zaxis: {
        title: 'Z (height, m)',
        range: axisRange,
        gridcolor: '#23364a',
        zeroline: true,
        zerolinecolor: '#6ca8ff',
        zerolinewidth: 1,
      },
      aspectmode: 'cube',
      camera,
    },
    font: { color: '#e0e1dd', size: 11 },
    margin: { l: 0, r: 0, t: 14, b: legendCfg.bottomMargin },
    legend: legendCfg.legend,
    uirevision: `trajectory-${state.currentEpisode}`,
  }, { responsive: true }).then((graphDiv) => {
    bindTrajectoryRelayoutIfNeeded(graphDiv);
  });

  state.trajectoryPlot.staticTraceCount = staticTraces.length;
  state.trajectoryPlot.dynamicTraceIndices = {};
  for (const [robot, idxObj] of Object.entries(dynamic.indices)) {
    state.trajectoryPlot.dynamicTraceIndices[robot] = {
      history: state.trajectoryPlot.staticTraceCount + idxObj.history,
      current: state.trajectoryPlot.staticTraceCount + idxObj.current,
      averageLine: Number.isInteger(idxObj.averageLine) ? state.trajectoryPlot.staticTraceCount + idxObj.averageLine : null,
      averageCurrent: Number.isInteger(idxObj.averageCurrent) ? state.trajectoryPlot.staticTraceCount + idxObj.averageCurrent : null,
    };
  }

  state.trajectoryPlot.overlaySig = overlaySig;
  state.trajectoryPlot.initialized = true;
  state.perf.plot3d_updates += 1;
}

async function restyleTrajectoryPlot(trajectoryData, overlay = null) {
  const overlayData = buildInferenceOverlayData(overlay, state.currentFrame);
  const overlaySig = inferenceOverlaySignature(overlay);
  if (overlaySig !== state.trajectoryPlot.overlaySig) {
    await reactTrajectoryPlot(trajectoryData, overlay);
    return;
  }

  const { requiredRadius } = computeTrajectoryRequiredRadius(trajectoryData, overlayData);

  if (requiredRadius > state.trajectoryPlot.axisRadius * 0.92) {
    state.trajectoryPlot.axisRadius = requiredRadius * 1.1;
    await reactTrajectoryPlot(trajectoryData, overlay);
    return;
  }

  const indices = [];
  const x = [];
  const y = [];
  const z = [];

  for (const robot of ['robot0', 'robot1']) {
    const idxObj = state.trajectoryPlot.dynamicTraceIndices[robot];
    const data = trajectoryData.robots[robot];
    if (idxObj && data) {
      indices.push(idxObj.history);
      x.push(data.history.map((p) => p[0]));
      y.push(data.history.map((p) => p[1]));
      z.push(data.history.map((p) => p[2]));

      indices.push(idxObj.current);
      x.push([data.current[0]]);
      y.push([data.current[1]]);
      z.push([data.current[2]]);
    }

    const average = overlayData.averages ? overlayData.averages[robot] : null;
    if (idxObj && average && Number.isInteger(idxObj.averageCurrent)) {
      indices.push(idxObj.averageCurrent);
      x.push([average.currentPredicted[0]]);
      y.push([average.currentPredicted[1]]);
      z.push([average.currentPredicted[2]]);
    }

  }

  if (indices.length > 0) {
    await Plotly.restyle('trajectory3d', { x, y, z }, indices);
    state.perf.plot3d_updates += 1;
  }
}

async function updateTrajectoryPlot(frameIndex, force = false) {
  if (typeof Plotly === 'undefined') {
    return;
  }

  if (state.trajectoryPlot.inFlight) {
    state.trajectoryPlot.pending = true;
    state.trajectoryPlot.pendingForce = state.trajectoryPlot.pendingForce || force;
    return;
  }

  const nowMs = performance.now();
  if (!force && nowMs - state.trajectoryPlot.lastUpdateMs < computePlotRefreshIntervalMs()) {
    return;
  }

  state.trajectoryPlot.inFlight = true;

  try {
    const keys = trajectoryAvailableKeys();
    state.trajectoryPlot.keys = keys;

    if (keys.length === 0) {
      await renderTrajectoryNoData('No robot*_eef_pos keys available');
      return;
    }

    ensureChunkStoreConfig(state.trajectoryStore, state.currentEpisode, keys);

    const length = Number(state.schema?.length || 0);
    const end = Math.min(length, frameIndex + 1);
    const start = Math.max(0, end - state.historyFrames);

    const prefetchEnd = Math.min(length, Math.max(end, state.targetFrame + 1) + state.trajectoryStore.chunkSize);
    const prefetchStart = Math.max(0, start - state.trajectoryStore.chunkSize);
    ensureChunksForRange(state.trajectoryStore, prefetchStart, prefetchEnd);

    const trajectoryData = buildTrajectoryFromCache(frameIndex);
    if (!trajectoryData.ready) {
      return;
    }

    if (Object.keys(trajectoryData.robots || {}).length === 0) {
      await renderTrajectoryNoData('No valid 3D points in current window');
      return;
    }

    const overlay = currentInferenceOverlay();
    const overlaySig = inferenceOverlaySignature(overlay);
    if (!state.trajectoryPlot.initialized || force || overlaySig !== state.trajectoryPlot.overlaySig) {
      await reactTrajectoryPlot(trajectoryData, overlay);
    } else {
      await restyleTrajectoryPlot(trajectoryData, overlay);
    }

    state.trajectoryPlot.lastFrame = frameIndex;
  } finally {
    state.trajectoryPlot.lastUpdateMs = performance.now();
    state.trajectoryPlot.inFlight = false;

    if (state.trajectoryPlot.pending) {
      const pendingForce = state.trajectoryPlot.pendingForce;
      state.trajectoryPlot.pending = false;
      state.trajectoryPlot.pendingForce = false;
      updateTrajectoryPlot(state.currentFrame, pendingForce).catch(() => {});
    }
  }
}

function requestPlotRefresh(force = false) {
  state.plot.pending = true;
  state.plot.force = state.plot.force || force;
}

async function flushPlotRefresh(nowMs = performance.now(), force = false) {
  if (!state.plot.pending) {
    return;
  }

  const shouldForce = force || state.plot.force;
  if (!shouldForce && nowMs - state.plot.lastRunMs < computePlotRefreshIntervalMs()) {
    return;
  }

  if (state.plot.running) {
    return;
  }

  state.plot.pending = false;
  state.plot.force = false;
  state.plot.running = true;

  try {
    await Promise.all([
      updateSignalsPlot(state.currentFrame, shouldForce),
      updateTrajectoryPlot(state.currentFrame, shouldForce),
    ]);
  } finally {
    state.plot.lastRunMs = performance.now();
    state.plot.running = false;
    updatePerfBadge();
  }
}

function onSelectedKeysChanged() {
  state.signalsPlot.structureSig = '';
  state.signalsPlot.initialized = false;
  requestPlotRefresh(true);
  flushPlotRefresh(performance.now(), true).catch(() => {});
}

function applyPreset(name) {
  const groups = state.schema?.key_groups || {};
  const presetKeys = new Set(groups[name] || []);

  state.selectedKeys.clear();
  for (const row of Array.from(el.keyList.children)) {
    const cb = row.querySelector('input[type="checkbox"]');
    if (!cb) {
      continue;
    }
    cb.checked = presetKeys.has(cb.value);
    if (cb.checked) {
      state.selectedKeys.add(cb.value);
    }
  }

  onSelectedKeysChanged();
}

function uncheckAllSignals() {
  state.selectedKeys.clear();
  for (const row of Array.from(el.keyList.children)) {
    const cb = row.querySelector('input[type="checkbox"]');
    if (!cb) {
      continue;
    }
    cb.checked = false;
  }
  onSelectedKeysChanged();
}

async function fetchTimingForEpisode(episodeIndex) {
  try {
    const query = new URLSearchParams({ fps_cap: String(state.fpsCap) });
    return await apiGet(`/api/episode/${episodeIndex}/timing?${query.toString()}`);
  } catch (_) {
    return {
      has_timestamps: false,
      median_dt_sec: null,
      p90_dt_sec: null,
      suggested_fps: state.fpsCap,
      frame_count: Number(state.schema?.length || 0),
    };
  }
}

async function refreshTimingForCurrentEpisode() {
  if (!state.schema) {
    return;
  }
  const ep = state.currentEpisode;
  const timing = await fetchTimingForEpisode(ep);
  if (ep !== state.currentEpisode) {
    return;
  }
  state.timing = timing;
  if (state.isPlaying) {
    state.playAnchorFrame = state.currentFrame;
    state.playAnchorMs = performance.now();
    state.targetFrame = state.currentFrame;
  }
  updatePerfBadge();
}

function resetStoresForEpisode() {
  resetMainPipeline(computeMainPipelineKey());

  resetChunkStore(state.signalStore, state.currentEpisode, []);
  state.signalsPlot = {
    initialized: false,
    structureSig: '',
    traceMeta: [],
    inFlight: false,
    pending: false,
    pendingForce: false,
    lastUpdateMs: 0,
    lastFrame: -1,
  };

  resetChunkStore(state.trajectoryStore, state.currentEpisode, trajectoryAvailableKeys());
  state.trajectoryPlot = {
    available: false,
    keys: trajectoryAvailableKeys(),
    initialized: false,
    inFlight: false,
    pending: false,
    pendingForce: false,
    lastUpdateMs: 0,
    lastFrame: -1,
    axisRadius: 0.8,
    staticTraceCount: 0,
    dynamicTraceIndices: {},
    overlaySig: '',
  };
}

async function runInferenceOverlay() {
  if (!hasEpisodes() || !state.schema) {
    return;
  }

  state.inference.yamlPath = (el.inferenceYamlPath?.value || '').trim();
  state.inference.serverHost = (el.inferenceServerHost?.value || DEFAULT_INFERENCE_SERVER_HOST).trim() || DEFAULT_INFERENCE_SERVER_HOST;
  state.inference.serverPort = sanitizeInferencePort(el.inferenceServerPort?.value);
  state.inference.inferenceMode = sanitizeInferenceMode(el.inferenceModeSelect?.value);
  state.inference.warmupSteps = sanitizeWarmupSteps(el.inferenceWarmupSteps?.value);
  state.inference.batchSize = sanitizeBatchSize(el.inferenceBatchSize?.value);
  syncInferenceControls();

  if (!state.inference.yamlPath) {
    setInferenceStatus('Provide a YAML path before running inference.', 'error');
    syncInferenceControls();
    return;
  }

  state.inference.running = true;
  setInferenceStatus(
    `Running ${inferenceModeLabel(state.inference.inferenceMode)} inference at episode ${state.currentEpisode}, frame ${state.currentFrame}…`,
    'pending',
  );
  syncInferenceControls();

  try {
    const payload = await apiPost('/api/inference/run', {
      episode_index: state.currentEpisode,
      frame_index: state.currentFrame,
      yaml_path: state.inference.yamlPath,
      server_host: state.inference.serverHost,
      server_port: state.inference.serverPort,
      warmup_steps: state.inference.warmupSteps,
      batch_size: state.inference.batchSize,
      inference_mode: state.inference.inferenceMode,
      no_gripper: !!state.inference.noGripper,
    });

    if (payload.predict_sampling_details && typeof payload.predict_sampling_details === 'object') {
      console.info('[inference predict] sampling details', payload.predict_sampling_details);
    }
    if (payload.long_sampling_details && typeof payload.long_sampling_details === 'object') {
      console.info('[inference long] sampling details', payload.long_sampling_details);
    }

    saveInferenceOverlay(payload);
    const actionShape = Array.isArray(payload.action_shape) ? payload.action_shape.join('x') : 'unknown';
    const batchCount = Math.max(1, Number(payload.batch_count_returned || 1));
    const episodeHistoryCount = state.inference.history.filter((overlay) => overlay.episode_index === state.currentEpisode).length;
    const branchNames = [];
    if (payload.predict_sampling_details && typeof payload.predict_sampling_details === 'object') {
      branchNames.push(
        ...Object.values(payload.predict_sampling_details)
          .map((detail) => String(detail?.branch || 'short')),
      );
    }
    if (payload.long_sampling_details && typeof payload.long_sampling_details === 'object') {
      branchNames.push(
        ...Object.values(payload.long_sampling_details)
          .map((detail) => String(detail?.branch || 'long')),
      );
    }
    const uniqueBranchNames = [...new Set(branchNames)].sort();
    const branchSummary = uniqueBranchNames.length > 0 ? uniqueBranchNames.join(', ') : 'short';
    const actionSummary = batchCount > 1
      ? `Received ${batchCount} batched trajectory set(s) with raw shape ${actionShape}.`
      : `Received ${actionShape} action(s).`;
    setInferenceStatus(
      `${actionSummary} Train-time inference sent 1 ${String(payload.request_type_sent || 'predict_action')} call ` +
      `with branch context(s) [${branchSummary}]. Saved ${episodeHistoryCount} inference run(s) for this episode.`,
      'success',
    );
    requestPlotRefresh(true);
    await flushPlotRefresh(performance.now(), true);
  } catch (err) {
    setInferenceStatus(String(err), 'error');
  } finally {
    state.inference.running = false;
    syncInferenceControls();
  }
}

async function deleteCurrentEpisode() {
  if (!hasEpisodes() || state.isDeletingEpisode) {
    return;
  }

  const episodeToDelete = state.currentEpisode;
  const confirmText = [
    `Delete episode ${episodeToDelete}?`,
    'This will modify the dataset on disk.',
    'For sidecar datasets, video folders will also be renamed to keep numbering contiguous.',
  ].join('\n');

  if (!window.confirm(confirmText)) {
    return;
  }

  state.isDeletingEpisode = true;
  updateDeleteButtonState();

  try {
    const deletePayload = await apiDelete(`/api/episode/${episodeToDelete}?delete_videos=true`);
    clearInferenceOverlay({ clearStatus: true, refreshPlot: false });
    clearProgressGraph({ clearStatus: true });
    state.summary = await apiGet('/api/dataset/summary');
    const episodesPayload = await apiGet('/api/episodes');
    state.episodes = episodesPayload.episodes || [];
    renderEpisodeOptions();

    if (!hasEpisodes()) {
      await renderNoEpisodesState();
      return;
    }

    let nextEpisode = Number(deletePayload.suggested_episode_index);
    if (!Number.isInteger(nextEpisode) || !state.episodes.some((ep) => ep.episode_index === nextEpisode)) {
      nextEpisode = state.episodes[0].episode_index;
    }

    await loadEpisode(nextEpisode);
  } catch (err) {
    window.alert(`Delete failed: ${String(err)}`);
  } finally {
    state.isDeletingEpisode = false;
    updateDeleteButtonState();
  }
}

async function loadEpisode(episodeIndex) {
  if (!hasEpisodes()) {
    await renderNoEpisodesState();
    return;
  }

  let resolvedEpisode = Number(episodeIndex);
  if (!state.episodes.some((ep) => ep.episode_index === resolvedEpisode)) {
    resolvedEpisode = state.episodes[0].episode_index;
  }

  stopPlayback();
  state.episodeLoadToken += 1;
  const token = state.episodeLoadToken;
  state.timeline.dragging = false;
  state.trajectoryCamera = null;
  state.trajectoryRelayoutBound = false;
  state.currentEpisode = resolvedEpisode;
  el.episodeSelect.value = String(state.currentEpisode);
  el.episodeSelect.disabled = false;
  updateDeleteButtonState();

  const [schemaPayload, eventsPayload, timingPayload] = await Promise.all([
    apiGet(`/api/episode/${state.currentEpisode}/schema`),
    apiGet(`/api/episode/${state.currentEpisode}/events`),
    fetchTimingForEpisode(state.currentEpisode),
  ]);

  if (token !== state.episodeLoadToken) {
    return;
  }

  state.schema = schemaPayload;
  state.events = eventsPayload.events || [];
  state.timing = timingPayload;

  const length = Number(state.schema.length || 0);
  const daggerSegments = await fetchDaggerSegments(state.currentEpisode, length, state.schema);
  if (token !== state.episodeLoadToken) {
    return;
  }
  state.daggerSegments = daggerSegments;

  el.timelineSlider.max = String(Math.max(0, length - 1));
  el.frameInput.max = String(Math.max(0, length - 1));

  buildStreamSelectors();
  buildCameraGrid();
  buildKeySelector();
  renderEventMarkers();
  renderEventList();

  resetStoresForEpisode();

  applyPreset('Core');

  showProgressPlotForCurrentEpisode();

  state.currentFrame = 0;
  state.targetFrame = 0;
  setFrameUI(0);

  await showFrameNow(0, {
    refreshPlots: false,
    previewPolicy: 'full',
    latestOnly: true,
    allowLookahead: false,
    forcePlotRefresh: false,
  });
  requestPlotRefresh(true);
  await flushPlotRefresh(performance.now(), true);

  updatePerfBadge();
}

function bindControls() {
  el.episodeSelect.addEventListener('change', async (e) => {
    if (!e.target.value) {
      return;
    }
    await loadEpisode(Number(e.target.value));
  });

  if (el.deleteEpisodeBtn) {
    el.deleteEpisodeBtn.addEventListener('click', () => {
      deleteCurrentEpisode().catch(() => {});
    });
  }

  el.playBtn.addEventListener('click', () => {
    if (state.isPlaying) {
      stopPlayback();
    } else {
      startPlayback();
    }
  });

  if (el.previewModeSelect) {
    el.previewModeSelect.addEventListener('change', () => {
      state.performanceControls.previewMode = el.previewModeSelect.value === 'hide-others' ? 'hide-others' : 'all';
      syncPerformanceControls();
      persistPerformanceSettings();
      updatePreviewSelectionClasses();
      updatePreviewFrames(state.currentFrame, 'full');
      updatePerfBadge();
    });
  }

  if (el.imageModeSelect) {
    el.imageModeSelect.addEventListener('change', () => {
      state.performanceControls.imageMode = el.imageModeSelect.value === 'dynamic' ? 'dynamic' : 'full';
      syncPerformanceControls();
      persistPerformanceSettings();
      resetMainPipeline(computeMainPipelineKey());
      setFrame(state.currentFrame, {
        refreshPlots: false,
        previewPolicy: 'full',
        latestOnly: true,
        allowLookahead: false,
        forcePlotRefresh: false,
      }).catch(() => {});
    });
  }

  if (el.plotRateSelect) {
    el.plotRateSelect.addEventListener('change', () => {
      state.performanceControls.plotRate = el.plotRateSelect.value === 'low' ? 'low' : 'full';
      syncPerformanceControls();
      persistPerformanceSettings();
      requestPlotRefresh(true);
      flushPlotRefresh(performance.now(), true).catch(() => {});
    });
  }

  el.frameInput.addEventListener('change', () => {
    setFrame(Number(el.frameInput.value), {
      refreshPlots: true,
      previewPolicy: 'full',
      latestOnly: true,
      allowLookahead: false,
      forcePlotRefresh: true,
    }).catch(() => {});
  });

  el.timeInput.addEventListener('change', onTimeJump);

  el.timelineSlider.addEventListener('pointerdown', () => {
    state.timeline.dragging = true;
    cancelHighQualityRefresh();
    updatePerfBadge();
  });

  const stopTimelineDrag = () => {
    if (!state.timeline.dragging) {
      return;
    }
    state.timeline.dragging = false;
    updatePerfBadge();
  };

  el.timelineSlider.addEventListener('pointerup', stopTimelineDrag);
  el.timelineSlider.addEventListener('pointercancel', stopTimelineDrag);

  el.timelineSlider.addEventListener('input', () => {
    setFrame(Number(el.timelineSlider.value), {
      refreshPlots: true,
      previewPolicy: 'auto',
      latestOnly: true,
      allowLookahead: false,
      forcePlotRefresh: false,
    }).catch(() => {});
  });

  el.timelineSlider.addEventListener('change', () => {
    state.timeline.dragging = false;
    setFrame(Number(el.timelineSlider.value), {
      refreshPlots: true,
      previewPolicy: 'full',
      latestOnly: true,
      allowLookahead: false,
      forcePlotRefresh: true,
    }).catch(() => {});
  });

  el.primaryStreamSelect.addEventListener('change', () => {
    state.primaryStream = el.primaryStreamSelect.value;
    updatePreviewSelectionClasses();
    resetMainPipeline(computeMainPipelineKey());
    setFrame(state.currentFrame, {
      refreshPlots: false,
      previewPolicy: 'full',
      latestOnly: true,
      allowLookahead: false,
      forcePlotRefresh: false,
    }).catch(() => {});
  });

  el.depthToggle.addEventListener('change', () => {
    resetMainPipeline(computeMainPipelineKey());
    setFrame(state.currentFrame, {
      refreshPlots: false,
      previewPolicy: 'full',
      latestOnly: true,
      allowLookahead: false,
      forcePlotRefresh: false,
    }).catch(() => {});
  });

  el.colormapSelect.addEventListener('change', () => {
    if (el.depthToggle.checked) {
      resetMainPipeline(computeMainPipelineKey());
      setFrame(state.currentFrame, {
        refreshPlots: false,
        previewPolicy: 'full',
        latestOnly: true,
        allowLookahead: false,
        forcePlotRefresh: false,
      }).catch(() => {});
    }
  });

  el.playModeSelect.addEventListener('change', () => {
    state.playMode = el.playModeSelect.value === 'fixed' ? 'fixed' : 'timestamp';
    if (state.isPlaying) {
      state.playAnchorFrame = state.currentFrame;
      state.playAnchorMs = performance.now();
      state.targetFrame = state.currentFrame;
    }
    updatePerfBadge();
  });

  el.fpsCapInput.addEventListener('change', () => {
    state.fpsCap = clamp(Math.round(toFiniteNumber(el.fpsCapInput.value, DEFAULT_FPS_CAP)), 1, 240);
    el.fpsCapInput.value = String(state.fpsCap);
    refreshTimingForCurrentEpisode().catch(() => {});
    updatePerfBadge();
  });

  el.speedInput.addEventListener('change', () => {
    state.speedMultiplier = clamp(toFiniteNumber(el.speedInput.value, DEFAULT_SPEED), 0.1, 8.0);
    el.speedInput.value = state.speedMultiplier.toFixed(1);
    if (state.isPlaying) {
      state.playAnchorFrame = state.currentFrame;
      state.playAnchorMs = performance.now();
      state.targetFrame = state.currentFrame;
    }
    updatePerfBadge();
  });

  el.historyInput.addEventListener('change', () => {
    state.historyFrames = Math.max(1, Math.round(toFiniteNumber(el.historyInput.value, 180)));
    el.historyInput.value = String(state.historyFrames);

    state.signalsPlot.structureSig = '';
    state.signalsPlot.initialized = false;
    state.trajectoryPlot.initialized = false;
    requestPlotRefresh(true);
    flushPlotRefresh(performance.now(), true).catch(() => {});
  });

  for (const btn of presetButtons) {
    btn.addEventListener('click', () => applyPreset(btn.dataset.preset));
  }

  if (el.inferenceYamlPath) {
    el.inferenceYamlPath.addEventListener('change', () => {
      state.inference.yamlPath = (el.inferenceYamlPath.value || '').trim();
      syncInferenceControls();
      persistInferenceSettings();
    });
  }

  if (el.inferenceServerHost) {
    el.inferenceServerHost.addEventListener('change', () => {
      state.inference.serverHost = (el.inferenceServerHost.value || DEFAULT_INFERENCE_SERVER_HOST).trim() || DEFAULT_INFERENCE_SERVER_HOST;
      syncInferenceControls();
      persistInferenceSettings();
    });
  }

  if (el.inferenceServerPort) {
    el.inferenceServerPort.addEventListener('change', () => {
      state.inference.serverPort = sanitizeInferencePort(el.inferenceServerPort.value);
      syncInferenceControls();
      persistInferenceSettings();
    });
  }

  if (el.inferenceModeSelect) {
    el.inferenceModeSelect.addEventListener('change', () => {
      state.inference.inferenceMode = sanitizeInferenceMode(el.inferenceModeSelect.value);
      syncInferenceControls();
      persistInferenceSettings();
    });
  }

  if (el.inferenceWarmupSteps) {
    el.inferenceWarmupSteps.addEventListener('change', () => {
      state.inference.warmupSteps = sanitizeWarmupSteps(el.inferenceWarmupSteps.value);
      syncInferenceControls();
      persistInferenceSettings();
    });
  }

  if (el.inferenceBatchSize) {
    el.inferenceBatchSize.addEventListener('change', () => {
      state.inference.batchSize = sanitizeBatchSize(el.inferenceBatchSize.value);
      syncInferenceControls();
      persistInferenceSettings();
    });
  }

  if (el.inferenceNoGripper) {
    el.inferenceNoGripper.addEventListener('change', () => {
      state.inference.noGripper = !!el.inferenceNoGripper.checked;
      persistInferenceSettings();
    });
  }

  if (el.uncheckSignalsBtn) {
    el.uncheckSignalsBtn.addEventListener('click', () => {
      uncheckAllSignals();
    });
  }

  if (el.runInferenceBtn) {
    el.runInferenceBtn.addEventListener('click', () => {
      runInferenceOverlay().catch(() => {});
    });
  }

  if (el.clearInferenceBtn) {
    el.clearInferenceBtn.addEventListener('click', () => {
      clearInferenceOverlay({ clearStatus: true, refreshPlot: true });
    });
  }

  if (el.progressYamlPath) {
    el.progressYamlPath.addEventListener('change', () => {
      state.progressGraph.yamlPath = (el.progressYamlPath.value || '').trim();
      persistProgressSettings();
      showProgressPlotForCurrentEpisode();
    });
  }

  if (el.progressServerHost) {
    el.progressServerHost.addEventListener('change', () => {
      state.progressGraph.serverHost = (el.progressServerHost.value || DEFAULT_PROGRESS_SERVER_HOST).trim() || DEFAULT_PROGRESS_SERVER_HOST;
      syncProgressControls();
      persistProgressSettings();
    });
  }

  if (el.progressServerPort) {
    el.progressServerPort.addEventListener('change', () => {
      state.progressGraph.serverPort = sanitizeProgressPort(el.progressServerPort.value);
      syncProgressControls();
      persistProgressSettings();
    });
  }

  if (el.progressEvalEvery) {
    el.progressEvalEvery.addEventListener('change', () => {
      state.progressGraph.evalEvery = sanitizeProgressEvalEvery(el.progressEvalEvery.value);
      persistProgressSettings();
      showProgressPlotForCurrentEpisode();
    });
  }

  if (el.runProgressBtn) {
    el.runProgressBtn.addEventListener('click', () => {
      runProgressGraph().catch(() => {});
    });
  }

  if (el.clearProgressBtn) {
    el.clearProgressBtn.addEventListener('click', () => {
      clearProgressGraph({ clearStatus: true });
    });
  }
}

async function init() {
  try {
    state.playMode = el.playModeSelect.value === 'fixed' ? 'fixed' : 'timestamp';
    state.fpsCap = clamp(Math.round(toFiniteNumber(el.fpsCapInput.value, DEFAULT_FPS_CAP)), 1, 240);
    state.speedMultiplier = clamp(toFiniteNumber(el.speedInput.value, DEFAULT_SPEED), 0.1, 8.0);
    state.historyFrames = Math.max(1, Math.round(toFiniteNumber(el.historyInput.value, 180)));
    state.inference.yamlPath = (el.inferenceYamlPath?.value || '').trim();
    state.inference.serverHost = (el.inferenceServerHost?.value || DEFAULT_INFERENCE_SERVER_HOST).trim() || DEFAULT_INFERENCE_SERVER_HOST;
    state.inference.serverPort = sanitizeInferencePort(el.inferenceServerPort?.value);
    state.inference.inferenceMode = sanitizeInferenceMode(el.inferenceModeSelect?.value);
    state.inference.warmupSteps = sanitizeWarmupSteps(el.inferenceWarmupSteps?.value);
    state.inference.batchSize = sanitizeBatchSize(el.inferenceBatchSize?.value);
    state.progressGraph.yamlPath = (el.progressYamlPath?.value || '').trim();
    state.progressGraph.serverHost = (el.progressServerHost?.value || DEFAULT_PROGRESS_SERVER_HOST).trim() || DEFAULT_PROGRESS_SERVER_HOST;
    state.progressGraph.serverPort = sanitizeProgressPort(el.progressServerPort?.value);
    state.progressGraph.evalEvery = sanitizeProgressEvalEvery(el.progressEvalEvery?.value);
    loadCachedPerformanceSettings();
    syncPerformanceControls();
    loadCachedInferenceSettings();
    syncInferenceControls();
    loadCachedProgressSettings();
    syncProgressControls();

    state.summary = await apiGet('/api/dataset/summary');

    if (!state.summary.supported) {
      setUnsupported(state.summary.unsupported_reason || 'Unsupported dataset format');
      return;
    }

    clearUnsupported();
    const episodesPayload = await apiGet('/api/episodes');
    state.episodes = episodesPayload.episodes || [];

    renderEpisodeOptions();
    updateDeleteButtonState();

    let initialEpisode = state.summary.requested_episode;
    if (initialEpisode === null || initialEpisode === undefined) {
      initialEpisode = state.episodes.length > 0 ? state.episodes[0].episode_index : 0;
    }

    bindControls();
    if (hasEpisodes()) {
      await loadEpisode(initialEpisode);
    } else {
      await renderNoEpisodesState();
    }
    updatePerfBadge();
  } catch (err) {
    setUnsupported(String(err));
  }
}

init();
