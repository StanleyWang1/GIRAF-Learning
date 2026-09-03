const TRACE_COLORS = [
  '#4d9cff', '#31d3c3', '#f6c85f', '#ff7b8b', '#b592ff', '#ff9f5a',
  '#72d572', '#64c7ff', '#f092c0', '#d4e157', '#80cbc4', '#ffcc80',
  '#90caf9', '#ce93d8', '#a5d6a7', '#ef9a9a',
];

const SAMPLE_KEYS = [
  'alignment_valid',
  'tracking',
  'clutch',
  'motor_command_accepted',
  'grasp_label',
  'control_age_ns',
  'motor_age_ns',
];

const state = {
  summary: null,
  episodes: [],
  schema: null,
  metrics: null,
  events: [],
  currentEpisode: 0,
  currentFrame: 0,
  selectedKeys: new Set(),
  playing: false,
  playHandle: 0,
  playAnchorTime: 0,
  playAnchorFrame: 0,
  dragging: false,
  loadToken: 0,
  signalToken: 0,
  signalTimer: 0,
  lastSignalRequest: 0,
  lastSignalPayload: null,
  framePending: null,
  frameRunning: false,
  frameObjectUrl: '',
};

const el = {
  datasetPath: document.getElementById('datasetPath'),
  summaryBadge: document.getElementById('summaryBadge'),
  errorBanner: document.getElementById('errorBanner'),
  previousEpisode: document.getElementById('previousEpisode'),
  nextEpisode: document.getElementById('nextEpisode'),
  episodeSelect: document.getElementById('episodeSelect'),
  frameInput: document.getElementById('frameInput'),
  playButton: document.getElementById('playButton'),
  speedSelect: document.getElementById('speedSelect'),
  episodeSubtitle: document.getElementById('episodeSubtitle'),
  frameStatus: document.getElementById('frameStatus'),
  fpsStatus: document.getElementById('fpsStatus'),
  frameLoading: document.getElementById('frameLoading'),
  cameraFrame: document.getElementById('cameraFrame'),
  timeline: document.getElementById('timeline'),
  eventMarkers: document.getElementById('eventMarkers'),
  timelineFrame: document.getElementById('timelineFrame'),
  timelineTime: document.getElementById('timelineTime'),
  metricsGrid: document.getElementById('metricsGrid'),
  metadataWarning: document.getElementById('metadataWarning'),
  sampleGrid: document.getElementById('sampleGrid'),
  historySelect: document.getElementById('historySelect'),
  presetButtons: document.getElementById('presetButtons'),
  keyList: document.getElementById('keyList'),
  plotWrap: document.querySelector('.plot-wrap'),
  signalPlot: document.getElementById('signalPlot'),
  emptyPlot: document.getElementById('emptyPlot'),
  signalLegend: document.getElementById('signalLegend'),
  eventCount: document.getElementById('eventCount'),
  eventList: document.getElementById('eventList'),
};

async function apiGet(path) {
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

function showError(error) {
  const message = error instanceof Error ? error.message : String(error);
  el.errorBanner.textContent = message;
  el.errorBanner.classList.remove('hidden');
}

function clearError() {
  el.errorBanner.textContent = '';
  el.errorBanner.classList.add('hidden');
}

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function percent(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) {
    return '—';
  }
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function numberText(value, digits = 1) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) {
    return '—';
  }
  return Number(value).toFixed(digits);
}

function basename(path) {
  const parts = String(path || '').split('/').filter(Boolean);
  return parts.length ? parts[parts.length - 1] : String(path || 'dataset');
}

function currentLength() {
  return Number(state.schema?.length || 0);
}

function currentEpisodeRecord() {
  return state.episodes.find(
    (episode) => Number(episode.episode_index) === state.currentEpisode,
  ) || null;
}

function setStatusPill(element, text, kind = 'neutral') {
  element.textContent = text;
  element.className = `status-pill ${kind}`;
}

function renderSummary() {
  const summary = state.summary;
  el.datasetPath.textContent = summary.input_path;
  el.datasetPath.title = summary.input_path;
  const cleaned = summary.cleaned ? ' · cleaned' : '';
  el.summaryBadge.textContent =
    `${Number(summary.episode_count).toLocaleString()} episodes · ` +
    `${Number(summary.total_steps).toLocaleString()} steps${cleaned}`;
}

function populateEpisodeSelect() {
  el.episodeSelect.replaceChildren();
  for (const episode of state.episodes) {
    const option = document.createElement('option');
    const source = episode.source_episode === null || episode.source_episode === undefined
      ? ''
      : ` · source ${episode.source_episode}`;
    const valid = episode.valid_ratio === null || episode.valid_ratio === undefined
      ? ''
      : ` · ${percent(episode.valid_ratio)} valid`;
    option.value = String(episode.episode_index);
    option.textContent =
      `${episode.episode_index}${source} · ${Number(episode.length).toLocaleString()} frames${valid}`;
    el.episodeSelect.appendChild(option);
  }
}

function updateEpisodeNavigation() {
  el.episodeSelect.value = String(state.currentEpisode);
  el.previousEpisode.disabled = state.currentEpisode <= 0;
  el.nextEpisode.disabled = state.currentEpisode >= state.episodes.length - 1;
}

function renderMetrics() {
  const metrics = state.metrics;
  const entries = [
    ['Frames', Number(metrics.length).toLocaleString()],
    ['Duration', `${numberText(metrics.duration_sec, 2)} s`],
    ['Valid', percent(metrics.valid_ratio)],
    ['Tracking', percent(metrics.tracking_ratio)],
    ['Clutch', percent(metrics.clutch_ratio)],
    ['Motor accepted', percent(metrics.motor_accepted_ratio)],
    ['Active motion', percent(metrics.active_motion_ratio)],
    ['Camera gaps', metrics.camera_sequence_gaps ?? '—'],
  ];
  el.metricsGrid.replaceChildren();
  for (const [label, value] of entries) {
    const card = document.createElement('div');
    card.className = 'metric';
    const labelNode = document.createElement('div');
    labelNode.className = 'metric-label';
    labelNode.textContent = label;
    const valueNode = document.createElement('div');
    valueNode.className = 'metric-value';
    valueNode.textContent = String(value);
    card.append(labelNode, valueNode);
    el.metricsGrid.appendChild(card);
  }

  if (metrics.metadata_counts_match === false) {
    el.metadataWarning.textContent =
      `Stored episode metadata says ${metrics.metadata_valid_steps} valid / ` +
      `${metrics.metadata_invalid_steps} invalid, but the per-step validity mask disagrees.`;
    el.metadataWarning.classList.remove('hidden');
  } else {
    el.metadataWarning.textContent = '';
    el.metadataWarning.classList.add('hidden');
  }

  el.fpsStatus.textContent = `${numberText(metrics.inferred_fps, 1)} FPS`;
  const source = metrics.source_episode === null || metrics.source_episode === undefined
    ? ''
    : ` · source episode ${metrics.source_episode}`;
  el.episodeSubtitle.textContent =
    `Episode ${metrics.episode_index}${source} · global rows ${metrics.start}–${metrics.end - 1}`;
}

function renderEvents() {
  el.eventCount.textContent = String(state.events.length);
  el.eventList.replaceChildren();
  el.eventMarkers.replaceChildren();
  const denominator = Math.max(1, currentLength() - 1);

  for (const event of state.events) {
    const marker = document.createElement('span');
    marker.className = `event-marker ${event.type}`;
    marker.style.left = `${(Number(event.idx) / denominator) * 100}%`;
    marker.title = `Frame ${event.idx}: ${event.label}`;
    el.eventMarkers.appendChild(marker);
  }

  if (!state.events.length) {
    const empty = document.createElement('div');
    empty.className = 'empty-message';
    empty.textContent = 'No transitions, gaps, or action jumps detected.';
    el.eventList.appendChild(empty);
    return;
  }

  for (const event of state.events.slice(0, 300)) {
    const row = document.createElement('button');
    row.className = 'event-row';
    row.type = 'button';
    row.addEventListener('click', () => setFrame(Number(event.idx), true));

    const type = document.createElement('span');
    type.className = `event-type ${event.type}`;
    type.textContent = event.type.replaceAll('_', ' ');
    const label = document.createElement('span');
    label.className = 'event-label';
    label.textContent = event.label;
    const frame = document.createElement('span');
    frame.className = 'event-frame';
    frame.textContent = `#${event.idx}`;
    row.append(type, label, frame);
    el.eventList.appendChild(row);
  }
}

function schemaKeyOrder() {
  return (state.schema?.keys || []).map((item) => item.key);
}

function selectedKeysInOrder() {
  return schemaKeyOrder().filter((key) => state.selectedKeys.has(key));
}

function updatePresetState() {
  const selected = [...state.selectedKeys].sort().join('\u0000');
  for (const button of el.presetButtons.querySelectorAll('button[data-group]')) {
    const group = state.schema?.key_groups?.[button.dataset.group] || [];
    button.classList.toggle('active', [...group].sort().join('\u0000') === selected);
  }
}

function setSelectedKeys(keys) {
  const allowed = new Set(schemaKeyOrder());
  state.selectedKeys = new Set(keys.filter((key) => allowed.has(key)));
  for (const checkbox of el.keyList.querySelectorAll('input[type="checkbox"]')) {
    checkbox.checked = state.selectedKeys.has(checkbox.value);
  }
  updatePresetState();
  scheduleSignals(true);
}

function populateSignalControls() {
  el.presetButtons.replaceChildren();
  const groupNames = ['Core', 'State', 'Commands', 'Health', 'Timing', 'All'];
  for (const groupName of groupNames) {
    const group = state.schema.key_groups?.[groupName] || [];
    if (!group.length) {
      continue;
    }
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.group = groupName;
    button.textContent = groupName;
    button.addEventListener('click', () => setSelectedKeys(group));
    el.presetButtons.appendChild(button);
  }
  const clearButton = document.createElement('button');
  clearButton.type = 'button';
  clearButton.textContent = 'Clear';
  clearButton.addEventListener('click', () => setSelectedKeys([]));
  el.presetButtons.appendChild(clearButton);

  el.keyList.replaceChildren();
  for (const keyInfo of state.schema.keys || []) {
    const label = document.createElement('label');
    label.className = 'key-option';
    label.title = `${keyInfo.group} · ${keyInfo.channels.join(', ')}`;
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.value = keyInfo.key;
    checkbox.addEventListener('change', () => {
      if (checkbox.checked) {
        state.selectedKeys.add(checkbox.value);
      } else {
        state.selectedKeys.delete(checkbox.value);
      }
      updatePresetState();
      scheduleSignals(true);
    });
    const text = document.createElement('span');
    text.textContent = keyInfo.key;
    label.append(checkbox, text);
    el.keyList.appendChild(label);
  }

  setSelectedKeys(state.schema.key_groups?.Core || []);
}

function estimatedTimeForFrame(frame) {
  const length = currentLength();
  if (length <= 1 || !state.metrics) {
    return 0;
  }
  return Number(state.metrics.duration_sec || 0) * Number(frame) / (length - 1);
}

function updateTimelineLabels() {
  const length = currentLength();
  el.timelineFrame.textContent =
    `Frame ${state.currentFrame.toLocaleString()} / ${Math.max(0, length - 1).toLocaleString()}`;
  el.timelineTime.textContent = `${estimatedTimeForFrame(state.currentFrame).toFixed(3)} s`;
}

function queueFrame(index, profile) {
  state.framePending = {
    episode: state.currentEpisode,
    index,
    profile,
    token: state.loadToken,
  };
  if (!state.frameRunning) {
    pumpFrames();
  }
}

async function pumpFrames() {
  state.frameRunning = true;
  while (state.framePending) {
    const target = state.framePending;
    state.framePending = null;
    try {
      const url =
        `/api/episode/${target.episode}/frame?idx=${target.index}` +
        `&profile=${encodeURIComponent(target.profile)}`;
      const response = await fetch(url, { cache: 'no-store' });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `Frame request failed (${response.status})`);
      }
      const blobUrl = URL.createObjectURL(await response.blob());
      const stillCurrent =
        target.token === state.loadToken &&
        target.episode === state.currentEpisode &&
        target.index === state.currentFrame;
      if (stillCurrent) {
        const previousUrl = state.frameObjectUrl;
        state.frameObjectUrl = blobUrl;
        el.cameraFrame.src = blobUrl;
        el.frameLoading.classList.add('hidden');
        if (previousUrl) {
          URL.revokeObjectURL(previousUrl);
        }
      } else {
        URL.revokeObjectURL(blobUrl);
      }
    } catch (error) {
      if (target.token === state.loadToken) {
        showError(error);
        el.frameLoading.textContent = 'Frame unavailable';
        el.frameLoading.classList.remove('hidden');
      }
    }
  }
  state.frameRunning = false;
}

function setFrame(index, forceSignals = false) {
  const length = currentLength();
  if (!length) {
    return;
  }
  const next = clamp(Math.round(Number(index) || 0), 0, length - 1);
  state.currentFrame = next;
  el.timeline.value = String(next);
  el.frameInput.value = String(next);
  updateTimelineLabels();
  setStatusPill(el.frameStatus, `Frame ${next}`, 'neutral');
  queueFrame(next, state.dragging || state.playing ? 'scrub' : 'full');
  scheduleSignals(forceSignals || !state.playing);
}

function stopPlayback() {
  if (!state.playing) {
    return;
  }
  state.playing = false;
  if (state.playHandle) {
    cancelAnimationFrame(state.playHandle);
    state.playHandle = 0;
  }
  el.playButton.textContent = 'Play';
  el.playButton.classList.remove('playing');
  queueFrame(state.currentFrame, 'full');
  scheduleSignals(true);
}

function playbackTick(now) {
  if (!state.playing) {
    return;
  }
  const speed = Number(el.speedSelect.value || 1);
  const fps = Math.max(1, Number(state.metrics?.inferred_fps || state.summary?.aligned_hz || 30));
  const elapsedSeconds = (now - state.playAnchorTime) / 1000;
  const target = state.playAnchorFrame + Math.floor(elapsedSeconds * fps * speed);
  if (target >= currentLength() - 1) {
    setFrame(currentLength() - 1, true);
    stopPlayback();
    return;
  }
  if (target !== state.currentFrame) {
    setFrame(target, false);
  }
  state.playHandle = requestAnimationFrame(playbackTick);
}

function togglePlayback() {
  if (state.playing) {
    stopPlayback();
    return;
  }
  if (state.currentFrame >= currentLength() - 1) {
    setFrame(0, true);
  }
  state.playing = true;
  state.playAnchorTime = performance.now();
  state.playAnchorFrame = state.currentFrame;
  el.playButton.textContent = 'Pause';
  el.playButton.classList.add('playing');
  state.playHandle = requestAnimationFrame(playbackTick);
}

function latestValue(payload, key) {
  const channel = payload?.series?.[key]?.[0];
  if (!channel || !Array.isArray(channel.values) || !channel.values.length) {
    return null;
  }
  const value = channel.values[channel.values.length - 1];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function renderCurrentSample(payload) {
  const definitions = [
    ['alignment_valid', 'Alignment', 'VALID', 'INVALID'],
    ['tracking', 'Tracking', 'ON', 'OFF'],
    ['clutch', 'Clutch', 'ON', 'OFF'],
    ['motor_command_accepted', 'Motor command', 'ACCEPTED', 'REJECTED'],
    ['grasp_label', 'Grasp', 'ON', 'OFF'],
  ];
  el.sampleGrid.replaceChildren();

  for (const [key, label, onText, offText] of definitions) {
    const value = latestValue(payload, key);
    const card = document.createElement('div');
    const enabled = value !== null && value > 0.5;
    card.className = `sample-flag ${value === null ? '' : enabled ? 'on' : 'off'}`;
    const labelNode = document.createElement('span');
    labelNode.textContent = label;
    const valueNode = document.createElement('strong');
    valueNode.textContent = value === null ? '—' : enabled ? onText : offText;
    card.append(labelNode, valueNode);
    el.sampleGrid.appendChild(card);
  }

  for (const [key, label] of [['control_age_ns', 'Control age'], ['motor_age_ns', 'Motor age']]) {
    const value = latestValue(payload, key);
    const card = document.createElement('div');
    card.className = 'sample-flag';
    const labelNode = document.createElement('span');
    labelNode.textContent = label;
    const valueNode = document.createElement('strong');
    valueNode.textContent = value === null ? '—' : `${value.toFixed(1)} ms`;
    card.append(labelNode, valueNode);
    el.sampleGrid.appendChild(card);
  }

  const valid = latestValue(payload, 'alignment_valid');
  if (valid === null) {
    setStatusPill(el.frameStatus, `Frame ${state.currentFrame}`, 'neutral');
  } else if (valid > 0.5) {
    setStatusPill(el.frameStatus, `Frame ${state.currentFrame} · VALID`, 'good');
  } else {
    setStatusPill(el.frameStatus, `Frame ${state.currentFrame} · INVALID`, 'bad');
  }
}

function finiteValues(channels) {
  const values = [];
  for (const channel of channels) {
    for (const value of channel.values || []) {
      if (typeof value === 'number' && Number.isFinite(value)) {
        values.push(value);
      }
    }
  }
  return values;
}

function renderSignalPlot(payload) {
  state.lastSignalPayload = payload;
  const keys = selectedKeysInOrder().filter((key) => payload.series?.[key]);
  const canvas = el.signalPlot;
  const context = canvas.getContext('2d');
  el.emptyPlot.classList.toggle('hidden', keys.length > 0);
  el.signalLegend.replaceChildren();
  if (!keys.length) {
    canvas.width = 1;
    canvas.height = 1;
    canvas.style.height = '330px';
    return;
  }

  const width = Math.max(520, el.plotWrap.clientWidth - 2);
  const rowHeight = 132;
  const topPadding = 13;
  const bottomPadding = 25;
  const height = Math.max(330, topPadding + keys.length * rowHeight + bottomPadding);
  const ratio = Math.max(1, window.devicePixelRatio || 1);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);

  const left = 62;
  const right = 14;
  const plotWidth = width - left - right;
  let colorIndex = 0;

  keys.forEach((key, rowIndex) => {
    const channels = payload.series[key] || [];
    const values = finiteValues(channels);
    const rowTop = topPadding + rowIndex * rowHeight;
    const plotTop = rowTop + 26;
    const plotHeight = rowHeight - 39;
    let minimum = values.length ? Math.min(...values) : -1;
    let maximum = values.length ? Math.max(...values) : 1;
    if (Math.abs(maximum - minimum) < 1e-12) {
      const padding = Math.max(0.5, Math.abs(maximum) * 0.1);
      minimum -= padding;
      maximum += padding;
    } else {
      const padding = (maximum - minimum) * 0.07;
      minimum -= padding;
      maximum += padding;
    }

    context.fillStyle = rowIndex % 2 ? 'rgba(17, 39, 60, 0.22)' : 'rgba(8, 19, 31, 0.16)';
    context.fillRect(0, rowTop, width, rowHeight - 2);
    context.fillStyle = '#dce8f3';
    context.font = '600 11px ui-monospace, SFMono-Regular, Menlo, monospace';
    context.fillText(key, 9, rowTop + 17);

    context.strokeStyle = '#20374d';
    context.lineWidth = 1;
    for (let grid = 0; grid <= 4; grid += 1) {
      const x = left + plotWidth * grid / 4;
      context.beginPath();
      context.moveTo(x, plotTop);
      context.lineTo(x, plotTop + plotHeight);
      context.stroke();
    }
    for (let grid = 0; grid <= 2; grid += 1) {
      const y = plotTop + plotHeight * grid / 2;
      context.beginPath();
      context.moveTo(left, y);
      context.lineTo(width - right, y);
      context.stroke();
    }

    if (minimum < 0 && maximum > 0) {
      const zeroY = plotTop + (maximum / (maximum - minimum)) * plotHeight;
      context.strokeStyle = '#496078';
      context.beginPath();
      context.moveTo(left, zeroY);
      context.lineTo(width - right, zeroY);
      context.stroke();
    }

    context.fillStyle = '#758ba0';
    context.font = '9px ui-monospace, SFMono-Regular, Menlo, monospace';
    context.fillText(maximum.toPrecision(3), 7, plotTop + 6);
    context.fillText(minimum.toPrecision(3), 7, plotTop + plotHeight);

    channels.forEach((channel) => {
      const color = TRACE_COLORS[colorIndex % TRACE_COLORS.length];
      colorIndex += 1;
      const series = channel.values || [];
      context.strokeStyle = color;
      context.lineWidth = 1.55;
      context.beginPath();
      let started = false;
      for (let index = 0; index < series.length; index += 1) {
        const value = series[index];
        if (typeof value !== 'number' || !Number.isFinite(value)) {
          started = false;
          continue;
        }
        const x = left + (series.length <= 1 ? plotWidth : plotWidth * index / (series.length - 1));
        const y = plotTop + (maximum - value) / (maximum - minimum) * plotHeight;
        if (!started) {
          context.moveTo(x, y);
          started = true;
        } else {
          context.lineTo(x, y);
        }
      }
      context.stroke();

      const last = [...series].reverse().find(
        (value) => typeof value === 'number' && Number.isFinite(value),
      );
      const chip = document.createElement('span');
      chip.className = 'legend-chip';
      const swatch = document.createElement('span');
      swatch.className = 'legend-swatch';
      swatch.style.background = color;
      const label = document.createElement('span');
      label.textContent = `${channel.name}: ${last === undefined ? '—' : Number(last).toPrecision(4)}`;
      chip.append(swatch, label);
      el.signalLegend.appendChild(chip);
    });

    context.strokeStyle = '#6d88a3';
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(width - right, plotTop);
    context.lineTo(width - right, plotTop + plotHeight);
    context.stroke();
  });

  const timestamps = payload.timestamps || [];
  const startTime = timestamps.length ? timestamps[0] : 0;
  const endTime = timestamps.length ? timestamps[timestamps.length - 1] : 0;
  context.fillStyle = '#758ba0';
  context.font = '9px ui-monospace, SFMono-Regular, Menlo, monospace';
  context.fillText(`${Number(startTime).toFixed(2)} s`, left, height - 8);
  const endLabel = `${Number(endTime).toFixed(2)} s`;
  context.fillText(endLabel, width - right - context.measureText(endLabel).width, height - 8);
}

async function updateSignals() {
  window.clearTimeout(state.signalTimer);
  state.signalTimer = 0;
  if (!state.schema) {
    return;
  }
  state.lastSignalRequest = performance.now();
  const token = ++state.signalToken;
  const episode = state.currentEpisode;
  const frame = state.currentFrame;
  const history = Math.max(1, Number(el.historySelect.value || 180));
  const start = Math.max(0, frame - history + 1);
  const end = frame + 1;
  const keys = [...new Set([...selectedKeysInOrder(), ...SAMPLE_KEYS])]
    .filter((key) => schemaKeyOrder().includes(key));
  if (!keys.length) {
    renderSignalPlot({ series: {}, timestamps: [] });
    renderCurrentSample({ series: {} });
    return;
  }

  const query = new URLSearchParams({
    keys: keys.join(','),
    start: String(start),
    end: String(end),
    stride: '1',
  });
  try {
    const payload = await apiGet(`/api/episode/${episode}/signals?${query}`);
    if (
      token !== state.signalToken ||
      episode !== state.currentEpisode ||
      frame !== state.currentFrame
    ) {
      return;
    }
    renderSignalPlot(payload);
    renderCurrentSample(payload);
  } catch (error) {
    if (token === state.signalToken) {
      showError(error);
    }
  }
}

function scheduleSignals(force = false) {
  if (!state.schema) {
    return;
  }
  const elapsed = performance.now() - state.lastSignalRequest;
  const minimumInterval = state.playing ? 120 : 25;
  if (force || elapsed >= minimumInterval) {
    updateSignals();
    return;
  }
  window.clearTimeout(state.signalTimer);
  state.signalTimer = window.setTimeout(updateSignals, minimumInterval - elapsed);
}

async function loadEpisode(episodeIndex) {
  stopPlayback();
  clearError();
  const episode = Number(episodeIndex);
  if (!Number.isInteger(episode) || episode < 0 || episode >= state.episodes.length) {
    throw new Error(`Episode ${episodeIndex} is out of range.`);
  }
  state.currentEpisode = episode;
  state.currentFrame = 0;
  state.schema = null;
  state.metrics = null;
  state.events = [];
  state.lastSignalPayload = null;
  state.selectedKeys.clear();
  const token = ++state.loadToken;
  ++state.signalToken;
  el.frameLoading.textContent = 'Loading frame…';
  el.frameLoading.classList.remove('hidden');
  updateEpisodeNavigation();

  const [schema, metrics, eventPayload] = await Promise.all([
    apiGet(`/api/episode/${episode}/schema`),
    apiGet(`/api/episode/${episode}/metrics`),
    apiGet(`/api/episode/${episode}/events`),
  ]);
  if (token !== state.loadToken) {
    return;
  }

  state.schema = schema;
  state.metrics = metrics;
  state.events = eventPayload.events || [];
  const maxFrame = Math.max(0, Number(schema.length) - 1);
  el.timeline.max = String(maxFrame);
  el.timeline.value = '0';
  el.frameInput.max = String(maxFrame);
  el.frameInput.value = '0';
  renderMetrics();
  renderEvents();
  populateSignalControls();
  updateTimelineLabels();
  const url = new URL(window.location.href);
  url.searchParams.set('episode', String(episode));
  window.history.replaceState({}, '', url);
  setFrame(0, true);
}

function bindControls() {
  el.episodeSelect.addEventListener('change', () => {
    loadEpisode(Number(el.episodeSelect.value)).catch(showError);
  });
  el.previousEpisode.addEventListener('click', () => {
    loadEpisode(state.currentEpisode - 1).catch(showError);
  });
  el.nextEpisode.addEventListener('click', () => {
    loadEpisode(state.currentEpisode + 1).catch(showError);
  });
  el.frameInput.addEventListener('change', () => setFrame(Number(el.frameInput.value), true));
  el.playButton.addEventListener('click', togglePlayback);
  el.speedSelect.addEventListener('change', () => {
    if (state.playing) {
      state.playAnchorTime = performance.now();
      state.playAnchorFrame = state.currentFrame;
    }
  });
  el.historySelect.addEventListener('change', () => scheduleSignals(true));

  el.timeline.addEventListener('pointerdown', () => {
    state.dragging = true;
  });
  el.timeline.addEventListener('input', () => setFrame(Number(el.timeline.value), false));
  const finishScrub = () => {
    if (!state.dragging) {
      return;
    }
    state.dragging = false;
    setFrame(Number(el.timeline.value), true);
  };
  el.timeline.addEventListener('pointerup', finishScrub);
  el.timeline.addEventListener('pointercancel', finishScrub);
  el.timeline.addEventListener('change', finishScrub);

  window.addEventListener('keydown', (event) => {
    const target = event.target;
    if (target instanceof HTMLInputElement || target instanceof HTMLSelectElement) {
      return;
    }
    if (event.code === 'Space') {
      event.preventDefault();
      togglePlayback();
    } else if (event.key === 'ArrowLeft') {
      event.preventDefault();
      stopPlayback();
      setFrame(state.currentFrame - 1, true);
    } else if (event.key === 'ArrowRight') {
      event.preventDefault();
      stopPlayback();
      setFrame(state.currentFrame + 1, true);
    }
  });

  let resizeTimer = 0;
  window.addEventListener('resize', () => {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => {
      if (state.lastSignalPayload) {
        renderSignalPlot(state.lastSignalPayload);
      }
    }, 100);
  });

  window.addEventListener('beforeunload', () => {
    if (state.frameObjectUrl) {
      URL.revokeObjectURL(state.frameObjectUrl);
    }
  });
}

async function initialize() {
  bindControls();
  try {
    const [summary, episodePayload] = await Promise.all([
      apiGet('/api/dataset/summary'),
      apiGet('/api/episodes'),
    ]);
    state.summary = summary;
    state.episodes = episodePayload.episodes || [];
    if (!state.episodes.length) {
      throw new Error('Dataset has no committed episodes.');
    }
    renderSummary();
    populateEpisodeSelect();

    const urlEpisode = Number(new URLSearchParams(window.location.search).get('episode'));
    const requested = Number.isInteger(urlEpisode) && urlEpisode >= 0
      ? urlEpisode
      : Number(summary.requested_episode || 0);
    const initialEpisode = clamp(requested, 0, state.episodes.length - 1);
    await loadEpisode(initialEpisode);
  } catch (error) {
    showError(error);
  }
}

initialize();
