const cfg = window.VOICEPI || {
  sampleRate: 16000,
  vadThreshold: 0.018,
  bargeInThreshold: 0.045,
  vadHoldMs: 650,
  minSpeechMs: 180,
  prerollMs: 450,
  audioChunkMs: 80
};

const els = {
  connBadge: document.getElementById('connBadge'),
  micBadge: document.getElementById('micBadge'),
  talkBadge: document.getElementById('talkBadge'),
  cameraBadge: document.getElementById('cameraBadge'),
  visionBadge: document.getElementById('visionBadge'),
  cameraStream: document.getElementById('cameraStream'),
  refreshVision: document.getElementById('refreshVision'),
  visionCurrent: document.getElementById('visionCurrent'),
  visionLogs: document.getElementById('visionLogs'),
  startBtn: document.getElementById('startBtn'),
  stopBtn: document.getElementById('stopBtn'),
  interruptBtn: document.getElementById('interruptBtn'),
  micMeter: document.getElementById('micMeter'),
  partial: document.getElementById('partial'),
  conversation: document.getElementById('conversation'),
  logs: document.getElementById('logs'),
  clearLogs: document.getElementById('clearLogs'),
  manualForm: document.getElementById('manualForm'),
  manualText: document.getElementById('manualText'),
  vadThreshold: document.getElementById('vadThreshold'),
  bargeInThreshold: document.getElementById('bargeInThreshold'),
  vadHold: document.getElementById('vadHold'),
  minSpeechMs: document.getElementById('minSpeechMs'),
  xaiRationale: document.getElementById('xaiRationale'),
  xaiRaw: document.getElementById('xaiRaw'),
  xaiMetrics: document.getElementById('xaiMetrics'),
  xaiPrompt: document.getElementById('xaiPrompt'),
  xaiRobot: document.getElementById('xaiRobot'),
  sysBadge: document.getElementById('sysBadge'),
  sysTempRow: document.getElementById('sysTempRow'),
  sysTempVal: document.getElementById('sysTempVal'),
  sysTempBar: document.getElementById('sysTempBar'),
  sysCpuRow: document.getElementById('sysCpuRow'),
  sysCpuVal: document.getElementById('sysCpuVal'),
  sysCpuBar: document.getElementById('sysCpuBar'),
  sysRamRow: document.getElementById('sysRamRow'),
  sysRamVal: document.getElementById('sysRamVal'),
  sysRamBar: document.getElementById('sysRamBar'),
  sysExtra: document.getElementById('sysExtra')
};

let socket;
let socketOpen = false;
let mediaStream;
let inputContext;
let playbackContext;
let processor;
let sourceNode;
let active = false;
let speaking = false;
let speechHoldUntil = 0;
let speechCandidateSince = 0;
let preroll = [];
let audioQueue = [];
let currentSource = null;
let playing = false;
let currentAssistantMsg = null;
let lastRaw = '';

function setBadge(el, text, cls = '') {
  el.className = `badge ${cls}`.trim();
  el.textContent = text;
}

function appendLog(obj) {
  const line = typeof obj === 'string' ? obj : JSON.stringify(obj, null, 0);
  els.logs.textContent += line + '\n';
  els.logs.scrollTop = els.logs.scrollHeight;
}

function updateCameraStatus(payload) {
  const cam = payload.camera || payload || {};
  if (!els.cameraBadge) return;
  if (!cam.enabled) setBadge(els.cameraBadge, 'camera: disabled');
  else if (cam.ok) setBadge(els.cameraBadge, `camera: on (${cam.size ? cam.size.join('×') : 'stream'})`, 'ok');
  else setBadge(els.cameraBadge, `camera: error${cam.error ? ' · ' + cam.error : ''}`, 'danger');

  const vision = payload.vision || {};
  if (els.visionBadge) {
    if (!vision.enabled) setBadge(els.visionBadge, 'vision: disabled');
    else setBadge(els.visionBadge, `vision: ${vision.backend || 'ready'}`, 'ok');
  }
  if (vision.latest) updateVision(vision.latest);
}

function formatObservation(data) {
  const objects = Array.isArray(data.objects) ? data.objects : [];
  const scene = data.scene || {};
  const motion = data.motion || {};
  const objectLines = objects.length
    ? objects.slice(0, 8).map((obj, idx) => {
        const conf = typeof obj.confidence === 'number' ? obj.confidence.toFixed(2) : (obj.confidence || 'n/a');
        const zone = obj.zone ? ` zone=${obj.zone}` : '';
        const area = typeof obj.area_ratio === 'number' ? ` area=${obj.area_ratio}` : '';
        const box = obj.bbox ? ` bbox=${JSON.stringify(obj.bbox)}` : '';
        return `object_${idx + 1}: ${obj.label || 'object'} confidence=${conf}${zone}${area}${box}`;
      })
    : [];
  const closeObstacles = Array.isArray(scene.close_obstacles) ? scene.close_obstacles : [];
  return [
    `time: ${data.iso || 'unknown'}`,
    `backend: ${data.backend || 'unknown'}`,
    `summary: ${data.summary || ''}`,
    `confidence: ${typeof data.confidence === 'number' ? data.confidence.toFixed(2) : data.confidence || 'n/a'}`,
    scene.attention ? `scene_attention: ${scene.attention}` : null,
    scene.person_count !== undefined ? `person_count: ${scene.person_count}` : null,
    scene.object_count !== undefined ? `objects_count: ${scene.object_count}` : (objects.length ? `objects_count: ${objects.length}` : null),
    scene.object_zones ? `object_zones: ${JSON.stringify(scene.object_zones)}` : null,
    closeObstacles.length ? `close_obstacles: ${JSON.stringify(closeObstacles)}` : null,
    motion.enabled !== undefined ? `motion_enabled: ${motion.enabled}` : null,
    motion.detected !== undefined ? `motion_detected: ${motion.detected}` : null,
    motion.zone ? `motion_zone: ${motion.zone}` : null,
    motion.changed_area_ratio !== undefined ? `motion_area: ${motion.changed_area_ratio}` : null,
    ...objectLines,
    data.error ? `error: ${data.error}` : null
  ].filter(Boolean).join('\n');
}

let lastVisionLogLine = '';
let lastVisionLogAt = 0;
function updateVision(data) {
  if (!data) return;
  if (els.visionCurrent) els.visionCurrent.textContent = formatObservation(data);
  if (els.visionBadge) setBadge(els.visionBadge, data.ok ? `vision: ${data.backend || 'ok'}` : 'vision: error', data.ok ? 'ok' : 'danger');
  if (els.visionLogs) {
    const line = `[${data.iso || new Date().toISOString()}] ${data.summary || JSON.stringify(data)}`;
    const now = Date.now();
    const changed = line.replace(/^\[[^\]]+\]\s*/, '') !== lastVisionLogLine;
    const important = data.reason && data.reason !== 'poll';
    if (important || changed || now - lastVisionLogAt > 10000) {
      lastVisionLogLine = line.replace(/^\[[^\]]+\]\s*/, '');
      lastVisionLogAt = now;
      els.visionLogs.textContent += line + '\n';
      els.visionLogs.scrollTop = els.visionLogs.scrollHeight;
    }
  }
}
function appendMessage(role, text = '') {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  const label = document.createElement('span');
  label.className = 'role';
  label.textContent = role === 'user' ? 'You' : 'Assistant';
  const body = document.createElement('span');
  body.className = 'body';
  body.textContent = text;
  div.appendChild(label);
  div.appendChild(body);
  els.conversation.appendChild(div);
  els.conversation.scrollTop = els.conversation.scrollHeight;
  return body;
}

function setTalking(text, cls = '') {
  setBadge(els.talkBadge, `dialogue: ${text}`, cls);
}

function sendEvent(event, data = {}) {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify({ event, data }));
}

function sendPcmBuffer(buf) {
  if (!socket || socket.readyState !== WebSocket.OPEN || !buf) return;
  socket.send(buf.slice(0));
}

function initSocket() {
  const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
  socket = new WebSocket(`${scheme}://${window.location.host}/ws`);
  socket.binaryType = 'arraybuffer';

  socket.addEventListener('open', () => {
    socketOpen = true;
    setBadge(els.connBadge, 'server: connected', 'ok');
  });
  socket.addEventListener('close', () => {
    socketOpen = false;
    setBadge(els.connBadge, 'server: disconnected', 'danger');
  });
  socket.addEventListener('error', () => setBadge(els.connBadge, 'server: error', 'danger'));
  socket.addEventListener('message', event => {
    let envelope;
    try { envelope = JSON.parse(event.data); } catch (err) { appendLog({ event: 'bad_ws_message', error: String(err) }); return; }
    handleServerEvent(envelope.event, envelope.data || {});
  });
}

function handleServerEvent(event, data) {
  if (event === 'server_ready') appendLog({ event: 'server_ready', ...data });
  else if (event === 'camera_status') updateCameraStatus(data);
  else if (event === 'vision_update') updateVision(data);
  else if (event === 'fatal_error') appendLog({ event: 'fatal_error', ...data });
  else if (event === 'log') appendLog(data);
  else if (event === 'stt_partial') { els.partial.textContent = data.text ? `… ${data.text}` : ''; }
  else if (event === 'stt_rejected') {
    els.partial.textContent = '';
    appendLog({ event: 'stt_rejected', ...data });
  }
  else if (event === 'stt_final') {
    els.partial.textContent = '';
    appendMessage('user', data.text);
  }
  else if (event === 'turn_start') {
    currentAssistantMsg = appendMessage('assistant', '');
    lastRaw = '';
    els.xaiRaw.textContent = '';
    els.xaiRationale.textContent = '';
    els.xaiMetrics.textContent = '';
    els.xaiPrompt.textContent = '';
    if (els.xaiRobot) els.xaiRobot.textContent = '';
    setTalking('thinking', 'warn');
  }
  else if (event === 'llm_token') {
    lastRaw = data.raw || (lastRaw + (data.token || ''));
    els.xaiRaw.textContent = lastRaw;
    if (currentAssistantMsg) {
      const answer = sanitizeStreamingMarkup(extractTagProgress(lastRaw, 'answer') || stripTagsAndRationale(lastRaw));
      currentAssistantMsg.textContent = answer.trim();
      els.conversation.scrollTop = els.conversation.scrollHeight;
    }
  }
  else if (event === 'xai_update') {
    if (data.rationale_partial) els.xaiRationale.textContent = sanitizeStreamingMarkup(data.rationale_partial); 
    if (data.rationale) els.xaiRationale.textContent = sanitizeStreamingMarkup(data.rationale);
    if (data.metrics) els.xaiMetrics.textContent = JSON.stringify(data.metrics, null, 2);
    if (data.messages) els.xaiPrompt.textContent = JSON.stringify(data.messages, null, 2);
  }
  else if (event === 'robot_update') {
    els.xaiRobot.textContent += JSON.stringify(data, null, 2) + '\n';
    els.xaiRobot.scrollTop = els.xaiRobot.scrollHeight;
    appendLog({ event: 'robot_update', ...data });
  }
  else if (event === 'tts_audio') {
    queueAudio(data.wav_b64);
    setTalking('speaking', 'ok');
  }
  else if (event === 'turn_done') {
    if (currentAssistantMsg && data.answer) currentAssistantMsg.textContent = data.answer;
    if (!playing && audioQueue.length === 0) setTalking('listening', 'ok');
  }
  else if (event === 'cancelled') {
    stopPlayback();
    setTalking('interrupted', 'warn');
    appendLog({ event: 'cancelled', ...data });
  }
  else if (event === 'turn_cancelled') {
    appendLog({ event: 'turn_cancelled', ...data });
  }
  else if (event === 'turn_error') {
    setTalking('error', 'danger');
    appendLog({ event: 'turn_error', ...data });
  }
  else appendLog({ event, data });
}

function extractTagProgress(raw, tag) {
  const startRe = new RegExp(`<${tag}\\s*>`, 'i');
  const endRe = new RegExp(`</${tag}\\s*>`, 'i');
  const start = raw.search(startRe);
  if (start < 0) return '';
  const afterStart = raw.slice(start).replace(startRe, '');
  const end = afterStart.search(endRe);
  return end >= 0 ? afterStart.slice(0, end) : afterStart;
}

function stripTagsAndRationale(text) {
  return sanitizeStreamingMarkup(text
    .replace(/<rationale\s*>[\s\S]*?<\/rationale\s*>/gi, '')
    .replace(/^\s*(Answer|Response)\s*:\s*/i, ''));
}

function sanitizeStreamingMarkup(text) {
  return String(text || '')
    .replace(/<rationale\s*>[\s\S]*?<\/rationale\s*>/gi, '')
    .replace(/<tool_call\s*>[\s\S]*?<\/tool_call\s*>/gi, '')
    .replace(/<[^>]*>/g, '')
    .replace(/<[^\s<>]*$/g, '')
    .replace(/[<>]/g, '')
    .trim();
}

function downsampleTo16k(float32, inputRate, outputRate) {
  if (inputRate === outputRate) return float32;
  const ratio = inputRate / outputRate;
  const newLen = Math.round(float32.length / ratio);
  const result = new Float32Array(newLen);
  let offset = 0;
  for (let i = 0; i < newLen; i++) {
    const nextOffset = Math.round((i + 1) * ratio);
    let accum = 0;
    let count = 0;
    for (let j = offset; j < nextOffset && j < float32.length; j++) {
      accum += float32[j];
      count++;
    }
    result[i] = count ? accum / count : 0;
    offset = nextOffset;
  }
  return result;
}

function floatTo16BitPCM(float32) {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function rms(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / Math.max(samples.length, 1));
}

function rememberPreroll(buf, level, isVoice) {
  // While audio is playing, ignore quiet speaker echo. Keep loud chunks so real barge-in still includes pre-roll.
  if (playing && !isVoice) return;
  preroll.push(buf.slice(0));
  const maxChunks = Math.max(2, Math.ceil((cfg.prerollMs || 450) / (cfg.audioChunkMs || 80)));
  while (preroll.length > maxChunks) preroll.shift();
}

async function startMic() {
  if (active) return;
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      channelCount: 1
    }
  });
  inputContext = new (window.AudioContext || window.webkitAudioContext)();
  playbackContext = playbackContext || new (window.AudioContext || window.webkitAudioContext)();
  sourceNode = inputContext.createMediaStreamSource(mediaStream);
  processor = inputContext.createScriptProcessor(4096, 1, 1);

  processor.onaudioprocess = event => {
    if (!active || !socket || socket.readyState !== WebSocket.OPEN) return;
    const input = event.inputBuffer.getChannelData(0);
    const level = rms(input);
    els.micMeter.value = Math.min(1, level * 12);

    const normalThreshold = parseFloat(els.vadThreshold.value || cfg.vadThreshold);
    const bargeThreshold = parseFloat(els.bargeInThreshold.value || cfg.bargeInThreshold || normalThreshold * 2.5);
    const threshold = playing ? bargeThreshold : normalThreshold;
    const holdMs = parseInt(els.vadHold.value || cfg.vadHoldMs, 10);
    const minSpeechMs = parseInt(els.minSpeechMs.value || cfg.minSpeechMs || 180, 10);
    const now = performance.now();
    const isVoice = level >= threshold;

    const pcmFloat = downsampleTo16k(input, inputContext.sampleRate, cfg.sampleRate);
    const pcm16 = floatTo16BitPCM(pcmFloat);
    const chunk = pcm16.buffer.slice(0);

    if (isVoice) {
      if (!speechCandidateSince) speechCandidateSince = now;
    } else {
      speechCandidateSince = 0;
    }

    const confirmedSpeech = isVoice && (now - speechCandidateSince >= minSpeechMs);

    if (!speaking) {
      rememberPreroll(chunk, level, isVoice);
      if (confirmedSpeech) {
        const wasPlaying = playing;
        if (wasPlaying) stopPlayback();
        speaking = true;
        speechHoldUntil = now + holdMs;
        setTalking(wasPlaying ? 'barge-in listening' : 'listening', 'ok');
        sendEvent('speech_start', { level, barge_in: wasPlaying });
        for (const oldChunk of preroll) sendPcmBuffer(oldChunk);
        preroll = [];
      }
      return;
    }

    sendPcmBuffer(chunk);
    if (isVoice) speechHoldUntil = now + holdMs;
    if (speaking && now > speechHoldUntil) {
      speaking = false;
      speechCandidateSince = 0;
      sendEvent('speech_end', {});
      setTalking('thinking', 'warn');
    }
  };

  sourceNode.connect(processor);
  processor.connect(inputContext.destination);
  active = true;
  els.startBtn.disabled = true;
  els.stopBtn.disabled = false;
  els.interruptBtn.disabled = false;
  setBadge(els.micBadge, 'microphone: on', 'ok');
  setTalking('listening', 'ok');
}

async function stopMic() {
  active = false;
  if (speaking) sendEvent('speech_end', {});
  speaking = false;
  speechCandidateSince = 0;
  preroll = [];
  stopPlayback();
  if (processor) processor.disconnect();
  if (sourceNode) sourceNode.disconnect();
  if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
  if (inputContext) await inputContext.close().catch(() => {});
  processor = null;
  sourceNode = null;
  mediaStream = null;
  inputContext = null;
  els.startBtn.disabled = false;
  els.stopBtn.disabled = true;
  els.interruptBtn.disabled = true;
  els.micMeter.value = 0;
  setBadge(els.micBadge, 'microphone: off');
  setTalking('idle');
}

function stopPlayback() {
  audioQueue = [];
  if (currentSource) {
    try { currentSource.stop(); } catch (_) {}
    currentSource = null;
  }
  playing = false;
}

function base64ToArrayBuffer(b64) {
  const binary = atob(b64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

async function queueAudio(wavB64) {
  if (!wavB64) return;
  playbackContext = playbackContext || new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await playbackContext.decodeAudioData(base64ToArrayBuffer(wavB64));
  audioQueue.push(audioBuffer);
  if (!playing) playNext();
}

function playNext() {
  if (!audioQueue.length) {
    playing = false;
    currentSource = null;
    if (active) setTalking('listening', 'ok');
    return;
  }
  playing = true;
  const buffer = audioQueue.shift();
  const src = playbackContext.createBufferSource();
  src.buffer = buffer;
  src.connect(playbackContext.destination);
  currentSource = src;
  src.onended = () => {
    if (currentSource === src) currentSource = null;
    playNext();
  };
  src.start();
}

for (const btn of document.querySelectorAll('.tab')) {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tabPanel').forEach(p => p.classList.add('hidden'));
    btn.classList.add('active');
    const id = btn.dataset.tab;
    document.getElementById(`xai${id.charAt(0).toUpperCase()}${id.slice(1)}`).classList.remove('hidden');
  });
}

els.startBtn.addEventListener('click', () => startMic().catch(err => {
  appendLog({ event: 'mic_error', error: String(err) });
  setBadge(els.micBadge, 'microphone: error', 'danger');
}));
els.stopBtn.addEventListener('click', stopMic);
els.interruptBtn.addEventListener('click', () => {
  stopPlayback();
  sendEvent('barge_in', { reason: 'manual_interrupt' });
});
els.clearLogs.addEventListener('click', () => { els.logs.textContent = ''; });
if (els.refreshVision) els.refreshVision.addEventListener('click', () => sendEvent('vision_analyze', {}));
if (els.cameraStream) {
  els.cameraStream.addEventListener('load', () => {
    if (els.cameraBadge && !els.cameraBadge.classList.contains('danger')) setBadge(els.cameraBadge, 'camera: stream loaded', 'ok');
  });
  els.cameraStream.addEventListener('error', () => {
    if (els.cameraBadge) setBadge(els.cameraBadge, 'camera: stream unavailable', 'danger');
  });
}
els.manualForm.addEventListener('submit', event => {
  event.preventDefault();
  const text = els.manualText.value.trim();
  if (!text) return;
  stopPlayback();
  sendEvent('manual_text', { text });
  els.manualText.value = '';
});

fetch('/camera/status')
  .then(r => r.json())
  .then(updateCameraStatus)
  .catch(err => appendLog({ event: 'camera_status_error', error: String(err) }));

// ---- Raspberry Pi system telemetry ----
function setMetric(rowEl, valEl, barEl, value, unit, severity, maxScale) {
  const sev = severity || (value === null || value === undefined ? 'unknown' : 'ok');
  if (rowEl) rowEl.className = `sysMetric ${sev}`;
  if (valEl) valEl.textContent = (value === null || value === undefined) ? 'n/a' : `${value}${unit}`;
  if (barEl) {
    const pct = (value === null || value === undefined) ? 0 : Math.max(0, Math.min(100, (value / maxScale) * 100));
    barEl.style.width = `${pct}%`;
  }
}

function renderSystemStats(data) {
  if (!data || !data.ok) {
    if (els.sysBadge) setBadge(els.sysBadge, 'system: unavailable');
    return;
  }
  const t = data.temperature || {};
  const c = data.cpu || {};
  const r = data.ram || {};
  setMetric(els.sysTempRow, els.sysTempVal, els.sysTempBar, t.value_c, ' °C', t.severity, 100);
  setMetric(els.sysCpuRow, els.sysCpuVal, els.sysCpuBar, c.value_pct, ' %', c.severity, 100);
  setMetric(els.sysRamRow, els.sysRamVal, els.sysRamBar, r.value_pct, ' %', r.severity, 100);

  const sevClass = data.severity === 'critical' ? 'danger' : (data.severity === 'warn' ? 'warn' : 'ok');
  const sevText = data.severity === 'critical' ? 'overheating/overloaded' : (data.severity === 'warn' ? 'high' : 'healthy');
  if (els.sysBadge) setBadge(els.sysBadge, `system: ${sevText}`, sevClass);

  if (els.sysExtra) {
    const bits = [];
    if (c.cores) bits.push(`${c.cores} cores`);
    if (c.freq_mhz) bits.push(`${Math.round(c.freq_mhz)} MHz`);
    if (Array.isArray(c.load_avg)) bits.push(`load ${c.load_avg.map(x => x.toFixed(2)).join(' / ')}`);
    if (r.used_mb && r.total_mb) bits.push(`${Math.round(r.used_mb)}/${Math.round(r.total_mb)} MB`);
    els.sysExtra.textContent = bits.join(' · ') || '';
  }
}

function pollSystemStats() {
  fetch('/system/stats')
    .then(r => r.json())
    .then(renderSystemStats)
    .catch(() => { if (els.sysBadge) setBadge(els.sysBadge, 'system: unavailable'); });
}

if (els.sysTempRow) {
  pollSystemStats();
  setInterval(pollSystemStats, 2000);
}

initSocket();