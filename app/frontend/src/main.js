/**
 * LTX-2 Studio — Main Application
 * SPA router, global state, API client, sidebar rendering.
 */
import './theme.css';

// ============================================================================
//  API CLIENT
// ============================================================================
export const api = {
  baseUrl: '',

  async get(path) {
    const res = await fetch(`${this.baseUrl}${path}`);
    if (!res.ok) throw new Error(`GET ${path}: ${res.status} ${res.statusText}`);
    return res.json();
  },

  async post(path, body, isFormData = false) {
    const opts = { method: 'POST' };
    if (isFormData) {
      opts.body = body;
    } else {
      opts.headers = { 'Content-Type': 'application/json' };
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(`${this.baseUrl}${path}`, opts);
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`POST ${path}: ${res.status} — ${text}`);
    }
    return res.json();
  },

  async put(path, body) {
    const res = await fetch(`${this.baseUrl}${path}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`PUT ${path}: ${res.status}`);
    return res.json();
  },

  async del(path) {
    const res = await fetch(`${this.baseUrl}${path}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(`DELETE ${path}: ${res.status}`);
    return res.json();
  },

  /**
   * Connect to a Server-Sent Events endpoint.
   * @param {string} path
   * @param {function} onMessage - Called with parsed JSON for each event
   * @param {function} onError - Called on error
   * @returns {EventSource}
   */
  sse(path, onMessage, onError) {
    const es = new EventSource(`${this.baseUrl}${path}`);
    es.onmessage = (e) => onMessage(JSON.parse(e.data));
    es.onerror = (e) => { if (onError) onError(e); es.close(); };
    return es;
  },
};


// ============================================================================
//  GLOBAL STATE
// ============================================================================
export const state = {
  currentPage: 'workspace',
  pipelines: [],
  selectedPipeline: null,
  pipelineParams: {},
  modelStatus: {},
  gpuAvailable: false,
  settings: {
    outputDir: '~/ltx-outputs',
    quantization: 'none',
    autoSaveHistory: true,
    enhancePrompt: false,
  },
  // Current generation job
  currentJob: null,

  // Listeners for state changes
  _listeners: [],
  onChange(fn) { this._listeners.push(fn); },
  notify(key) { this._listeners.forEach(fn => fn(key, this)); },
};


// ============================================================================
//  TOAST NOTIFICATIONS
// ============================================================================
export function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type === 'error' ? 'error' : ''}`;
  toast.innerHTML = `
    <div class="toast-dot"></div>
    <span class="toast-message">${message}</span>
  `;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(20px)';
    toast.style.transition = 'all 300ms ease';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}


// ============================================================================
//  SIDEBAR
// ============================================================================
const NAV_ITEMS = [
  { id: 'workspace', label: '[Workspace]', icon: 'terminal' },
  { id: 'history',   label: '[History]',   icon: 'clock-history' },
  { id: 'settings',  label: '[Settings]',  icon: 'gear' },
];

function renderHeader() {
  const header = document.getElementById('top-header');
  if (!header) return;
  header.innerHTML = `
    <div class="flex items-center gap-4">
      <div class="flex items-center gap-2 px-2">
        <sl-icon name="terminal" class="text-primary text-xl"></sl-icon>
        <h1 class="font-bold text-sm tracking-widest uppercase text-text-main">LTX-2</h1>
      </div>
    </div>

    <nav class="header-nav flex items-center gap-8" id="nav-links">
      ${NAV_ITEMS.map(item => `
        <a href="#${item.id}"
           data-page="${item.id}"
           class="flex items-center gap-2 transition-colors pb-1 border-b-2 ${state.currentPage === item.id ? 'border-primary text-primary' : 'border-transparent text-text-muted hover:text-text-main'}">
          <sl-icon name="${item.icon}" class="text-[14px]"></sl-icon>
          <span class="font-semibold text-xs tracking-wider uppercase">${item.label.replace('[', '').replace(']', '')}</span>
        </a>
      `).join('')}
    </nav>

    <div class="flex items-center gap-4 text-[10px] tracking-widest uppercase px-2">
      <div class="flex items-center gap-2 text-text-muted">
        <span>STATUS:</span>
        <span class="text-primary flex items-center gap-1 font-bold">
          <span class="status-dot ${state.gpuAvailable ? '' : 'offline'}" style="width:6px;height:6px;border-radius:50%;background:currentColor;"></span>
          ${state.gpuAvailable ? (state.modelStatus?.gpu_name || 'GPU READY') : 'CPU ONLY'}
        </span>
      </div>
    </div>
  `;

  // Bind nav click events
  header.querySelectorAll('[data-page]').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      navigateTo(link.dataset.page);
    });
  });
}


// ============================================================================
//  ROUTER
// ============================================================================
const pages = {};

export function registerPage(name, renderFn) {
  pages[name] = renderFn;
}

export function navigateTo(pageName) {
  if (!pages[pageName]) return;
  state.currentPage = pageName;
  window.location.hash = pageName;

  // Update header active state
  document.querySelectorAll('.header-nav [data-page]').forEach(link => {
    link.classList.toggle('text-primary', link.dataset.page === pageName);
    const span = link.querySelector('span');
    if (span) span.style.color = link.dataset.page === pageName ? 'var(--color-accent)' : '';
  });

  // Render the page
  const container = document.getElementById('page-content');
  container.innerHTML = '';
  container.className = 'main-content fade-in';
  pages[pageName](container);
}

function handleHashChange() {
  const hash = window.location.hash.slice(1) || 'workspace';
  if (pages[hash]) {
    navigateTo(hash);
  }
}


// ============================================================================
//  INITIALIZATION
// ============================================================================
async function loadInitialData() {
  try {
    // Try to load pipelines from backend
    const data = await api.get('/api/pipelines');
    state.pipelines = data.pipelines || [];
    if (state.pipelines.length > 0 && !state.selectedPipeline) {
      state.selectedPipeline = state.pipelines[0].id;
    }
  } catch {
    // Backend not running — use fallback pipeline definitions
    console.warn('Backend not available, using fallback pipeline data');
    state.pipelines = getFallbackPipelines();
    state.selectedPipeline = 'ti2vid_two_stages';
  }

  try {
    const status = await api.get('/api/models/status');
    state.modelStatus = status;
    state.gpuAvailable = status.gpu_available || false;
  } catch {
    state.gpuAvailable = false;
    state.modelStatus = {};
  }

  try {
    const settings = await api.get('/api/settings');
    Object.assign(state.settings, settings);
  } catch {
    // Use defaults
  }
}


/**
 * Fallback pipeline definitions for when backend is not available.
 * These match exactly the pipelines from ltx-pipelines.
 */
function getFallbackPipelines() {
  const videoGuiderDefaults = {
    cfg_scale: { value: 3.0, min: 0, max: 20, step: 0.1 },
    stg_scale: { value: 1.0, min: 0, max: 5, step: 0.1 },
    rescale_scale: { value: 0.7, min: 0, max: 1, step: 0.05 },
    modality_scale: { value: 3.0, min: 0, max: 10, step: 0.1 },
    skip_step: { value: 0, min: 0, max: 10, step: 1 },
    stg_blocks: { value: '28', type: 'text' },
  };
  const audioGuiderDefaults = {
    cfg_scale: { value: 7.0, min: 0, max: 20, step: 0.1 },
    stg_scale: { value: 1.0, min: 0, max: 5, step: 0.1 },
    rescale_scale: { value: 0.7, min: 0, max: 1, step: 0.05 },
    modality_scale: { value: 3.0, min: 0, max: 10, step: 0.1 },
    skip_step: { value: 0, min: 0, max: 10, step: 1 },
    stg_blocks: { value: '28', type: 'text' },
  };

  const baseGenParams = [
    { name: 'prompt', label: 'Prompt', type: 'textarea', group: 'prompt', required: true },
    { name: 'negative_prompt', label: 'Negative Prompt', type: 'textarea', group: 'prompt',
      default: 'blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, artifacts' },
    { name: 'seed', label: 'Seed (-1 for random)', type: 'int', default: -1, min: -1, max: 4294967295, group: 'generation' },
    { name: 'height', label: 'Height', type: 'int', default: 1024, min: 64, max: 2160, step: 64, group: 'generation' },
    { name: 'width', label: 'Width', type: 'int', default: 1536, min: 64, max: 3840, step: 64, group: 'generation' },
    { name: 'num_frames', label: 'Frames', type: 'int', default: 121, min: 9, max: 257, step: 8, group: 'generation',
      description: 'Must satisfy (F-1) % 8 == 0, e.g. 9, 17, 25, ..., 121, ..., 257' },
    { name: 'frame_rate', label: 'Frame Rate', type: 'float', default: 24.0, min: 1, max: 60, step: 1, group: 'generation' },
    { name: 'num_inference_steps', label: 'Steps', type: 'int', default: 30, min: 1, max: 100, step: 1, group: 'generation' },
    { name: 'enhance_prompt', label: 'Enhance Prompt', type: 'bool', default: false, group: 'generation' },
  ];

  const videoGuiderParams = Object.entries(videoGuiderDefaults).map(([key, v]) => ({
    name: `video_${key}`, label: `Video ${key.replace(/_/g, ' ')}`,
    type: v.type || 'float', default: v.value, min: v.min, max: v.max, step: v.step, group: 'video_guidance',
  }));

  const audioGuiderParams = Object.entries(audioGuiderDefaults).map(([key, v]) => ({
    name: `audio_${key}`, label: `Audio ${key.replace(/_/g, ' ')}`,
    type: v.type || 'float', default: v.value, min: v.min, max: v.max, step: v.step, group: 'audio_guidance',
  }));

  const guidedParams = [...baseGenParams, ...videoGuiderParams, ...audioGuiderParams];

  return [
    {
      id: 'ti2vid_two_stages', name: 'Text/Image → Video (2-Stage)',
      description: 'Production quality. Two-stage generation with upsampling. Recommended.',
      params: [...guidedParams,
        { name: 'images', label: 'Conditioning Images', type: 'image_list', group: 'conditioning' },
      ],
      requires: ['checkpoint', 'distilled_lora', 'spatial_upsampler', 'gemma'],
    },
    {
      id: 'ti2vid_two_stages_hq', name: 'Text/Image → Video (2-Stage HQ)',
      description: 'Same as 2-Stage but uses res_2s sampler for higher quality with fewer steps.',
      params: [...guidedParams,
        { name: 'images', label: 'Conditioning Images', type: 'image_list', group: 'conditioning' },
        { name: 'distilled_lora_strength_stage_1', label: 'Distilled LoRA Str. (Stage 1)', type: 'float', default: 0.25, min: 0, max: 1, step: 0.05, group: 'advanced' },
        { name: 'distilled_lora_strength_stage_2', label: 'Distilled LoRA Str. (Stage 2)', type: 'float', default: 0.5, min: 0, max: 1, step: 0.05, group: 'advanced' },
      ],
      requires: ['checkpoint', 'distilled_lora', 'spatial_upsampler', 'gemma'],
    },
    {
      id: 'ti2vid_one_stage', name: 'Text/Image → Video (1-Stage)',
      description: 'Single-stage, lower resolution. For educational/prototyping use.',
      params: [...guidedParams.map(p => {
        if (p.name === 'height') return { ...p, default: 512, step: 32 };
        if (p.name === 'width') return { ...p, default: 768, step: 32 };
        return p;
      }),
        { name: 'images', label: 'Conditioning Images', type: 'image_list', group: 'conditioning' },
      ],
      requires: ['checkpoint', 'gemma'],
    },
    {
      id: 'distilled', name: 'Distilled (Fast)',
      description: 'Fastest inference with 8 predefined sigma steps. No guidance needed.',
      params: [
        ...baseGenParams.filter(p => !['negative_prompt', 'num_inference_steps'].includes(p.name)),
        { name: 'images', label: 'Conditioning Images', type: 'image_list', group: 'conditioning' },
      ],
      requires: ['distilled_checkpoint', 'spatial_upsampler', 'gemma'],
    },
    {
      id: 'ic_lora', name: 'IC-LoRA (Video-to-Video)',
      description: 'Video-to-video with In-Context LoRA. Requires distilled model + IC-LoRA.',
      params: [
        ...baseGenParams.filter(p => !['negative_prompt', 'num_inference_steps'].includes(p.name)),
        { name: 'images', label: 'Conditioning Images', type: 'image_list', group: 'conditioning' },
        { name: 'video_conditioning', label: 'Reference Video', type: 'file', accept: 'video/*', group: 'conditioning', required: true },
        { name: 'video_conditioning_strength', label: 'Conditioning Strength', type: 'float', default: 1.0, min: 0, max: 1, step: 0.05, group: 'conditioning' },
        { name: 'conditioning_attention_strength', label: 'Attention Strength', type: 'float', default: 1.0, min: 0, max: 1, step: 0.05, group: 'conditioning' },
        { name: 'skip_stage_2', label: 'Skip Stage 2 (half res)', type: 'bool', default: false, group: 'advanced' },
      ],
      requires: ['distilled_checkpoint', 'spatial_upsampler', 'gemma', 'ic_lora'],
    },
    {
      id: 'keyframe_interpolation', name: 'Keyframe Interpolation',
      description: 'Interpolate between keyframe images with smooth transitions.',
      params: [...guidedParams,
        { name: 'images', label: 'Keyframe Images', type: 'image_list', group: 'conditioning', required: true,
          description: 'At least 2 keyframe images with frame indices' },
      ],
      requires: ['checkpoint', 'distilled_lora', 'spatial_upsampler', 'gemma'],
    },
    {
      id: 'a2vid_two_stage', name: 'Audio → Video (2-Stage)',
      description: 'Generate video driven by input audio. Audio is preserved in output.',
      params: [
        ...baseGenParams.filter(p => p.name !== 'negative_prompt'),
        { name: 'negative_prompt', label: 'Negative Prompt', type: 'textarea', group: 'prompt', default: '' },
        ...videoGuiderParams,
        { name: 'images', label: 'Conditioning Images', type: 'image_list', group: 'conditioning' },
        { name: 'audio_path', label: 'Input Audio', type: 'file', accept: 'audio/*', group: 'conditioning', required: true },
        { name: 'audio_start_time', label: 'Audio Start Time (s)', type: 'float', default: 0.0, min: 0, max: 3600, step: 0.1, group: 'conditioning' },
        { name: 'audio_max_duration', label: 'Audio Max Duration (s)', type: 'float', default: null, min: 0, max: 3600, step: 0.1, group: 'conditioning' },
      ],
      requires: ['checkpoint', 'distilled_lora', 'spatial_upsampler', 'gemma'],
    },
    {
      id: 'retake', name: 'Retake (Region Regeneration)',
      description: 'Regenerate a specific time region of an existing video.',
      params: [
        { name: 'video_path', label: 'Source Video', type: 'file', accept: 'video/*', group: 'input', required: true },
        { name: 'prompt', label: 'Prompt', type: 'textarea', group: 'prompt', required: true },
        { name: 'start_time', label: 'Start Time (s)', type: 'float', default: 0, min: 0, max: 3600, step: 0.1, group: 'input', required: true },
        { name: 'end_time', label: 'End Time (s)', type: 'float', default: 1.0, min: 0, max: 3600, step: 0.1, group: 'input', required: true },
        { name: 'seed', label: 'Seed', type: 'int', default: 10, min: 0, max: 4294967295, group: 'generation' },
        { name: 'regenerate_video', label: 'Regenerate Video', type: 'bool', default: true, group: 'generation' },
        { name: 'regenerate_audio', label: 'Regenerate Audio', type: 'bool', default: true, group: 'generation' },
        { name: 'enhance_prompt', label: 'Enhance Prompt', type: 'bool', default: false, group: 'generation' },
      ],
      requires: ['distilled_checkpoint', 'gemma'],
    },
  ];
}


// ============================================================================
//  BOOT
// ============================================================================
async function boot() {
  renderHeader();
  await loadInitialData();
  renderHeader(); // Re-render with GPU status

  // Dynamically import pages
  const { renderWorkspace } = await import('./pages/workspace.js');
  const { renderHistory } = await import('./pages/history.js');
  const { renderSettings } = await import('./pages/settings.js');

  registerPage('workspace', renderWorkspace);
  registerPage('history', renderHistory);
  registerPage('settings', renderSettings);

  window.addEventListener('hashchange', handleHashChange);
  handleHashChange();
}

boot().catch(err => {
  console.error('Boot failed:', err);
  showToast('BOOT_FAILED: ' + err.message, 'error');
});
