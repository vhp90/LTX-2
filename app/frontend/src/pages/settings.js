/**
 * LTX-2 Studio — Settings Page
 * Model management, API tokens, and application preferences.
 */
import { state, api, showToast } from '../main.js';

// ============================================================================
//  RENDER SETTINGS
// ============================================================================
export function renderSettings(container) {
  container.innerHTML = `
    <div style="overflow-y: auto; height: 100%;">
      <div class="settings-container">

        <!-- Header -->
        <div class="settings-header">
          <h1 class="font-display font-bold text-xl uppercase tracking-widest">Settings</h1>
          <p class="sublabel" style="margin-top: var(--space-2);">System configuration and model management</p>
        </div>

        <!-- MODEL STATUS -->
        <div class="settings-section">
          <div class="flex items-center justify-between">
            <h2 class="section-title">Model Status</h2>
            <button class="btn-ghost" id="refresh-models-btn">
              <sl-icon name="arrow-clockwise"></sl-icon> Refresh
            </button>
          </div>
          <div id="model-status-list" class="model-status-grid">
            ${renderModelStatus(state.modelStatus)}
          </div>
          <button class="btn-primary" id="download-models-btn" style="width: 100%; margin-top: var(--space-2);">
            <sl-icon name="download"></sl-icon>
            Download Missing Models
          </button>
          <div id="download-progress" class="hidden" style="margin-top: var(--space-2);">
            <sl-progress-bar id="dl-progress-bar" value="0"></sl-progress-bar>
            <span class="sublabel" id="dl-progress-label">Starting download...</span>
          </div>
        </div>

        <!-- API CREDENTIALS -->
        <div class="settings-section">
          <h2 class="section-title">API Credentials</h2>
          <div style="display: flex; flex-direction: column; gap: var(--space-3);">
            <div>
              <label class="label">HuggingFace Token</label>
              <sl-input id="hf-token" type="password" size="small" placeholder="hf_..." password-toggle
                value="${state.settings.hf_token || ''}" style="margin-top: var(--space-1);">
                <sl-icon name="key" slot="prefix"></sl-icon>
              </sl-input>
              <span class="sublabel" style="margin-top: 2px;">Required for downloading models from HuggingFace</span>
            </div>
            <div>
              <label class="label">CivitAI Token</label>
              <sl-input id="civitai-token" type="password" size="small" placeholder="Token..." password-toggle
                value="${state.settings.civitai_token || ''}" style="margin-top: var(--space-1);">
                <sl-icon name="key" slot="prefix"></sl-icon>
              </sl-input>
              <span class="sublabel" style="margin-top: 2px;">Required for downloading LoRAs from CivitAI</span>
            </div>
          </div>
        </div>

        <!-- MODEL PATHS -->
        <div class="settings-section">
          <h2 class="section-title">Model Paths</h2>
          <div style="display: flex; flex-direction: column; gap: var(--space-3);">
            <div>
              <label class="label">Models Directory</label>
              <sl-input id="models-dir" size="small" value="${state.settings.models_dir || '~/ltx-models'}" style="margin-top: var(--space-1);">
                <sl-icon name="folder2" slot="prefix"></sl-icon>
              </sl-input>
            </div>
            <div>
              <label class="label">Output Directory</label>
              <sl-input id="output-dir" size="small" value="${state.settings.output_dir || '~/ltx-outputs'}" style="margin-top: var(--space-1);">
                <sl-icon name="folder2" slot="prefix"></sl-icon>
              </sl-input>
            </div>
          </div>
        </div>

        <!-- DEFAULTS -->
        <div class="settings-section">
          <h2 class="section-title">Default Preferences</h2>

          <div class="settings-toggle-row">
            <div class="settings-toggle-info">
              <span class="settings-toggle-title">Auto-Save History</span>
              <span class="settings-toggle-desc">Automatically save all generation results</span>
            </div>
            <sl-switch id="auto-save-toggle" ${state.settings.autoSaveHistory !== false ? 'checked' : ''}></sl-switch>
          </div>

          <div class="settings-toggle-row">
            <div class="settings-toggle-info">
              <span class="settings-toggle-title">Auto Enhance Prompt</span>
              <span class="settings-toggle-desc">Enhance prompts via the Gemma text encoder by default</span>
            </div>
            <sl-switch id="enhance-prompt-toggle" ${state.settings.enhancePrompt ? 'checked' : ''}></sl-switch>
          </div>

          <div class="settings-toggle-row">
            <div class="settings-toggle-info">
              <span class="settings-toggle-title">Torch Compile</span>
              <span class="settings-toggle-desc">Enable torch.compile for transformer blocks (slower first run, faster subsequent)</span>
            </div>
            <sl-switch id="torch-compile-toggle" ${state.settings.torchCompile ? 'checked' : ''}></sl-switch>
          </div>

          <div style="margin-top: var(--space-3);">
            <label class="label">Default Pipeline</label>
            <sl-select id="default-pipeline" value="${state.settings.defaultPipeline || 'ti2vid_two_stages'}" size="small" style="margin-top: var(--space-1);">
              ${state.pipelines.map(p => `<sl-option value="${p.id}">${p.name}</sl-option>`).join('')}
            </sl-select>
          </div>

          <div style="margin-top: var(--space-3);">
            <label class="label">Default Quantization</label>
            <sl-select id="default-quant" value="${state.settings.quantization || 'none'}" size="small" style="margin-top: var(--space-1);">
              <sl-option value="none">None (BF16)</sl-option>
              <sl-option value="fp8-cast">FP8 Cast</sl-option>
              <sl-option value="fp8-scaled-mm">FP8 Scaled MM</sl-option>
            </sl-select>
          </div>

          <div style="margin-top: var(--space-3);">
            <label class="label">Streaming Prefetch Count</label>
            <sl-input id="streaming-prefetch" type="number" size="small" min="1" max="8"
              value="${state.settings.streamingPrefetchCount || ''}"
              placeholder="Disabled" style="margin-top: var(--space-1);">
            </sl-input>
            <span class="sublabel" style="margin-top: 2px;">Layer streaming: prefetch N layers ahead (saves VRAM)</span>
          </div>

          <div style="margin-top: var(--space-3);">
            <label class="label">Max Batch Size</label>
            <sl-select id="max-batch-size" value="${state.settings.maxBatchSize || '1'}" size="small" style="margin-top: var(--space-1);">
              <sl-option value="1">1 (Sequential)</sl-option>
              <sl-option value="2">2</sl-option>
              <sl-option value="4">4 (Batch all passes)</sl-option>
            </sl-select>
            <span class="sublabel" style="margin-top: 2px;">Higher values reduce PCIe transfers with layer streaming</span>
          </div>
        </div>

        <!-- RESOLUTION PRESETS -->
        <div class="settings-section">
          <h2 class="section-title">Resolution Presets</h2>
          <div id="resolution-presets" style="display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--space-2);">
            ${renderPresetButtons()}
          </div>
        </div>

        <!-- SYSTEM INFO -->
        <div class="settings-section">
          <h2 class="section-title">System Info</h2>
          <div style="display: flex; flex-direction: column; gap: var(--space-1); font-size: var(--text-xs); color: var(--color-text-muted);">
            <div class="flex justify-between">
              <span>GPU:</span>
              <span id="sys-gpu" class="${state.gpuAvailable ? 'text-primary' : ''}">${state.gpuAvailable ? 'Available' : 'Not Available'}</span>
            </div>
            <div class="flex justify-between">
              <span>Backend:</span>
              <span id="sys-backend">Checking...</span>
            </div>
            <div class="flex justify-between">
              <span>LTX-2 Version:</span>
              <span>v2.3.0</span>
            </div>
          </div>
        </div>

        <!-- SAVE -->
        <div style="padding-bottom: var(--space-8);">
          <button class="btn-primary" id="save-settings-btn" style="width: 100%;">
            <sl-icon name="check-lg"></sl-icon> Save Settings
          </button>
        </div>

      </div>
    </div>
  `;

  bindSettingsEvents(container);
  checkBackendStatus(container);
}


// ============================================================================
//  MODEL STATUS RENDERER
// ============================================================================
const REQUIRED_MODELS = [
  { key: 'checkpoint', name: 'LTX-2.3 Checkpoint', file: 'ltx-2.3-22b-dev.safetensors' },
  { key: 'distilled_lora', name: 'Distilled LoRA', file: 'ltx-2.3-22b-distilled-lora-384.safetensors' },
  { key: 'spatial_upsampler', name: 'Spatial Upsampler', file: 'ltx-2.3-spatial-upscaler-x2-1.1.safetensors' },
  { key: 'gemma', name: 'Gemma 3 Encoder', file: 'google/gemma-3-4b-it' },
];

function renderModelStatus(status) {
  return REQUIRED_MODELS.map(model => {
    const modelStatus = status?.models?.[model.key];
    const isReady = modelStatus === 'ready' || modelStatus === true;
    const isDownloading = modelStatus === 'downloading';
    const badge = isReady ? 'ready' : isDownloading ? 'downloading' : 'missing';
    const label = isReady ? 'READY' : isDownloading ? 'DL...' : 'MISSING';

    return `
      <div class="model-status-item">
        <div>
          <div class="model-status-name">${model.name}</div>
          <div class="sublabel">${model.file}</div>
        </div>
        <span class="model-status-badge ${badge}">${label}</span>
      </div>
    `;
  }).join('');
}


// ============================================================================
//  RESOLUTION PRESETS
// ============================================================================
function renderPresetButtons() {
  const presets = [
    { label: '512×768', w: 768, h: 512 },
    { label: '768×512', w: 512, h: 768 },
    { label: '1024×1536', w: 1536, h: 1024 },
    { label: '1536×1024', w: 1024, h: 1536 },
    { label: '1088×1920', w: 1920, h: 1088 },
    { label: '1920×1088', w: 1088, h: 1920 },
  ];

  return presets.map(p => `
    <button class="btn-ghost" style="font-size: 10px; padding: var(--space-2);" data-preset-w="${p.w}" data-preset-h="${p.h}">
      ${p.label}
    </button>
  `).join('');
}


// ============================================================================
//  EVENT BINDINGS
// ============================================================================
function bindSettingsEvents(container) {
  // Save settings
  container.querySelector('#save-settings-btn')?.addEventListener('click', async () => {
    const settings = {
      hf_token: container.querySelector('#hf-token')?.value || '',
      civitai_token: container.querySelector('#civitai-token')?.value || '',
      models_dir: container.querySelector('#models-dir')?.value || '~/ltx-models',
      output_dir: container.querySelector('#output-dir')?.value || '~/ltx-outputs',
      autoSaveHistory: container.querySelector('#auto-save-toggle')?.checked ?? true,
      enhancePrompt: container.querySelector('#enhance-prompt-toggle')?.checked ?? false,
      torchCompile: container.querySelector('#torch-compile-toggle')?.checked ?? false,
      defaultPipeline: container.querySelector('#default-pipeline')?.value || 'ti2vid_two_stages',
      quantization: container.querySelector('#default-quant')?.value || 'none',
      streamingPrefetchCount: parseInt(container.querySelector('#streaming-prefetch')?.value) || null,
      maxBatchSize: parseInt(container.querySelector('#max-batch-size')?.value) || 1,
    };

    Object.assign(state.settings, settings);

    try {
      await api.put('/api/settings', settings);
      showToast('SETTINGS_SAVED');
    } catch {
      // Save locally if backend not available
      localStorage.setItem('ltx2-settings', JSON.stringify(settings));
      showToast('SETTINGS_SAVED (local only)');
    }
  });

  // Download models
  container.querySelector('#download-models-btn')?.addEventListener('click', async () => {
    try {
      const progressEl = container.querySelector('#download-progress');
      const progressBar = container.querySelector('#dl-progress-bar');
      const progressLabel = container.querySelector('#dl-progress-label');
      if (progressEl) progressEl.classList.remove('hidden');

      const result = await api.post('/api/models/download', {
        hf_token: container.querySelector('#hf-token')?.value,
        civitai_token: container.querySelector('#civitai-token')?.value,
        models_dir: container.querySelector('#models-dir')?.value,
      });

      if (result.job_id) {
        const es = api.sse(`/api/models/download/${result.job_id}/progress`, (data) => {
          if (progressBar) progressBar.value = data.progress || 0;
          if (progressLabel) progressLabel.textContent = data.message || 'Downloading...';

          if (data.status === 'complete') {
            es.close();
            if (progressEl) progressEl.classList.add('hidden');
            showToast('MODELS_DOWNLOADED');
            refreshModelStatus(container);
          }
        });
      }
    } catch (err) {
      showToast(`DOWNLOAD_FAILED: ${err.message}`, 'error');
    }
  });

  // Refresh model status
  container.querySelector('#refresh-models-btn')?.addEventListener('click', () => {
    refreshModelStatus(container);
  });
}

async function refreshModelStatus(container) {
  try {
    const status = await api.get('/api/models/status');
    state.modelStatus = status;
    const list = container.querySelector('#model-status-list');
    if (list) list.innerHTML = renderModelStatus(status);
  } catch {
    showToast('Cannot reach backend', 'error');
  }
}

async function checkBackendStatus(container) {
  const el = container.querySelector('#sys-backend');
  if (!el) return;
  try {
    await api.get('/api/health');
    el.textContent = 'Connected';
    el.style.color = 'var(--color-success)';
  } catch {
    el.textContent = 'Not Running';
    el.style.color = 'var(--color-accent)';
  }
}
