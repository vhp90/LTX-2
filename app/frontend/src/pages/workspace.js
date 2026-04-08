/**
 * LTX-2 Studio — Workspace Page
 * The main video generation workspace with dynamic pipeline parameters.
 */
import { state, api, showToast } from '../main.js';
import Panzoom from '@panzoom/panzoom';

// ============================================================================
//  PARAMETER CONTROL RENDERER
// ============================================================================
function renderParamControl(param, currentValue) {
  const id = `param-${param.name}`;
  const val = currentValue !== undefined ? currentValue : param.default;

  // Render textareas specifically for prompts
  if (param.type === 'textarea') {
    return `
      <div style="display: flex; flex-direction: column; gap: var(--space-1); ${param.name === 'prompt' ? 'flex: 1; min-height: 120px;' : 'min-height: 90px;'} mb-4">
        <label class="label ${param.name === 'negative_prompt' ? 'text-accent' : 'text-primary'}">${param.label}</label>
        <sl-textarea id="input-${param.name}" data-param="${param.name}" data-type="textarea"
          value="${val || ''}"
          placeholder="${param.name === 'prompt' ? 'Describe your video...' : 'What to avoid...'}"
          rows="${param.name === 'prompt' ? 6 : 3}"
          resize="auto"
          style="flex: 1;">
        </sl-textarea>
      </div>
    `;
  }

  // Render file uploads
  if (param.type === 'file' || param.type === 'image_list') {
    return `
      <div style="display: flex; flex-direction: column; gap: var(--space-1); mb-4">
        <label class="label">${param.label}</label>
        ${param.type === 'image_list' ? `
          <div class="upload-zone" id="upload-${param.name}" data-param="${param.name}" data-accept="image/*">
            <sl-icon name="image"></sl-icon>
            <span class="upload-label">Drop images or click to browse</span>
          </div>
          <div id="preview-${param.name}" style="display: flex; gap: var(--space-2); flex-wrap: wrap;"></div>
        ` : `
          <div class="upload-zone" id="upload-${param.name}" data-param="${param.name}" data-accept="${param.accept || '*'}">
            <sl-icon name="${param.accept?.includes('audio') ? 'music-note-beamed' : 'film'}"></sl-icon>
            <span class="upload-label">Drop file or click to browse</span>
          </div>
          <div id="preview-${param.name}"></div>
        `}
        ${param.description ? `<span class="sublabel">${param.description}</span>` : ''}
      </div>
    `;
  }

  // We map num_frames to a UI-only 'duration_seconds' input, and hide num_frames
  if (param.name === 'num_frames') {
    return `
      <div class="param-slider-row" data-virtual-param="duration_seconds">
        <div class="param-slider-header">
          <label class="label">Duration (s)</label>
          <sl-input id="duration-input" type="number" size="small"
            value="5" min="1" max="20" step="0.5"
            style="width: 72px; text-align: right;">
          </sl-input>
        </div>
        <sl-range id="duration-range" min="1" max="20" step="0.5" value="5" style="--track-active-offset: 0;"></sl-range>
        <div class="param-slider-range">
          <span>1s</span>
          <span>20s</span>
        </div>
        <span class="sublabel" style="text-align: right; color: var(--color-primary); font-size: 10px;">LTX Frames: <span id="computed-frames-display">121</span></span>
      </div>
      <!-- Hidden input for actual param payload -->
      <input type="hidden" id="hidden-num-frames" data-param="num_frames" data-type="int" value="121" />
    `;
  }

  switch (param.type) {
    case 'float':
    case 'int': {
      let step = param.step || (param.type === 'float' ? 0.1 : 1);
      let max = param.max ?? 100;
      // Guarantee width/height steps
      if (param.name === 'width' || param.name === 'height') {
        step = 32;
        max = 1920;
      }
      const min = param.min ?? 0;
      return `
        <div class="param-slider-row" data-param="${param.name}">
          <div class="param-slider-header">
            <div class="flex items-center gap-2">
              <label class="label" for="${id}">${param.label}</label>
            </div>
            <sl-input id="${id}" type="number" size="small"
              value="${val}" min="${min}" max="${max}" step="${step}"
              style="width: 72px; text-align: right;"
              data-param="${param.name}" data-type="${param.type}">
            </sl-input>
          </div>
          <sl-range min="${min}" max="${max}" step="${step}" value="${val}"
            data-param="${param.name}" data-type="${param.type}"
            style="--track-active-offset: 0;">
          </sl-range>
          <div class="param-slider-range">
            <span>${min}</span>
            <span>${max}</span>
          </div>
        </div>
      `;
    }
    case 'bool':
      return `
        <div class="param-slider-row" data-param="${param.name}" style="flex-direction: row; justify-content: space-between; align-items: center;">
          <label class="label">${param.label}</label>
          <sl-switch ${val ? 'checked' : ''} data-param="${param.name}" data-type="bool"></sl-switch>
        </div>
      `;
    case 'select':
      return `
        <div class="param-slider-row" data-param="${param.name}">
          <label class="label">${param.label}</label>
          <sl-select value="${val}" data-param="${param.name}" data-type="select" size="small">
            ${(param.options || []).map(o => `<sl-option value="${o.value}">${o.label}</sl-option>`).join('')}
          </sl-select>
        </div>
      `;
    case 'text':
      return `
        <div class="param-slider-row" data-param="${param.name}">
          <label class="label">${param.label}</label>
          <sl-input value="${val || ''}" data-param="${param.name}" data-type="text" size="small"
            placeholder="${param.description || ''}">
          </sl-input>
        </div>
      `;
    default:
      return '';
  }
}

function groupParams(params) {
  const groups = {};
  for (const p of params) {
    if (p.type === 'textarea' || p.type === 'image_list' || p.type === 'file') {
      if (!groups['prompts_and_files']) groups['prompts_and_files'] = [];
      groups['prompts_and_files'].push(p);
      continue;
    }
    const g = p.group || 'other';
    if (!groups[g]) groups[g] = [];
    groups[g].push(p);
  }
  return groups;
}

const GROUP_LABELS = {
  generation: 'Generation',
  video_guidance: 'Video Guidance',
  audio_guidance: 'Audio Guidance',
  conditioning: 'Conditioning',
  advanced: 'Advanced',
  input: 'Input',
  other: 'Other',
};


// ============================================================================
//  CURRENT PARAMETER VALUES & STATE
// ============================================================================
let paramValues = {};
let uploadedFiles = {};  // { param_name: File | File[] }
let availableLoras = [];
let selectedLoras = [];

function initParamValues(params) {
  const oldValues = { ...paramValues };
  paramValues = {};
  for (const p of params) {
    if (oldValues[p.name] !== undefined) {
      paramValues[p.name] = oldValues[p.name];
    } else if (p.default !== undefined && p.default !== null) {
      paramValues[p.name] = p.default;
    }
  }
}

function collectParamValues(container) {
  // Collect from range/input/switch/select controls
  container.querySelectorAll('[data-param]').forEach(el => {
    // skip hidden structural items without proper values
    if (el.tagName === 'DIV') return;
    
    const name = el.dataset.param;
    const type = el.dataset.type;
    if (!name || !type) return;

    if (el.tagName === 'SL-RANGE' || el.tagName === 'INPUT') { // Hidden inputs are 'INPUT'
      paramValues[name] = type === 'int' ? parseInt(el.value) : parseFloat(el.value);
    } else if (el.tagName === 'SL-INPUT') {
      if (type === 'int') paramValues[name] = parseInt(el.value);
      else if (type === 'float') paramValues[name] = parseFloat(el.value);
      else paramValues[name] = el.value;
    } else if (el.tagName === 'SL-SWITCH') {
      paramValues[name] = el.checked;
    } else if (el.tagName === 'SL-SELECT') {
      paramValues[name] = el.value;
    } else if (el.tagName === 'SL-TEXTAREA') {
      paramValues[name] = el.value;
    }
  });

  // Collect Duration computation manually to ensure it didn't get overridden incorrectly
  const durationRg = container.querySelector('#duration-range');
  const fpsInput = container.querySelector('sl-input[data-param="frame_rate"]');
  if (durationRg && fpsInput) {
    const duration = parseFloat(durationRg.value);
    const fps = parseFloat(fpsInput.value);
    const computedFrames = Math.max(9, Math.round((duration * fps - 1) / 8) * 8 + 1);
    paramValues['num_frames'] = computedFrames;
  }
}


// ============================================================================
//  RENDER WORKSPACE
// ============================================================================
export function renderWorkspace(container) {
  const pipeline = state.pipelines.find(p => p.id === state.selectedPipeline) || state.pipelines[0];
  if (!pipeline) {
    container.innerHTML = `<div style="padding: 40px; text-align: center;">
      <p class="text-muted">No pipelines available. Start the backend server.</p>
    </div>`;
    return;
  }

  // Init param values for this pipeline (persists overlapping values to avoid wiping everything)
  initParamValues(pipeline.params);

  const groups = groupParams(pipeline.params);
  const promptFiles = groups['prompts_and_files'] || [];
  
  // Create 3-column layout explicitly
  container.innerHTML = `
    <div style="display: flex; flex: 1; height: 100%; overflow: hidden;">

      <!-- LEFT PANEL: Parameters, Pipeline, Execute -->
      <aside id="left-sidebar" class="panel" style="width: 400px; border-right: 1px solid var(--color-border); flex-shrink: 0; overflow-y: auto;">
        
        <!-- Action Header -->
        <div class="panel-header" style="flex-direction: column; gap: var(--space-2); align-items: stretch;">
          <div class="param-group">
            <sl-select id="pipeline-select" value="${pipeline.id}" size="small" style="width: 100%;">
              ${state.pipelines.map(p => `
                <sl-option value="${p.id}">${p.name}</sl-option>
              `).join('')}
            </sl-select>
            <div class="text-xs text-text-muted mt-1 uppercase tracking-widest">${pipeline.description}</div>
          </div>
          <button class="btn-primary" id="execute-btn" style="width: 100%; height: 44px; font-size: var(--text-sm);">
            <sl-icon name="lightning-charge"></sl-icon>
            EXECUTE RENDER
          </button>
        </div>

        <div class="panel-body" style="display: flex; flex-direction: column; gap: var(--space-4);">
          <!-- Prompts and Files Top -->
          <div class="prompt-stack">
            ${promptFiles.map(p => renderParamControl(p, paramValues[p.name])).join('')}
          </div>

          <!-- Dimension Controls Injection -->
          ${pipeline.params.find(p => p.name === 'width') ? `
            <div class="param-group" style="padding-top: var(--space-2); border-top: 1px solid var(--color-border);">
              <sl-switch id="auto-dims-switch" checked>Auto-Dimensions from Image</sl-switch>
              <span class="sublabel mt-1 block">When enabled, dimensions are inferred from the first conditioning image and clamped to LTX-2 valid multiples of 32 (max 1920x1088).</span>
            </div>
          ` : ''}

          <!-- Dynamic parameter groups -->
          ${Object.entries(groups).filter(([k]) => k !== 'prompts_and_files').map(([groupKey, params]) => `
            <sl-details ${groupKey === 'generation' ? 'open' : ''} summary="${GROUP_LABELS[groupKey] || groupKey}">
              <div id="group-${groupKey}" style="display: flex; flex-direction: column; gap: var(--space-4);">
                ${params.map(p => renderParamControl(p, paramValues[p.name])).join('')}
              </div>
            </sl-details>
          `).join('')}
        </div>
      </aside>

      <!-- CENTER PANEL: Viewport completely filling the middle -->
      <section class="viewport-container" id="canvas-wrapper" style="flex: 1; min-width: 0; position: relative; background: #080808; overflow: hidden;">
        <!-- Toggles -->
        <button class="btn-icon panel-toggle-btn" id="toggle-left-sidebar" style="position: absolute; top: var(--space-2); left: var(--space-2); z-index: 10; background: var(--color-surface); border: 1px solid var(--color-border); padding: 4px;" title="Toggle Settings Panel">
          <sl-icon name="layout-sidebar"></sl-icon>
        </button>
        <button class="btn-icon panel-toggle-btn" id="toggle-right-sidebar" style="position: absolute; top: var(--space-2); right: var(--space-2); z-index: 10; background: var(--color-surface); border: 1px solid var(--color-border); padding: 4px;" title="Toggle LoRA Panel">
          <sl-icon name="layout-sidebar-reverse"></sl-icon>
        </button>

        <div id="canvas-content" style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; transform-origin: 0 0;">
          <div id="video-container" style="display: none; position: relative; box-shadow: 0 10px 40px rgba(0,0,0,0.8); border: 1px solid var(--color-border);">
            <!-- Native video controls directly attached to the box exactly like comfyui -->
            <video id="output-video" controls loop style="display: block; max-width: 100vw; max-height: 100vh; outline: none; background: #000;"></video>
          </div>
          <!-- Centered text for when empty -->
          <div id="viewport-empty-state" style="color: var(--color-text-muted); font-size: 0.8rem; letter-spacing: 2px; text-transform: uppercase;">
             Infinite Canvas [Drag to Pan, Ctrl+Scroll to Zoom]
          </div>
        </div>

        <!-- Progress overlay (floating at bottom) -->
        <div id="generation-progress" class="hidden" style="position: absolute; bottom: 30px; left: 50%; transform: translateX(-50%); width: 400px; background: var(--color-surface); border: 1px solid var(--color-border); padding: var(--space-3); z-index: 20; box-shadow: 0 5px 20px rgba(0,0,0,0.5);">
          <sl-progress-bar id="progress-bar" value="0" style="width: 100%;"></sl-progress-bar>
          <div class="flex justify-between" style="padding-top: var(--space-2); font-size: var(--text-xs); color: var(--color-text-muted);">
            <span class="sublabel" id="progress-label">Initializing...</span>
            <span class="sublabel" id="progress-pct" style="color: var(--color-primary);">0%</span>
          </div>
        </div>
      </section>

      <!-- RIGHT PANEL: LoRA Settings ONLY -->
      <aside id="right-sidebar" class="panel" style="width: 300px; border-left: 1px solid var(--color-border); flex-shrink: 0; overflow-y: auto;">
        <div class="panel-header">
          <span class="sublabel font-bold uppercase tracking-widest text-primary">[CUSTOM LORAS]</span>
        </div>
        <div class="panel-body" style="display: flex; flex-direction: column; gap: var(--space-4);">
          <div id="lora-list" style="display: flex; flex-direction: column; gap: var(--space-2);">
            <div class="sublabel" id="lora-empty-state">No custom LoRAs loaded</div>
          </div>
          <button class="btn-ghost" id="add-lora-btn" style="width: 100%;">
            <sl-icon name="plus-lg"></sl-icon> Add LoRA
          </button>
        </div>
      </aside>
    </div>
  `;

  // ======== EVENT BINDINGS ========
  bindPipelineSelector(container);
  bindDurationFramesSync(container);
  bindAutoDimensions(container);
  bindParamSyncEvents(container);
  bindFileUploads(container);
  bindExecuteButton(container);
  bindCanvas(container);
  bindLoraControls(container);

  // Bind Sidebar Toggles
  const toggleLeft = container.querySelector('#toggle-left-sidebar');
  const leftPanel = container.querySelector('#left-sidebar');
  if (toggleLeft && leftPanel) {
    toggleLeft.onclick = () => {
      const isCollapsed = leftPanel.style.display === 'none';
      leftPanel.style.display = isCollapsed ? 'block' : 'none';
      toggleLeft.innerHTML = isCollapsed ? '<sl-icon name="layout-sidebar"></sl-icon>' : '<sl-icon name="layout-sidebar-inset"></sl-icon>';
    };
  }

  const toggleRight = container.querySelector('#toggle-right-sidebar');
  const rightPanel = container.querySelector('#right-sidebar');
  if (toggleRight && rightPanel) {
    toggleRight.onclick = () => {
      const isCollapsed = rightPanel.style.display === 'none';
      rightPanel.style.display = isCollapsed ? 'block' : 'none';
      toggleRight.innerHTML = isCollapsed ? '<sl-icon name="layout-sidebar-reverse"></sl-icon>' : '<sl-icon name="layout-sidebar-inset-reverse"></sl-icon>';
    };
  }
}


// ============================================================================
//  EVENT BINDINGS
// ============================================================================

async function bindLoraControls(container) {
  try {
    const res = await api.get('/api/models/loras');
    availableLoras = res.loras || [];
  } catch (e) {
    console.warn('Could not load LoRAs');
  }

  const listEl = container.querySelector('#lora-list');
  const addBtn = container.querySelector('#add-lora-btn');
  const emptyState = container.querySelector('#lora-empty-state');

  function renderLoraList() {
    if (!listEl) return;
    
    // Remove all old dynamically injected lora divs to rebuild
    listEl.querySelectorAll('.lora-item').forEach(el => el.remove());

    if (selectedLoras.length === 0) {
      if (emptyState) emptyState.style.display = 'block';
      return;
    } else {
      if (emptyState) emptyState.style.display = 'none';
    }
    
    selectedLoras.forEach((lora, idx) => {
      const div = document.createElement('div');
      div.className = 'lora-item';
      div.style.background = 'var(--color-bg)';
      div.style.border = '1px solid var(--color-border)';
      div.style.padding = 'var(--space-2)';
      div.style.display = 'flex';
      div.style.flexDirection = 'column';
      div.style.gap = 'var(--space-2)';
      
      div.innerHTML = `
        <div class="flex items-center justify-between">
          <sl-select size="small" value="${lora.path}" style="flex: 1;" class="lora-select">
            <sl-option value="">-- Select LoRA --</sl-option>
            ${availableLoras.map(a => `<sl-option value="${a.path}">${a.name}</sl-option>`).join('')}
          </sl-select>
          <button class="btn-icon lora-remove" style="width: 24px; height: 24px;">
            <sl-icon name="x" class="text-accent"></sl-icon>
          </button>
        </div>
        <div class="flex items-center gap-2">
          <span class="sublabel">Strength</span>
          <sl-range min="-2" max="2" step="0.05" value="${lora.strength}" style="flex: 1; --track-active-offset: 50%;" class="lora-strength"></sl-range>
          <span class="text-xs strength-display" style="width: 30px; text-align: right;">${lora.strength}</span>
        </div>
      `;

      div.querySelector('.lora-select').addEventListener('sl-change', (e) => {
        selectedLoras[idx].path = e.target.value;
      });
      const rng = div.querySelector('.lora-strength');
      const disp = div.querySelector('.strength-display');
      rng.addEventListener('sl-input', (e) => {
        selectedLoras[idx].strength = parseFloat(rng.value);
        disp.textContent = rng.value;
      });
      div.querySelector('.lora-remove').onclick = () => {
        selectedLoras.splice(idx, 1);
        renderLoraList();
      };
      
      listEl.appendChild(div);
    });
  }

  if (addBtn) {
    // using onclick prevents duplicate event bindings if container is re-processed
    addBtn.onclick = () => {
      selectedLoras.push({ path: '', strength: 0.8 });
      renderLoraList();
    };
  }

  renderLoraList();
}

function bindPipelineSelector(container) {
  const select = container.querySelector('#pipeline-select');
  if (!select) return;
  // Ignore the initial mount bubbled event by explicitly ensuring it's user intent
  select.addEventListener('sl-change', (e) => {
    // If value hasn't changed from state, ignore
    if (state.selectedPipeline === select.value) return; 
    
    // Save any current parameter text values before re-rendering so they persist
    collectParamValues(container);

    state.selectedPipeline = select.value;
    
    const pageContent = document.getElementById('page-content');
    renderWorkspace(pageContent);
  });
}

function bindDurationFramesSync(container) {
  const durRange = container.querySelector('#duration-range');
  const durInput = container.querySelector('#duration-input');
  const computedDisp = container.querySelector('#computed-frames-display');
  const hiddenFrames = container.querySelector('#hidden-num-frames');
  const fpsInput = container.querySelector('sl-input[data-param="frame_rate"]');
  const vpTotalFrames = container.querySelector('#vp-total-frames');

  function updateFrames() {
    if (!durRange || !fpsInput) return;
    const duration = parseFloat(durRange.value);
    const fps = parseFloat(fpsInput.value);
    // LTX rule: (F-1) % 8 == 0
    const computedFrames = Math.max(9, Math.round((duration * fps - 1) / 8) * 8 + 1);
    
    if (computedDisp) computedDisp.textContent = computedFrames;
    if (hiddenFrames) hiddenFrames.value = computedFrames;
    if (vpTotalFrames) vpTotalFrames.textContent = computedFrames;
    paramValues['num_frames'] = computedFrames;
  }

  if (durRange && durInput) {
    durRange.addEventListener('sl-input', () => { durInput.value = durRange.value; updateFrames(); });
    durInput.addEventListener('sl-change', () => { durRange.value = durInput.value; updateFrames(); });
  }

  if (fpsInput) {
    fpsInput.addEventListener('sl-change', updateFrames);
    const fpsRange = container.querySelector('sl-range[data-param="frame_rate"]');
    if (fpsRange) fpsRange.addEventListener('sl-input', updateFrames);
  }

  // initial sync run
  updateFrames();
}

function bindAutoDimensions(container) {
  const autoSwitch = container.querySelector('#auto-dims-switch');
  const widthInput = container.querySelector('sl-input[data-param="width"]');
  const widthRange = container.querySelector('sl-range[data-param="width"]');
  const heightInput = container.querySelector('sl-input[data-param="height"]');
  const heightRange = container.querySelector('sl-range[data-param="height"]');

  if (!autoSwitch || !widthInput || !heightInput) return;

  function toggleDimControls() {
    const isAuto = autoSwitch.checked;
    widthInput.disabled = isAuto;
    if (widthRange) widthRange.disabled = isAuto;
    heightInput.disabled = isAuto;
    if (heightRange) heightRange.disabled = isAuto;
  }

  autoSwitch.addEventListener('sl-change', toggleDimControls);
  toggleDimControls();

  // Export a function we can call when an image is uploaded to force dimensions
  container._triggerAutoDimensions = (imgObject) => {
    if (!autoSwitch.checked) return;
    const cw = imgObject.width;
    const ch = imgObject.height;
    
    // Max dimension target
    const MAX_PIXELS = 1920 * 1088;
    let scale = 1.0;
    if (cw * ch > MAX_PIXELS) {
      scale = Math.sqrt(MAX_PIXELS / (cw * ch));
    }
    
    let nw = cw * scale;
    let nh = ch * scale;

    // Snap to 32
    nw = Math.round(nw / 32) * 32;
    nh = Math.round(nh / 32) * 32;
    
    // Clamp to min 64
    nw = Math.max(64, nw);
    nh = Math.max(64, nh);

    widthInput.value = nw;
    if (widthRange) widthRange.value = nw;
    heightInput.value = nh;
    if (heightRange) heightRange.value = nh;
    
    paramValues['width'] = nw;
    paramValues['height'] = nh;
    showToast(`Auto-adjusted dimensions to ${nw}x${nh} (Max limits & 32px multiple applied)`);
  };
}

function bindParamSyncEvents(container) {
  // Sync range ↔ input
  container.querySelectorAll('sl-range[data-param]').forEach(range => {
    const name = range.dataset.param;
    const input = container.querySelector(`sl-input[data-param="${name}"]`);

    range.addEventListener('sl-input', () => {
      if (input) input.value = range.value;
      paramValues[name] = range.dataset.type === 'int' ? parseInt(range.value) : parseFloat(range.value);
      updateViewportStats(container);
    });
  });

  container.querySelectorAll('sl-input[data-param]').forEach(input => {
    const name = input.dataset.param;
    const range = container.querySelector(`sl-range[data-param="${name}"]`);

    input.addEventListener('sl-change', () => {
      if (range) range.value = input.value;
      const type = input.dataset.type;
      if (type === 'int') paramValues[name] = parseInt(input.value);
      else if (type === 'float') paramValues[name] = parseFloat(input.value);
      else paramValues[name] = input.value;
      updateViewportStats(container);
    });
  });

  // Sync switches
  container.querySelectorAll('sl-switch[data-param]').forEach(sw => {
    sw.addEventListener('sl-change', () => {
      paramValues[sw.dataset.param] = sw.checked;
    });
  });

  // Sync textareas
  container.querySelectorAll('sl-textarea[data-param]').forEach(ta => {
    ta.addEventListener('sl-input', () => {
      paramValues[ta.dataset.param] = ta.value;
    });
  });
}

function updateViewportStats(container) {
  const fps = container.querySelector('#vp-fps');
  if (fps && paramValues.frame_rate) fps.textContent = paramValues.frame_rate;
}

function bindFileUploads(container) {
  container.querySelectorAll('.upload-zone').forEach(zone => {
    const paramName = zone.dataset.param;
    const accept = zone.dataset.accept || '*';
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = accept;
    fileInput.style.display = 'none';
    if (accept.includes('image')) fileInput.multiple = true;
    container.appendChild(fileInput);

    zone.addEventListener('click', () => fileInput.click());
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', (e) => {
      e.preventDefault();
      zone.classList.remove('dragover');
      handleFiles(paramName, e.dataTransfer.files, container);
    });

    fileInput.addEventListener('change', () => {
      handleFiles(paramName, fileInput.files, container);
    });
  });
}

function handleFiles(paramName, files, container) {
  if (!files.length) return;
  const preview = container.querySelector(`#preview-${paramName}`);

  // Image reading logic to potentially trigger auto-resize
  if (files.length > 0 && files[0].type.startsWith('image') && container._triggerAutoDimensions) {
    const objUrl = URL.createObjectURL(files[0]);
    const img = new Image();
    img.onload = () => container._triggerAutoDimensions(img);
    img.src = objUrl;
  }

  if (Array.from(files).some(f => f.type.startsWith('image'))) {
    const existing = uploadedFiles[paramName] || [];
    uploadedFiles[paramName] = [...existing, ...Array.from(files)];

    if (preview) {
      preview.innerHTML = uploadedFiles[paramName].map((f, i) => `
        <div class="upload-preview" style="width: 60px; height: 60px; position: relative;">
          <img src="${URL.createObjectURL(f)}" alt="${f.name}">
          <button class="upload-remove" data-param="${paramName}" data-index="${i}">
            <sl-icon name="x" style="font-size: 12px;"></sl-icon>
          </button>
        </div>
      `).join('');

      preview.querySelectorAll('.upload-remove').forEach(btn => {
        btn.onclick = (e) => {
          e.stopPropagation();
          const idx = parseInt(btn.dataset.index);
          uploadedFiles[paramName].splice(idx, 1);
          handleFiles(paramName, [], container); 
          if (uploadedFiles[paramName].length === 0) {
            preview.innerHTML = '';
          }
        };
      });
    }
  } else {
    // Single file like video/audio
    uploadedFiles[paramName] = files[0];
    if (preview) {
      preview.innerHTML = `
        <div class="flex items-center gap-2" style="padding: var(--space-2); background: var(--color-bg); border: 1px solid var(--color-border);">
          <sl-icon name="${files[0].type.startsWith('audio') ? 'music-note-beamed' : 'film'}" style="color: var(--color-primary);"></sl-icon>
          <span class="text-xs truncate" style="flex: 1;">${files[0].name}</span>
          <button class="btn-icon" style="width: 20px; height: 20px;" data-param="${paramName}">
            <sl-icon name="x" style="font-size: 12px; color: var(--color-accent);"></sl-icon>
          </button>
        </div>
      `;
      preview.querySelector('button').onclick = () => {
        delete uploadedFiles[paramName];
        preview.innerHTML = '';
        const zone = container.querySelector(`#upload-${paramName}`);
        if (zone) zone.style.display = '';
      };
    }
  }

  const zone = container.querySelector(`#upload-${paramName}`);
  if (zone && !(files[0]?.type?.startsWith('image'))) {
    zone.style.display = uploadedFiles[paramName] ? 'none' : '';
  }
}

function bindExecuteButton(container) {
  const btn = container.querySelector('#execute-btn');
  if (!btn) return;

  btn.onclick = async () => {
    collectParamValues(container);

    const pipeline = state.pipelines.find(p => p.id === state.selectedPipeline);
    if (!pipeline) return;

    const missing = pipeline.params.filter(p => p.required && !paramValues[p.name] && !uploadedFiles[p.name]);
    if (missing.length > 0) {
      showToast(`MISSING_REQUIRED: ${missing.map(p => p.label).join(', ')}`, 'error');
      return;
    }

    const request = {
      pipeline: state.selectedPipeline,
      params: { 
        ...paramValues, 
        custom_loras: selectedLoras.filter(l => l.path), 
      },
      quantization: container.querySelector('#quantization-select')?.value || 'none',
    };

    try {
      btn.disabled = true;
      btn.innerHTML = '<div class="loading-spinner"></div> GENERATING...';

      const progressEl = container.querySelector('#generation-progress');
      const progressBar = container.querySelector('#progress-bar');
      const progressLabel = container.querySelector('#progress-label');
      const progressPct = container.querySelector('#progress-pct');
      const statusEl = container.querySelector('#viewport-status');

      if (progressEl) progressEl.classList.remove('hidden');
      if (statusEl) {
        statusEl.innerHTML = '<span>[GENERATING]</span><span class="progress-pulse">PROCESSING...</span>';
      }

      const result = await api.post('/api/generate', request);

      if (result.job_id) {
        const es = api.sse(`/api/generate/${result.job_id}/progress`, (data) => {
          if (progressBar) progressBar.value = data.progress || 0;
          if (progressLabel) progressLabel.textContent = data.stage || 'Processing...';
          if (progressPct) progressPct.textContent = `${Math.round(data.progress || 0)}%`;

          if (data.status === 'complete') {
            es.close();
            onGenerationComplete(container, data);
          } else if (data.status === 'error') {
            es.close();
            onGenerationError(container, data.error);
          }
        }, () => {
          pollJobStatus(result.job_id, container);
        });
      }
    } catch (err) {
      showToast(`GENERATION_FAILED: ${err.message}`, 'error');
      onGenerationError(container, err.message);
    }
  };
}

function onGenerationComplete(container, data) {
  const btn = container.querySelector('#execute-btn');
  const progressEl = container.querySelector('#generation-progress');
  const emptyState = container.querySelector('#viewport-empty-state');
  const videoCont = container.querySelector('#video-container');
  const video = container.querySelector('#output-video');

  if (btn) {
    btn.disabled = false;
    btn.innerHTML = '<sl-icon name="lightning-charge"></sl-icon> EXECUTE RENDER';
  }
  if (progressEl) progressEl.classList.add('hidden');

  if (data.output_path && video) {
    video.src = `/outputs/${data.output_path}`;
    if (emptyState) emptyState.style.display = 'none';
    if (videoCont) videoCont.style.display = 'block';
    video.load();
    video.play().catch(e => console.log('Autoplay blocked:', e));
  }
  showToast('RENDER_COMPLETE');
}

function onGenerationError(container, errorMsg) {
  const btn = container.querySelector('#execute-btn');
  const progressEl = container.querySelector('#generation-progress');

  if (btn) {
    btn.disabled = false;
    btn.innerHTML = '<sl-icon name="lightning-charge"></sl-icon> EXECUTE RENDER';
  }
  if (progressEl) progressEl.classList.add('hidden');
}

async function pollJobStatus(jobId, container) {
  const check = async () => {
    try {
      const data = await api.get(`/api/generate/${jobId}/status`);
      const progressBar = container.querySelector('#progress-bar');
      const progressLabel = container.querySelector('#progress-label');
      const progressPct = container.querySelector('#progress-pct');

      if (progressBar) progressBar.value = data.progress || 0;
      if (progressLabel) progressLabel.textContent = data.stage || 'Processing...';
      if (progressPct) progressPct.textContent = `${Math.round(data.progress || 0)}%`;

      if (data.status === 'complete') {
        onGenerationComplete(container, data);
      } else if (data.status === 'error') {
        onGenerationError(container, data.error);
      } else {
        setTimeout(check, 2000);
      }
    } catch {
      setTimeout(check, 5000);
    }
  };
  check();
}

function bindCanvas(container) {
  const elem = container.querySelector('#canvas-content');
  const wrapper = container.querySelector('#canvas-wrapper');
  if (!elem || !wrapper) return;
  
  const panzoom = Panzoom(elem, {
    maxScale: 15,
    minScale: 0.1,
    cursor: 'default',
    canvas: true,
  });

  // Enable double-click to pan via raw dragging or similar
  // By default, Panzoom lets you drag the element.

  // Pan
  elem.parentElement.addEventListener('pointerdown', panzoom.handleDown);
  document.addEventListener('pointermove', panzoom.handleMove);
  document.addEventListener('pointerup', panzoom.handleUp);
  
  // Custom Zoom handling
  wrapper.addEventListener('wheel', (e) => {
    // Only zoom when Ctrl or Cmd is pressed, simulating proper node canvas behavior
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      panzoom.zoomWithWheel(e);
    } else {
      // Allow regular standard scrolling to pan if desired (or optionally prevent)
      // ComfyUI pans loosely on drag
    }
  });
}
