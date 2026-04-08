/**
 * LTX-2 Studio — History Page
 * Grid view of past generations with metadata overlays.
 */
import { state, api, showToast, navigateTo } from '../main.js';

// ============================================================================
//  RENDER HISTORY
// ============================================================================
export function renderHistory(container) {
  container.innerHTML = `
    <div style="display: flex; flex-direction: column; height: 100%;">
      <!-- Header -->
      <div style="display: flex; align-items: center; justify-content: space-between; padding: var(--space-4) var(--space-6); border-bottom: 1px solid var(--color-border); flex-shrink: 0;">
        <div class="flex items-center gap-4">
          <h1 class="font-display font-bold text-lg uppercase tracking-widest">History</h1>
          <span class="sublabel" id="history-count"></span>
        </div>
        <div class="flex items-center gap-3">
          <sl-input id="history-search" placeholder="Search prompts..." size="small" style="width: 220px;" clearable>
            <sl-icon name="search" slot="prefix"></sl-icon>
          </sl-input>
          <sl-select id="history-filter" value="all" size="small" style="width: 140px;">
            <sl-option value="all">All</sl-option>
            <sl-option value="completed">Completed</sl-option>
            <sl-option value="failed">Failed</sl-option>
          </sl-select>
          <sl-select id="history-sort" value="newest" size="small" style="width: 140px;">
            <sl-option value="newest">Newest</sl-option>
            <sl-option value="oldest">Oldest</sl-option>
          </sl-select>
        </div>
      </div>

      <!-- Grid -->
      <div id="history-grid" style="flex: 1; overflow-y: auto; padding: var(--space-6);">
        <div id="history-items" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: var(--space-4);"></div>
        <div id="history-empty" class="hidden" style="display: flex; flex-direction: column; align-items: center; justify-content: center; gap: var(--space-4); padding: var(--space-12); color: var(--color-text-muted);">
          <sl-icon name="clock-history" style="font-size: 48px; opacity: 0.3;"></sl-icon>
          <span class="section-title">No generations yet</span>
          <span class="sublabel">Your generation history will appear here</span>
          <button class="btn-ghost" id="goto-workspace">
            <sl-icon name="terminal"></sl-icon> Go to Workspace
          </button>
        </div>
      </div>
    </div>

    <!-- Detail dialog -->
    <sl-dialog id="history-detail-dialog" label="Generation Details" style="--width: 640px;">
      <div id="history-detail-content"></div>
      <sl-button slot="footer" variant="default" id="detail-close">Close</sl-button>
      <sl-button slot="footer" variant="primary" id="detail-fork">
        <sl-icon name="shuffle" slot="prefix"></sl-icon> Fork to Workspace
      </sl-button>
    </sl-dialog>
  `;

  loadHistory(container);
  bindHistoryEvents(container);
}


// ============================================================================
//  LOAD HISTORY DATA
// ============================================================================
let historyItems = [];

async function loadHistory(container) {
  try {
    const data = await api.get('/api/history');
    historyItems = data.items || [];
  } catch {
    // Backend not running — show demo data
    historyItems = getDemoHistory();
  }
  renderHistoryGrid(container, historyItems);
}

function renderHistoryGrid(container, items) {
  const grid = container.querySelector('#history-items');
  const empty = container.querySelector('#history-empty');
  const count = container.querySelector('#history-count');

  if (!grid) return;

  if (items.length === 0) {
    grid.innerHTML = '';
    if (empty) empty.classList.remove('hidden');
    if (count) count.textContent = '';
    return;
  }

  if (empty) empty.classList.add('hidden');
  if (count) count.textContent = `${items.length} generations`;

  grid.innerHTML = items.map((item, idx) => `
    <div class="card ${item.status === 'error' ? 'error' : ''} slide-in-right" data-index="${idx}" style="animation-delay: ${idx * 30}ms;">
      <div class="card-thumbnail">
        ${item.thumbnail
          ? `<img src="${item.thumbnail}" alt="Generation thumbnail" loading="lazy">`
          : `<div style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;">
               <sl-icon name="${item.status === 'error' ? 'exclamation-triangle' : 'camera-reels'}"
                 style="font-size: 32px; color: ${item.status === 'error' ? 'var(--color-accent)' : 'var(--color-text-muted)'}; opacity: 0.3;"></sl-icon>
             </div>`
        }
        <div class="card-overlay">
          <div style="display: flex; flex-direction: column; gap: var(--space-2); font-size: var(--text-xs);">
            <div class="flex justify-between">
              <span class="text-muted">PIPELINE:</span>
              <span class="text-primary">${item.pipeline || 'unknown'}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-muted">SEED:</span>
              <span>${item.params?.seed || '—'}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-muted">STEPS:</span>
              <span>${item.params?.num_inference_steps || '—'}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-muted">CFG:</span>
              <span>${item.params?.video_cfg_scale || '—'}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-muted">SIZE:</span>
              <span>${item.params?.width || '—'}×${item.params?.height || '—'}</span>
            </div>
          </div>
        </div>
      </div>
      <div class="card-meta">
        <span class="card-prompt">${escapeHtml(item.prompt || 'No prompt')}</span>
        <div class="flex items-center gap-2">
          <span class="card-badge ${item.status === 'error' ? 'error' : ''}">${item.status === 'error' ? 'FAILED' : formatDuration(item.duration)}</span>
          <button class="btn-icon card-delete" data-id="${item.id}" title="Delete" style="width: 20px; height: 20px; opacity: 0.4;">
            <sl-icon name="trash" style="font-size: 12px; color: var(--color-accent);"></sl-icon>
          </button>
        </div>
      </div>
    </div>
  `).join('');

  // Bind card clicks
  grid.querySelectorAll('.card').forEach(card => {
    card.addEventListener('click', (e) => {
      // Don't open detail if delete button was clicked
      if (e.target.closest('.card-delete')) return;
      const idx = parseInt(card.dataset.index);
      showDetail(container, items[idx]);
    });
  });

  // Bind delete buttons
  grid.querySelectorAll('.card-delete').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const id = btn.dataset.id;
      try {
        await api.del(`/api/history/${id}`);
        historyItems = historyItems.filter(i => i.id !== id);
        renderHistoryGrid(container, historyItems);
        showToast('Generation deleted');
      } catch {
        showToast('Failed to delete', 'error');
      }
    });
  });
}


// ============================================================================
//  DETAIL DIALOG
// ============================================================================
function showDetail(container, item) {
  const dialog = container.querySelector('#history-detail-dialog');
  const content = container.querySelector('#history-detail-content');
  if (!dialog || !content) return;

  content.innerHTML = `
    <div style="display: flex; flex-direction: column; gap: var(--space-4);">
      ${item.output_path ? `
        <video src="/outputs/${item.output_path}" controls style="width: 100%; border: 1px solid var(--color-border);"></video>
      ` : `
        <div style="aspect-ratio: 16/9; background: var(--color-bg); border: 1px solid var(--color-border); display: flex; align-items: center; justify-content: center;">
          <span class="text-muted">No output available</span>
        </div>
      `}

      <div style="display: flex; flex-direction: column; gap: var(--space-2); font-size: var(--text-xs);">
        <div class="param-group-title">Prompt</div>
        <div style="padding: var(--space-2); background: var(--color-bg); border: 1px solid var(--color-border); white-space: pre-wrap; color: var(--color-text-secondary);">
          ${escapeHtml(item.prompt || 'N/A')}
        </div>

        <div class="param-group-title" style="margin-top: var(--space-2);">Parameters</div>
        ${Object.entries(item.params || {}).map(([k, v]) => `
          <div class="flex justify-between" style="padding: 2px 0; border-bottom: 1px solid rgba(39,39,42,0.3);">
            <span class="text-muted">${k}:</span>
            <span>${v}</span>
          </div>
        `).join('')}
      </div>

      <div class="flex justify-between" style="font-size: 10px; color: var(--color-text-muted);">
        <span>${item.created_at || ''}</span>
        <span class="${item.status === 'error' ? 'text-accent' : ''}">${item.status}${item.error ? ': ' + escapeHtml(item.error) : ''}</span>
      </div>
    </div>
  `;

  // Fork button
  const forkBtn = container.querySelector('#detail-fork');
  if (forkBtn) {
    forkBtn.onclick = () => {
      // Copy params to workspace
      if (item.params) {
        Object.assign(state.settings, { lastForkedParams: item.params });
      }
      state.selectedPipeline = item.pipeline || state.selectedPipeline;
      dialog.hide();
      navigateTo('workspace');
      showToast('PARAMS_FORKED: Settings loaded from history');
    };
  }

  const closeBtn = container.querySelector('#detail-close');
  if (closeBtn) closeBtn.onclick = () => dialog.hide();

  dialog.show();
}


// ============================================================================
//  EVENT BINDINGS
// ============================================================================
function bindHistoryEvents(container) {
  // Search
  const search = container.querySelector('#history-search');
  search?.addEventListener('sl-input', () => {
    const q = search.value.toLowerCase();
    const filtered = historyItems.filter(item =>
      (item.prompt || '').toLowerCase().includes(q) ||
      (item.pipeline || '').toLowerCase().includes(q)
    );
    renderHistoryGrid(container, filtered);
  });

  // Filter
  const filter = container.querySelector('#history-filter');
  filter?.addEventListener('sl-change', () => {
    let filtered = [...historyItems];
    if (filter.value === 'completed') filtered = filtered.filter(i => i.status === 'completed');
    if (filter.value === 'failed') filtered = filtered.filter(i => i.status === 'error');
    renderHistoryGrid(container, filtered);
  });

  // Sort
  const sort = container.querySelector('#history-sort');
  sort?.addEventListener('sl-change', () => {
    const sorted = [...historyItems];
    if (sort.value === 'oldest') sorted.reverse();
    renderHistoryGrid(container, sorted);
  });

  // Go to workspace button (in empty state)
  const goBtn = container.querySelector('#goto-workspace');
  goBtn?.addEventListener('click', () => navigateTo('workspace'));
}


// ============================================================================
//  HELPERS
// ============================================================================
function escapeHtml(str) {
  const el = document.createElement('span');
  el.textContent = str;
  return el.innerHTML;
}

function formatDuration(seconds) {
  if (!seconds) return '—';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

function getDemoHistory() {
  // Demo data for when backend is not running
  return [
    {
      id: 'demo-1', pipeline: 'ti2vid_two_stages', status: 'completed', duration: 145,
      prompt: 'A cinematic aerial shot of a coastal city at sunset, golden light reflecting on glass skyscrapers',
      params: { seed: 42, num_inference_steps: 30, video_cfg_scale: 3.0, width: 1536, height: 1024 },
      created_at: '2026-04-08 10:30',
    },
    {
      id: 'demo-2', pipeline: 'ti2vid_two_stages', status: 'completed', duration: 98,
      prompt: 'Close-up of a cyberpunk character with neon reflections in rain',
      params: { seed: 123, num_inference_steps: 25, video_cfg_scale: 4.0, width: 1536, height: 1024 },
      created_at: '2026-04-08 11:15',
    },
    {
      id: 'demo-3', pipeline: 'distilled', status: 'error', duration: null,
      prompt: 'Abstract fluid simulation with metallic textures',
      params: { seed: 7, width: 1024, height: 1024 },
      created_at: '2026-04-08 11:45',
    },
  ];
}
