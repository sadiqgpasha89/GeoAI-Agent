const domainConfig = {
  flood: { label: 'FLOOD EVENT', title: 'Flood impact assessment', location: 'Riverbend, CA', prompt: 'Ask anything about this imagery...', alerts: 12 },
  wildfire: { label: 'WILDFIRE EVENT', title: 'Wildfire perimeter scan', location: 'Pine Ridge, CA', prompt: 'Ask about smoke, burn scars, or exposure...', alerts: 7 },
  earthquake: { label: 'SEISMIC EVENT', title: 'Earthquake impact assessment', location: 'Northpoint, CA', prompt: 'Ask about damage, debris, or blocked roads...', alerts: 18 },
  urban: { label: 'PLANNING STUDY', title: 'Urban growth assessment', location: 'East District, CA', prompt: 'Ask about land use, buildings, or access...', alerts: 4 },
  agriculture: { label: 'CROP HEALTH', title: 'Crop health assessment', location: 'Valley farm district', prompt: 'Ask about fields, stress, or anomalies...', alerts: 3 }
};

const responses = {
  'Describe this scene in detail': 'The scene shows an urban edge beside a broad water body. A divided road crosses the lower portion, with visible water encroachment along the central crossing. Built-up blocks sit at the upper-left; three structures appear within the affected zone.',
  'How many buildings are flooded?': 'Three buildings appear flooded near the upper-left of the scene. The count is based on visible water overlap and is estimated with 82% confidence. I have marked the two clearest evidence regions; the third is partially occluded.',
  'Is the road at the center flooded?': 'Yes. The central road appears completely inundated across the visible crossing. This is a high-impact observation with 91% confidence. A field check is recommended before reopening access.',
  'Summarize flood impact': 'Impact is concentrated around the central road crossing and upper-left building cluster. I found 3 potentially affected buildings and 1 compromised road segment. Overall assessment confidence is 91%.',
  'Find the most affected area': 'The highest-impact area is the central crossing, where water spans the full road width. The upper-left building cluster is the next priority for field verification.'
};

const state = { domain: 'flood', processing: false };
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function showToast(message) {
  const toast = $('#toast');
  toast.textContent = message;
  toast.classList.add('show');
  window.clearTimeout(showToast.timeout);
  showToast.timeout = window.setTimeout(() => toast.classList.remove('show'), 2600);
}

function addMessage(text, type = 'agent', confidence) {
  const conversation = $('#conversation');
  const wrapper = document.createElement('div');
  wrapper.className = type === 'user' ? 'user-message' : 'agent-message';
  if (type === 'user') {
    wrapper.innerHTML = `<div class="user-bubble">${text}</div>`;
  } else {
    wrapper.innerHTML = `<div class="mini-orb">✦</div><div><p>${text}</p>${confidence ? `<div class="response-meta"><span>✓ Grounded in image evidence</span><b>${confidence}% confidence</b></div>` : ''}</div>`;
  }
  conversation.append(wrapper);
  conversation.scrollTop = conversation.scrollHeight;
}

function normalizedPrompt(input) {
  const exact = Object.keys(responses).find((key) => key.toLowerCase() === input.trim().toLowerCase());
  if (exact) return exact;
  const lower = input.toLowerCase();
  if (lower.includes('building')) return 'How many buildings are flooded?';
  if (lower.includes('road')) return 'Is the road at the center flooded?';
  if (lower.includes('describe') || lower.includes('scene')) return 'Describe this scene in detail';
  if (lower.includes('impact') || lower.includes('summary')) return 'Summarize flood impact';
  if (lower.includes('area') || lower.includes('hotspot')) return 'Find the most affected area';
  return null;
}

function runAnalysis(prompt) {
  if (state.processing) return;
  state.processing = true;
  addMessage(prompt, 'user');
  const agent = $('#conversation');
  const thinking = document.createElement('div');
  thinking.className = 'agent-message thinking';
  thinking.innerHTML = '<div class="mini-orb">✦</div><div><p><span class="thinking-dots">● ● ●</span> Routing to visual grounding and geospatial tools…</p></div>';
  agent.append(thinking);
  agent.scrollTop = agent.scrollHeight;
  window.setTimeout(() => {
    thinking.remove();
    const key = normalizedPrompt(prompt);
    const answer = key ? responses[key] : `I can investigate that against the current ${domainConfig[state.domain].title.toLowerCase()}. Try asking about affected assets, scene description, or a specific object to ground.`;
    addMessage(answer, 'agent', key === 'Is the road at the center flooded?' ? 91 : key ? 82 : null);
    state.processing = false;
  }, 720);
}

function setDomain(domain) {
  state.domain = domain;
  const config = domainConfig[domain];
  $$('.domain').forEach((item) => item.classList.toggle('active', item.dataset.domain === domain));
  $('#domainEyebrow').textContent = config.label;
  $('#workspaceDescription').textContent = `${config.title} · ${config.location}`;
  $('#alertCount').innerHTML = `${config.alerts} <small>events</small>`;
  $('#chatInput').placeholder = config.prompt;
  $('.breadcrumbs strong').textContent = config.title;
  showToast(`${config.title} workspace loaded`);
}

$$('.domain').forEach((item) => item.addEventListener('click', () => setDomain(item.dataset.domain)));
$$('[data-prompt]').forEach((item) => item.addEventListener('click', () => runAnalysis(item.dataset.prompt)));
$('#chatForm').addEventListener('submit', (event) => {
  event.preventDefault();
  const input = $('#chatInput');
  const prompt = input.value.trim();
  if (!prompt) return;
  input.value = '';
  runAnalysis(prompt);
});
$('#newAnalysis').addEventListener('click', () => {
  $('#fileInput').click();
});
$('#fileInput').addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    showToast('Please choose a supported image file');
    return;
  }
  const imageUrl = URL.createObjectURL(file);
  $('#sceneImage').src = imageUrl;
  $('.image-title strong').textContent = file.name;
  $('.image-title small').textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB · Local upload`;
  $('#conversation').innerHTML = '<div class="agent-message"><div class="mini-orb">✦</div><div><p>New analysis ready. Upload an image or ask me what to investigate.</p></div></div>';
  showToast(`${file.name} loaded for analysis`);
});
$('#evidenceToggle').addEventListener('click', () => {
  $('#imageStage').classList.toggle('hide-evidence');
  showToast($('#imageStage').classList.contains('hide-evidence') ? 'Evidence overlays hidden' : 'Evidence overlays visible');
});
$$('.view-control').forEach((button) => button.addEventListener('click', () => {
  $$('.view-control').forEach((item) => item.classList.remove('active'));
  button.classList.add('active');
  if (button.textContent.includes('Map')) showToast('Map layer preview enabled');
  if (button.textContent.includes('Compare')) showToast('Comparison mode needs a second image');
}));
