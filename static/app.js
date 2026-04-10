/* ─────────────────────────────────────────────────────────────
   標案文件解析系統 — 前端邏輯
   ───────────────────────────────────────────────────────────── */

'use strict';

// ═══════════ 陣營選擇器（全域狀態）═══════════════════════════════════════

let currentCamp = 'azure';
const _allParseModels = [];  // 快取 /api/models
const _allQueryModels = [];  // 快取 /api/query-models
const _allQueryTypes  = [];  // 快取 /api/query-types
const _allIndexModels = [];  // 快取 /api/index-models

/** 從 model_id 前綴推斷陣營（與後端 model_id_to_camp 一致）*/
function campOf(modelId) {
  if (!modelId) return 'azure';
  if (modelId.startsWith('ollama')) return 'local';
  if (modelId === 'gemini') return 'gemini';
  return 'azure';
}

// ═══════════════════════════════ Tab 切換 ═══════════════════════════════

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => {
      p.classList.remove('active');
      p.classList.add('hidden');
    });
    btn.classList.add('active');
    const panel = document.getElementById(`tab-${target}`);
    panel.classList.remove('hidden');
    panel.classList.add('active');

    if (target === 'parse') loadParseFolders();
    if (target === 'index') loadIndexFolders();
    if (target === 'query') loadQueryTypes();
  });
});

// ═══════════════════════════════ 上傳 Tab ═══════════════════════════════

const uploadForm        = document.getElementById('upload-form');
const folderNameInput   = document.getElementById('folder-name');
const docTypeSelect     = document.getElementById('doc-type');
const fileInput         = document.getElementById('file-input');
const dropZone          = document.getElementById('drop-zone');
const selectedFilesList = document.getElementById('selected-files-list');
const uploadBtn         = document.getElementById('upload-btn');
const uploadResult      = document.getElementById('upload-result');
const uploadMessage     = document.getElementById('upload-message');
const backupNotice      = document.getElementById('backup-notice');
const folderContents    = document.getElementById('folder-contents');
const folderContentsTitle = document.getElementById('folder-contents-title');
const folderFilesList   = document.getElementById('folder-files-list');

// ── 用 Map<filename, File> 作為唯一真值來源，解決多次拖曳後 fileInput.files 被 GC 的問題
const selectedFilesMap = new Map();

function mergeFiles(fileList) {
  Array.from(fileList).forEach(f => selectedFilesMap.set(f.name, f));
  renderSelectedFiles();
}

function renderSelectedFiles() {
  selectedFilesList.innerHTML = '';
  selectedFilesMap.forEach((_, name) => {
    const li = document.createElement('li');
    // 移除按鈕（×）讓使用者可以個別刪除
    li.innerHTML = `<span>${name}</span><button type="button" class="file-remove" data-name="${name}" title="移除">×</button>`;
    selectedFilesList.appendChild(li);
  });
}

selectedFilesList.addEventListener('click', e => {
  const btn = e.target.closest('.file-remove');
  if (btn) { selectedFilesMap.delete(btn.dataset.name); renderSelectedFiles(); }
});

fileInput.addEventListener('change', () => { mergeFiles(fileInput.files); fileInput.value = ''; });

// ── 防止瀏覽器在 drop-zone 外接收 drop 時開啟/下載檔案 ────────
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => {
  if (!dropZone.contains(e.target)) e.preventDefault();
});

// ── 點擊 drop-zone 觸發 file dialog（file input 因 pointer-events:none 無法自接收點擊）
dropZone.addEventListener('click', () => fileInput.click());

// ── 拖曳支援（使用計數器修正子元素 dragleave 誤觸發問題）────────
let _dragCounter = 0;

dropZone.addEventListener('dragenter', e => {
  e.preventDefault();
  _dragCounter++;
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => {
  _dragCounter--;
  if (_dragCounter <= 0) { _dragCounter = 0; dropZone.classList.remove('drag-over'); }
});
dropZone.addEventListener('dragover', e => e.preventDefault());
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  e.stopPropagation();
  _dragCounter = 0;
  dropZone.classList.remove('drag-over');
  mergeFiles(e.dataTransfer.files);
});

// ── 上傳提交 ────────────────────────────────────────────────

uploadForm.addEventListener('submit', async e => {
  e.preventDefault();

  const folderName = folderNameInput.value.trim();
  const docType    = docTypeSelect.value;

  if (!folderName) { showUploadMsg('請輸入資料夾名稱', 'error'); return; }
  if (!docType)    { showUploadMsg('請選擇文件類型', 'error'); return; }
  if (!selectedFilesMap.size) { showUploadMsg('請選擇至少一個文件', 'error'); return; }

  const formData = new FormData();
  formData.append('folder_name', folderName);
  formData.append('doc_type', docType);
  selectedFilesMap.forEach(f => formData.append('files', f));

  uploadBtn.disabled = true;
  uploadBtn.textContent = '上傳中…';
  uploadResult.classList.remove('hidden');
  showUploadMsg('正在上傳，請稍候…', 'info');
  backupNotice.classList.add('hidden');
  folderContents.classList.add('hidden');

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      showUploadMsg(`❌ 上傳失敗：${data.detail || res.statusText}`, 'error');
      return;
    }

    showUploadMsg(
      `✅ 成功上傳 ${data.saved_files.length} 個檔案至資料夾「${data.folder}」（類型：${data.doc_type}）`,
      'success'
    );

    if (data.backup) {
      backupNotice.textContent = `⚠️ 偵測到衝突檔案，原資料夾已備份至：${data.backup}`;
      backupNotice.classList.remove('hidden');
    }

    await refreshFolderContents(folderName);

    // 重置表單
    uploadForm.reset();
    selectedFilesMap.clear();
    selectedFilesList.innerHTML = '';

  } catch (err) {
    showUploadMsg(`❌ 網路錯誤：${err.message}`, 'error');
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.textContent = '⬆️ 上傳';
  }
});

async function refreshFolderContents(folderName) {
  try {
    const res = await fetch(`/api/folders/${encodeURIComponent(folderName)}/files`);
    if (!res.ok) return;
    const data = await res.json();
    folderContentsTitle.textContent = `📁 ${folderName}（${data.doc_type}）— 資料夾內容`;
    folderFilesList.innerHTML = data.files.map(f => `<li>${f}</li>`).join('');
    folderContents.classList.remove('hidden');
  } catch (_) { /* 靜默失敗 */ }
}

function showUploadMsg(text, type) {
  uploadMessage.textContent = text;
  uploadMessage.className   = `message ${type}`;
}

// ═══════════════════════════════ 解析 Tab ═══════════════════════════════

const parseFolderSelect  = document.getElementById('parse-folder');
const parseFileSelect    = document.getElementById('parse-file');
const parseModelSelect   = document.getElementById('parse-model');
const parseModelHint     = document.getElementById('parse-model-hint');
const refreshFoldersBtn  = document.getElementById('refresh-folders-btn');
const parseBtn           = document.getElementById('parse-btn');
const stopBtn            = document.getElementById('stop-btn');
const logOutput          = document.getElementById('log-output');
const logStatus          = document.getElementById('log-status');
const autoScrollChk      = document.getElementById('auto-scroll');
const clearLogBtn        = document.getElementById('clear-log-btn');

let activeEventSource = null;

// ── 載入可解析資料夾 ─────────────────────────────────────────

async function loadParseFolders() {
  parseFolderSelect.innerHTML = '<option value="" disabled selected>— 載入中 —</option>';
  parseBtn.disabled = true;

  // 並行載入資料夾與模型清單
  await Promise.all([_loadParseFoldersData(), loadModels()]);
}

async function _loadParseFoldersData() {
  parseFolderSelect.innerHTML = '<option value="" disabled selected>— 載入中 —</option>';
  parseBtn.disabled = true;

  try {
    const res  = await fetch('/api/parse-folders');
    const data = await res.json();

    parseFolderSelect.innerHTML = '<option value="" disabled selected>— 選擇資料夾 —</option>';
    if (!data.length) {
      parseFolderSelect.innerHTML = '<option value="" disabled selected>（尚無可解析資料夾）</option>';
      return;
    }
    data.forEach(folder => {
      const opt = document.createElement('option');
      opt.value       = folder.name;
      opt.textContent = `${folder.name}（${folder.doc_type}，${folder.file_count} 個檔案）${folder.parsing ? ' ⏳' : ''}`;
      opt.disabled    = folder.parsing;
      parseFolderSelect.appendChild(opt);
    });
  } catch (err) {
    parseFolderSelect.innerHTML = '<option value="" disabled selected>（載入失敗）</option>';
  }
}

// ── 載入 AI 模型清單 ────────────────────────────────

async function loadModels() {
  parseModelSelect.innerHTML = '<option value="" disabled selected>— 載入中 —</option>';
  try {
    const data = await fetch('/api/models').then(r => r.json());
    _allParseModels.length = 0;
    _allParseModels.push(...data);
    _renderParseModels();
  } catch (err) {
    parseModelSelect.innerHTML = '<option value="" disabled selected>（載入失敗）</option>';
  }
}

function _renderParseModels() {
  parseModelSelect.innerHTML = '';
  const filtered = _allParseModels.filter(m => m.camp === currentCamp);
  if (!filtered.length) {
    const opt = document.createElement('option');
    opt.disabled = opt.selected = true;
    opt.textContent = `（${currentCamp} 陣營無可用 Vision 模型）`;
    parseModelSelect.appendChild(opt);
    parseModelHint.textContent = '⚠️ 請切換陣營或在 .env 設定對應 Key';
    parseModelHint.style.color = 'var(--warning)';
    return;
  }
  let hasSelected = false;
  filtered.forEach(m => {
    const opt = document.createElement('option');
    opt.value       = m.id;
    opt.textContent = m.available ? m.label : `${m.label}  ⚠️（未設定 API Key）`;
    opt.disabled    = !m.available;
    if (!hasSelected && m.available) { opt.selected = true; hasSelected = true; }
    parseModelSelect.appendChild(opt);
  });
  parseModelHint.textContent = hasSelected ? '' : '⚠️ 沒有可用模型，請先設定 .env';
  parseModelHint.style.color = hasSelected ? '' : 'var(--warning)';
}

// ── 載入可選擇的檔案 ─────────────────────────────────────────

async function loadParseFiles(folderName) {
  parseFileSelect.innerHTML = '<option value="">全部（批次解析）</option>';
  if (!folderName) return;

  try {
    const res  = await fetch(`/api/parse-folders/${encodeURIComponent(folderName)}/files`);
    const data = await res.json();
    data.files.forEach(f => {
      const opt = document.createElement('option');
      opt.value       = f;
      opt.textContent = f;
      parseFileSelect.appendChild(opt);
    });
  } catch (_) { /* 靜默失敗 */ }
}

parseFolderSelect.addEventListener('change', () => {
  const selected = parseFolderSelect.value;
  parseBtn.disabled = !selected;
  loadParseFiles(selected);
});

refreshFoldersBtn.addEventListener('click', loadParseFolders);

// ── 清除 Log ─────────────────────────────────────────────────

clearLogBtn.addEventListener('click', () => {
  logOutput.textContent = '';
  setLogStatus('', '');
});

// ── 自動捲動 ─────────────────────────────────────────────────

function scrollLogToBottom() {
  if (autoScrollChk.checked) {
    logOutput.scrollTop = logOutput.scrollHeight;
  }
}

function appendLog(line) {
  logOutput.textContent += line + '\n';
  scrollLogToBottom();
}

function setLogStatus(text, cls) {
  logStatus.textContent = text;
  logStatus.className   = `log-status ${cls}`;
}

// ── 開始解析（SSE）───────────────────────────────────────────

parseBtn.addEventListener('click', async () => {
  const folderName = parseFolderSelect.value;
  const fileName   = parseFileSelect.value || null;
  const modelId    = parseModelSelect.value || null;

  if (!folderName) return;

  // 關閉既有連線（以防萬一）
  if (activeEventSource) {
    activeEventSource.close();
    activeEventSource = null;
  }

  logOutput.textContent = '';
  setLogStatus('⏳ 正在啟動解析…', 'running');
  parseBtn.disabled = true;
  stopBtn.classList.remove('hidden');

  // SSE 使用 POST，需透過 fetch + ReadableStream 模擬
  // （瀏覽器原生 EventSource 僅支援 GET，改用 fetch 手動解析 SSE stream）
  let abortController = new AbortController();
  activeEventSource = abortController; // 以 AbortController 充當「連線控制器」

  stopBtn.onclick = () => {
    abortController.abort();
    setLogStatus('⏹ 已中止連線', 'error');
    resetParseUI();
  };

  try {
    const res = await fetch('/api/parse/stream', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ folder_name: folderName, file_name: fileName, model_id: modelId }),
      signal:  abortController.signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      setLogStatus(`❌ ${err.detail}`, 'error');
      resetParseUI();
      return;
    }

    setLogStatus('🔄 解析中…', 'running');

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // 保留不完整的行

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const msg = line.slice(6);
          if (msg === 'done') {
            setLogStatus('✅ 解析完成！', 'done');
            resetParseUI();
            return;
          }
          appendLog(msg);
        } else if (line.startsWith('event: done')) {
          // event 在 data 前一行的情況
          setLogStatus('✅ 解析完成！', 'done');
          resetParseUI();
          return;
        }
        // 忽略心跳 ": heartbeat"
      }
    }

    // stream 正常結束
    setLogStatus('✅ 解析完成！', 'done');

  } catch (err) {
    if (err.name !== 'AbortError') {
      setLogStatus(`❌ 連線錯誤：${err.message}`, 'error');
      appendLog(`❌ 錯誤：${err.message}`);
    }
  } finally {
    resetParseUI();
  }
});

function resetParseUI() {
  activeEventSource = null;
  parseBtn.disabled = false;
  stopBtn.classList.add('hidden');
  // 重新整理資料夾狀態
  loadParseFolders();
}

// ═══════════════════════════════ 初始化 ═══════════════════════════════

// 頁面載入時若 parse tab 已顯示則載入資料夾（通常不會，預防性處理）
if (document.getElementById('tab-parse').classList.contains('active')) {
  loadParseFolders();
}

// ═══════════════════════ 索引 Tab ═══════════════════════

const indexFolderSelect  = document.getElementById('index-folder');
const refreshIndexBtn    = document.getElementById('refresh-index-btn');
const indexModelSelect   = document.getElementById('index-model');
const indexModelHint     = document.getElementById('index-model-hint');
const indexBtn           = document.getElementById('index-btn');
const indexStopBtn       = document.getElementById('index-stop-btn');
const indexFolderHint    = document.getElementById('index-folder-hint');
const indexLogOutput     = document.getElementById('index-log-output');
const indexLogStatus     = document.getElementById('index-log-status');
const indexAutoScrollChk = document.getElementById('index-auto-scroll');
const indexClearLogBtn   = document.getElementById('index-clear-log-btn');

let activeIndexAbort = null;

// ── 載入可建立索引的資料夾 ────────────────────────────

async function loadIndexFolders() {
  indexFolderSelect.innerHTML = '<option value="" disabled selected>— 載入中 —</option>';
  indexBtn.disabled = true;

  await Promise.all([_loadIndexFoldersData(), loadIndexModels()]);
}

async function _loadIndexFoldersData() {
  indexFolderSelect.innerHTML = '<option value="" disabled selected>— 選擇資料夾 —</option>';
  try {
    const res  = await fetch('/api/index-folders');
    const data = await res.json();
    if (!data.length) {
      indexFolderSelect.innerHTML = '<option value="" disabled selected>（尚無可建立索引的資料夾）</option>';
      return;
    }
    data.forEach(f => {
      const opt = document.createElement('option');
      opt.value       = f.name;
      const statusIcon  = f.indexed ? '✅' : '🆕';
      const settingsTag = f.has_settings ? '' : '  ⚠️無settings';
      opt.textContent = `${statusIcon} ${f.name}（${f.md_count} 個 .md${settingsTag}）${f.indexing ? ' ⏳' : ''}`;
      opt.disabled    = f.indexing;
      indexFolderSelect.appendChild(opt);
    });
  } catch (err) {
    indexFolderSelect.innerHTML = '<option value="" disabled selected>（載入失敗）</option>';
  }
}

// ── 載入 GraphRAG 索引模型清單 ─────────────────────────

async function loadIndexModels() {
  indexModelSelect.innerHTML = '<option value="" disabled selected>— 載入中 —</option>';
  try {
    const data = await fetch('/api/index-models').then(r => r.json());
    _allIndexModels.length = 0;
    _allIndexModels.push(...data);
    _renderIndexModels();
  } catch (err) {
    indexModelSelect.innerHTML = '<option value="" disabled selected>（載入失敗）</option>';
  }
}

function _renderIndexModels() {
  indexModelSelect.innerHTML = '';
  if (currentCamp === 'local') {
    const opt = document.createElement('option');
    opt.disabled = opt.selected = true;
    opt.textContent = '（Local 陣營：GraphRAG 索引由 settings_ollama.yaml 設定）';
    indexModelSelect.appendChild(opt);
    indexModelHint.textContent = '⚠️ 請確認 ragtest/local/{type}/ 目錄下有對應的 settings.yaml';
    indexModelHint.style.color = 'var(--warning)';
    return;
  }
  const filtered = _allIndexModels.filter(m => campOf(m.id) === currentCamp);
  let hasSelected = false;
  filtered.forEach(m => {
    const opt = document.createElement('option');
    opt.value       = m.id;
    opt.textContent = m.available ? m.label : `${m.label}  ⚠️（無 settings 模板）`;
    opt.disabled    = !m.available;
    if (!hasSelected && m.available) { opt.selected = true; hasSelected = true; }
    indexModelSelect.appendChild(opt);
  });
  if (!hasSelected) {
    indexModelHint.textContent = '⚠️ 沒有可用的索引模型，請檢查 settings.yaml 模板檔是否存在';
    indexModelHint.style.color = 'var(--warning)';
  } else {
    indexModelHint.textContent = '';
  }
}

indexFolderSelect.addEventListener('change', () => {
  const f = indexFolderSelect.value;
  indexBtn.disabled = !f;
  indexFolderHint.textContent = '';
});

refreshIndexBtn.addEventListener('click', loadIndexFolders);

indexClearLogBtn.addEventListener('click', () => {
  indexLogOutput.textContent = '';
  setIndexLogStatus('', '');
});

function appendIndexLog(line) {
  indexLogOutput.textContent += line + '\n';
  if (indexAutoScrollChk.checked) indexLogOutput.scrollTop = indexLogOutput.scrollHeight;
}

function setIndexLogStatus(text, cls) {
  indexLogStatus.textContent = text;
  indexLogStatus.className   = `log-status ${cls}`;
}

// ── 開始建立索引（SSE）────────────────────────────────

indexBtn.addEventListener('click', async () => {
  const folderName = indexFolderSelect.value;
  if (!folderName) return;

  const modelId = indexModelSelect.value || null;

  if (activeIndexAbort) { activeIndexAbort.abort(); activeIndexAbort = null; }

  indexLogOutput.textContent = '';
  setIndexLogStatus('⏳ 正在啟動索引建立…', 'running');
  indexBtn.disabled = true;
  indexStopBtn.classList.remove('hidden');

  const abortController = new AbortController();
  activeIndexAbort = abortController;

  indexStopBtn.onclick = () => {
    abortController.abort();
    setIndexLogStatus('⏹ 已中止連線', 'error');
    resetIndexUI();
  };

  try {
    const res = await fetch('/api/index/stream', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ folder_name: folderName, model_id: modelId }),
      signal:  abortController.signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      setIndexLogStatus(`❌ ${err.detail}`, 'error');
      resetIndexUI();
      return;
    }

    setIndexLogStatus('🔄 索引建立中…（可能需要數分鐘）', 'running');

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const msg = line.slice(6);
          if (msg === 'done') {
            setIndexLogStatus('✅ 索引建立完成！', 'done');
            resetIndexUI();
            return;
          }
          appendIndexLog(msg);
        } else if (line.startsWith('event: done')) {
          setIndexLogStatus('✅ 索引建立完成！', 'done');
          resetIndexUI();
          return;        } else if (line.startsWith('event: error')) {
          // 下一行會是 data: exit N，狀態已願示在 log。
          setIndexLogStatus('\u274c 索引建立失敗（請查看上方 Log）', 'error');
          resetIndexUI();
          return;        }
      }
    }
    setIndexLogStatus('✅ 索引建立完成！', 'done');

  } catch (err) {
    if (err.name !== 'AbortError') {
      setIndexLogStatus(`❌ 連線錯誤：${err.message}`, 'error');
      appendIndexLog(`❌ 錯誤：${err.message}`);
    }
  } finally {
    resetIndexUI();
  }
});

function resetIndexUI() {
  activeIndexAbort = null;
  indexBtn.disabled = false;
  indexStopBtn.classList.add('hidden');
  loadIndexFolders();
}

// ═══════════════════════════════ 查詢 Tab ═══════════════════════════════

const queryTypeSelect   = document.getElementById('query-type');
const queryTypeHint     = document.getElementById('query-type-hint');
const queryModeSelect   = document.getElementById('query-mode');
const queryModelSelect  = document.getElementById('query-model');
const queryModelHint    = document.getElementById('query-model-hint');
const queryInput        = document.getElementById('query-input');
const queryShowContext  = document.getElementById('query-show-context');
const queryBtn          = document.getElementById('query-btn');
const queryStopBtn      = document.getElementById('query-stop-btn');
const queryLogStatus    = document.getElementById('query-log-status');
const queryLogOutput    = document.getElementById('query-log-output');
const queryAutoScroll   = document.getElementById('query-auto-scroll');
const queryClearLogBtn  = document.getElementById('query-clear-log-btn');
const refreshQueryTypesBtn = document.getElementById('refresh-query-types-btn');

let activeQueryAbort = null;

function setQueryLogStatus(msg, state) {
  queryLogStatus.textContent = msg;
  queryLogStatus.className = 'log-status ' + (state || '');
}

function appendQueryLog(text) {
  if (!text || text.trim() === '') return;
  queryLogOutput.textContent += text + '\n';
  if (queryAutoScroll.checked) queryLogOutput.scrollTop = queryLogOutput.scrollHeight;
}

queryClearLogBtn.addEventListener('click', () => { queryLogOutput.textContent = ''; setQueryLogStatus('', ''); });
refreshQueryTypesBtn.addEventListener('click', loadQueryTypes);

async function loadQueryTypes() {
  try {
    const [typesData, modelsData] = await Promise.all([
      fetch('/api/query-types').then(r => r.json()),
      fetch('/api/query-models').then(r => r.json()),
    ]);
    _allQueryTypes.length = 0;  _allQueryTypes.push(...typesData);
    _allQueryModels.length = 0; _allQueryModels.push(...modelsData);
    _renderQueryTypes();
    _renderQueryModels();
  } catch (e) {
    queryTypeHint.textContent = '⚠️ 無法載入索引清單';
  }
}

function _renderQueryTypes() {
  queryTypeSelect.innerHTML = '<option value="" disabled selected>— 選擇資料類型 —</option>';
  const prefix   = currentCamp + '/';
  const filtered = _allQueryTypes.filter(t => t.id.startsWith(prefix));
  if (!filtered.length) {
    const opt = document.createElement('option');
    opt.disabled = opt.selected = true;
    opt.textContent = `（${currentCamp} 陣營尚無索引資料）`;
    queryTypeSelect.appendChild(opt);
  } else {
    filtered.forEach(t => {
      const opt = document.createElement('option');
      opt.value = t.id;
      opt.textContent = t.available ? t.label : `${t.label}  ⚠️ 尚未索引`;
      opt.disabled = !t.available;
      queryTypeSelect.appendChild(opt);
    });
    const firstAvail = filtered.find(t => t.available);
    if (firstAvail) queryTypeSelect.value = firstAvail.id;
  }
  updateQueryBtn();
}

function _renderQueryModels() {
  queryModelSelect.innerHTML = '';
  const filtered = _allQueryModels.filter(m => campOf(m.id) === currentCamp);
  if (!filtered.length) {
    const opt = document.createElement('option');
    opt.disabled = opt.selected = true;
    opt.textContent = `（${currentCamp} 陣營無可用查詢模型）`;
    queryModelSelect.appendChild(opt);
    queryModelHint.textContent = currentCamp === 'local' ? '⚠️ Local 陣營查詢模型尚未支援' : '⚠️ 無可用模型';
    queryModelHint.style.color = 'var(--warning)';
    return;
  }
  let hasSelected = false;
  filtered.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = m.available ? m.label : `${m.label}  ⚠️ 不可用`;
    opt.disabled = !m.available;
    if (m.available && !hasSelected) { queryModelSelect.value = m.id; hasSelected = true; }
    queryModelSelect.appendChild(opt);
  });
  queryModelHint.textContent = hasSelected ? '' : '⚠️ 無可用模型';
  queryModelHint.style.color = hasSelected ? '' : 'var(--warning)';
  updateQueryBtn();
}

function updateQueryBtn() {
  const ready = queryTypeSelect.value && queryModelSelect.value && queryInput.value.trim().length > 0;
  queryBtn.disabled = !ready;
}

queryTypeSelect.addEventListener('change', updateQueryBtn);
queryModelSelect.addEventListener('change', updateQueryBtn);
queryInput.addEventListener('input', updateQueryBtn);

queryStopBtn.addEventListener('click', () => {
  if (activeQueryAbort) { activeQueryAbort.abort(); activeQueryAbort = null; }
});

queryBtn.addEventListener('click', async () => {
  const docType     = queryTypeSelect.value;
  const mode        = queryModeSelect.value;
  const modelId     = queryModelSelect.value;
  const queryText   = queryInput.value.trim();
  const showContext = queryShowContext.checked;

  if (!docType || !queryText) { setQueryLogStatus('請選擇資料類型並輸入查詢問題', 'error'); return; }

  queryLogOutput.textContent = '';
  setQueryLogStatus('🔄 查詢中…', 'running');
  queryBtn.disabled = true;
  queryStopBtn.classList.remove('hidden');
  activeQueryAbort = new AbortController();

  try {
    const res = await fetch('/api/query', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ doc_type: docType, mode, query: queryText, show_context: showContext, model_id: modelId }),
      signal:  activeQueryAbort.signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      setQueryLogStatus(`❌ 錯誤：${err.detail}`, 'error');
      return;
    }

    setQueryLogStatus('🔄 查詢中…', 'running');

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const msg = line.slice(6);
          if (msg === 'done') {
            setQueryLogStatus('✅ 查詢完成！', 'done');
            resetQueryUI();
            return;
          }
          appendQueryLog(msg);
        } else if (line.startsWith('event: done')) {
          setQueryLogStatus('✅ 查詢完成！', 'done');
          resetQueryUI();
          return;
        } else if (line.startsWith('event: error')) {
          setQueryLogStatus('❌ 查詢失敗（請查看上方 Log）', 'error');
          resetQueryUI();
          return;
        }
      }
    }
    setQueryLogStatus('✅ 查詢完成！', 'done');

  } catch (err) {
    if (err.name !== 'AbortError') {
      setQueryLogStatus(`❌ 連線錯誤：${err.message}`, 'error');
      appendQueryLog(`❌ 錯誤：${err.message}`);
    }
  } finally {
    resetQueryUI();
  }
});

function resetQueryUI() {
  activeQueryAbort = null;
  queryBtn.disabled = false;
  queryStopBtn.classList.add('hidden');
  updateQueryBtn();
}

// ═══════ 陣營選擇器：監聽 radio 切換，重新過濾所有 Tab 的模型下拉 ═══════

document.querySelectorAll('input[name="camp"]').forEach(radio => {
  radio.addEventListener('change', () => {
    currentCamp = radio.value;
    document.querySelectorAll('.camp-opt').forEach(el => {
      el.classList.toggle('active', el.dataset.camp === currentCamp);
    });
    // 各 Tab 模型下拉即時重新過濾（不重新 fetch）
    _renderParseModels();
    _renderIndexModels();
    _renderQueryTypes();
    _renderQueryModels();
  });
});

