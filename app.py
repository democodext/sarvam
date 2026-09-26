import os
import sys
import tempfile
import webbrowser
from flask import Flask, jsonify, render_template_string, request

# Ensure UTF-8 output formatting for Windows console
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from mean_db import (
    DEFAULT_DB_PATH,
    get_all_commitments,
    get_all_conversations,
    get_commitment_by_id,
    get_conversation_by_id,
    load_env,
    update_commitment_status,
)
from mean_engine import (
    ConfigurationError,
    MeanEngineError,
    ProviderError,
    ResponseParsingError,
    SchemaValidationError,
    analyze_conversation,
)
from sarvam_transcription import (
    AudioValidationError,
    TranscriptionAuthError,
    TranscriptionProviderError,
    transcribe_audio,
)

app = Flask(__name__)


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = (
        "no-cache, no-store, must-revalidate, max-age=0"
    )
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MEAN AI — Conversational Intelligence Command Center</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-obsidian: #080b10;
            --panel-bg: rgba(15, 23, 42, 0.75);
            --panel-border: rgba(255, 255, 255, 0.1);
            --emerald-accent: #10b981;
            --emerald-glow: rgba(16, 185, 129, 0.25);
            --cyan-accent: #06b6d4;
            --cyan-glow: rgba(6, 182, 212, 0.2);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }
        html, body {
            background-color: #080b10 !important;
            color: #f8fafc !important;
            font-family: 'Plus Jakarta Sans', sans-serif;
            min-height: 100vh;
            margin: 0;
            padding: 0;
            overflow-x: hidden;
        }
        .code-font { font-family: 'JetBrains Mono', monospace; }
        .glass-panel {
            background: var(--panel-bg) !important;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--panel-border) !important;
            border-radius: 14px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45);
        }
        .sidebar { min-height: 100vh; border-right: 1px solid var(--panel-border); }
        .nav-btn {
            color: var(--text-muted);
            border-radius: 10px;
            padding: 10px 16px;
            font-weight: 500;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 10px;
            width: 100%;
            border: none;
            background: transparent;
            text-align: left;
        }
        .nav-btn:hover, .nav-btn.active {
            color: var(--text-main);
            background: rgba(16, 185, 129, 0.12);
            border-left: 3px solid var(--emerald-accent);
        }
        .status-dot {
            width: 8px; height: 8px;
            background-color: var(--emerald-accent);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--emerald-accent);
            display: inline-block;
        }
        /* Futuristic Neural Orb Visualizer */
        .orb-container {
            position: relative;
            width: 70px; height: 70px;
            display: flex; align-items: center; justify-content: center;
        }
        .orb-core {
            width: 28px; height: 28px;
            background: radial-gradient(circle, var(--cyan-accent) 0%, var(--emerald-accent) 100%);
            border-radius: 50%;
            box-shadow: 0 0 25px var(--emerald-accent);
            animation: pulse-glow 3s infinite ease-in-out;
        }
        .orb-ring {
            position: absolute;
            width: 56px; height: 56px;
            border: 1px dashed rgba(6, 182, 212, 0.4);
            border-radius: 50%;
            animation: rotate-ring 12s linear infinite;
        }
        @keyframes pulse-glow {
            0%, 100% { transform: scale(0.95); opacity: 0.8; }
            50% { transform: scale(1.1); opacity: 1; box-shadow: 0 0 35px var(--cyan-accent); }
        }
        @keyframes rotate-ring { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        
        .form-control, .form-select {
            background-color: rgba(15, 23, 42, 0.9) !important;
            border: 1px solid var(--panel-border) !important;
            color: #f8fafc !important;
            border-radius: 10px;
        }
        .form-control:focus, .form-select:focus {
            background-color: rgba(15, 23, 42, 0.98) !important;
            border-color: var(--emerald-accent) !important;
            color: #f8fafc !important;
            box-shadow: 0 0 0 0.25rem var(--emerald-glow) !important;
        }
        .btn-emerald {
            background: linear-gradient(135deg, #059669 0%, #10b981 100%);
            color: white !important; border: none; font-weight: 600; border-radius: 10px;
            box-shadow: 0 4px 14px var(--emerald-glow); transition: all 0.2s;
        }
        .btn-emerald:hover { background: linear-gradient(135deg, #10b981 0%, #34d399 100%); transform: translateY(-1px); }
        .evidence-quote {
            border-left: 3px solid var(--cyan-accent);
            background: rgba(6, 182, 212, 0.08);
            padding: 8px 14px; border-radius: 0 8px 8px 0; color: #cbd5e1;
        }
        .status-badge { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.5px; padding: 4px 8px; border-radius: 6px; }
        .spinner-overlay {
            display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(8, 11, 16, 0.9); backdrop-filter: blur(8px); z-index: 2000;
            align-items: center; justify-content: center;
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- LEFT SIDEBAR -->
            <div class="col-md-2 sidebar p-3 glass-panel d-flex flex-column justify-content-between">
                <div>
                    <div class="d-flex align-items-center gap-2 mb-1">
                        <div class="status-dot"></div>
                        <h2 class="h5 fw-bold mb-0 text-white tracking-wide">MEAN <span style="color:#10b981">AI</span></h2>
                    </div>
                    <p class="small text-muted mb-4 code-font" style="font-size: 0.72rem;">Conversational Intelligence</p>

                    <nav class="d-flex flex-column gap-2">
                        <button class="nav-btn active" onclick="showTab('workspace')">⚡ Command Center</button>
                        <button class="nav-btn" onclick="showTab('commitments'); loadCommitments();">📋 Commitments</button>
                        <button class="nav-btn" onclick="showTab('history'); loadConversations();">📜 History</button>
                    </nav>
                </div>

                <div class="p-2 glass-panel mt-auto rounded-3">
                    <div class="d-flex align-items-center justify-content-between">
                        <small class="text-muted">Sarvam SDK</small>
                        <span class="badge bg-success-subtle text-success border border-success-subtle">CONNECTED</span>
                    </div>
                </div>
            </div>

            <!-- MAIN WORKSPACE -->
            <div class="col-md-7 p-4">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <div>
                        <h1 class="h3 fw-bold mb-1">Conversational Intelligence</h1>
                        <p class="text-muted small mb-0">Understand what was said. Clarify what remains unsaid.</p>
                    </div>

                    <!-- Neural Orb Centerpiece -->
                    <div class="d-flex align-items-center gap-3 glass-panel px-3 py-2">
                        <div class="orb-container">
                            <div class="orb-ring"></div>
                            <div class="orb-core"></div>
                        </div>
                        <div>
                            <small class="text-muted d-block code-font" style="font-size:0.7rem;">ENGINE STATE</small>
                            <span class="fw-bold code-font" id="engineStateLabel" style="color:#10b981;">READY</span>
                        </div>
                    </div>
                </div>

                <!-- Command Center Panes -->
                <div id="tab-workspace">
                    <div class="glass-panel p-4 mb-4">
                        <ul class="nav nav-tabs border-bottom-0 mb-3" id="inputTab" role="tablist">
                            <li class="nav-item">
                                <button class="nav-link active code-font text-white bg-transparent border-0 border-bottom border-2" style="border-color:#10b981 !important;" id="text-mode-tab" data-bs-toggle="tab" data-bs-target="#text-mode" type="button">💬 TEXT INPUT</button>
                            </li>
                            <li class="nav-item">
                                <button class="nav-link code-font text-muted bg-transparent border-0" id="audio-mode-tab" data-bs-toggle="tab" data-bs-target="#audio-mode" type="button">🎙️ AUDIO FILE</button>
                            </li>
                        </ul>

                        <div class="tab-content" id="inputTabContent">
                            <!-- Text Mode -->
                            <div class="tab-pane fade show active" id="text-mode">
                                <div class="mb-3">
                                    <label class="form-label small text-muted">CONTEXT / BACKGROUND:</label>
                                    <input type="text" id="textContext" class="form-control" value="A client is discussing a video delivery deadline with an editor.">
                                </div>
                                <div class="mb-3">
                                    <label class="form-label small text-muted">CONVERSATION TRANSCRIPT:</label>
                                    <textarea id="textTranscript" class="form-control code-font" rows="5">Client: "Video Friday tak mil jayega?"&#10;Editor: "Haan, Friday tak bhej dunga."&#10;Client: "Friday morning?"&#10;Editor: "Haan bhai, dekh lenge."</textarea>
                                </div>
                                <button class="btn btn-emerald px-4 py-2" onclick="runTextAnalysis()">🚀 Run Intelligence Extraction</button>
                            </div>

                            <!-- Audio Mode -->
                            <div class="tab-pane fade" id="audio-mode">
                                <div class="mb-3">
                                    <label class="form-label small text-muted">SELECT AUDIO FILE (.mp3, .wav, .m4a, .flac, .webm):</label>
                                    <input type="file" id="audioFileInput" class="form-control" accept=".mp3,.wav,.m4a,.aac,.flac,.ogg,.opus,.webm,.amr">
                                </div>
                                <div class="mb-3">
                                    <label class="form-label small text-muted">CONTEXT:</label>
                                    <input type="text" id="audioContext" class="form-control" value="Audio transcript review">
                                </div>
                                <button class="btn btn-emerald px-4 py-2 mb-3" onclick="transcribeAudio()">🎙️ Transcribe Audio</button>

                                <div id="transcriptReviewArea" style="display: none;" class="glass-panel p-3 border border-info">
                                    <div class="badge bg-info text-dark mb-2" id="langMetadata">Language Detected: hi-IN</div>
                                    <label class="form-label small text-muted d-block">REVIEW & CORRECT TRANSCRIPT:</label>
                                    <textarea id="reviewedTranscript" class="form-control code-font mb-3" rows="4"></textarea>
                                    <button class="btn btn-emerald px-4 py-2" onclick="runAudioAnalysis()">🚀 Confirm & Analyze</button>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Analysis Output Container -->
                    <div id="resultsContainer" style="display: none;">
                        <div class="glass-panel p-4" style="border-color: rgba(16,185,129,0.3) !important;">
                            <h4 class="h5 fw-bold mb-3" style="color:#10b981;">📌 Analysis Results</h4>
                            
                            <div class="mb-4">
                                <h6 class="small text-muted code-font">EXECUTIVE SUMMARY</h6>
                                <p id="summaryText" class="lead fs-6 text-white mb-0"></p>
                            </div>

                            <div class="mb-4">
                                <h6 class="small text-muted code-font">EXPLICIT STATEMENTS</h6>
                                <ul id="statementsList" class="list-group list-group-flush bg-transparent"></ul>
                            </div>

                            <div class="mb-4">
                                <h6 class="small text-muted code-font">COMMITMENTS IDENTIFIED</h6>
                                <div id="commitmentsContainer"></div>
                            </div>

                            <div class="mb-4">
                                <h6 class="small text-muted code-font">AMBIGUITIES & UNRESOLVED POINTS</h6>
                                <div id="ambiguitiesContainer"></div>
                            </div>

                            <div class="p-3 rounded-3 mb-3" style="background: rgba(6,182,212,0.12); border: 1px solid rgba(6,182,212,0.3);">
                                <h6 class="small fw-bold text-info mb-1 code-font">💡 RECOMMENDED CLARIFICATION QUESTION</h6>
                                <p id="clarificationText" class="fs-6 fw-bold text-white mb-0"></p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Commitments Tab -->
                <div id="tab-commitments" style="display:none;">
                    <div class="glass-panel p-4">
                        <div class="d-flex justify-content-between align-items-center mb-4">
                            <h5 class="fw-bold mb-0">📋 Tracked Commitments</h5>
                            <div class="btn-group">
                                <button class="btn btn-sm btn-outline-light active" onclick="filterCommitments('all', this)">All</button>
                                <button class="btn btn-sm btn-outline-success" onclick="filterCommitments('explicit', this)">Explicit</button>
                                <button class="btn btn-sm btn-outline-warning" onclick="filterCommitments('pending', this)">Pending</button>
                                <button class="btn btn-sm btn-outline-danger" onclick="filterCommitments('overdue', this)">Overdue</button>
                                <button class="btn btn-sm btn-outline-info" onclick="filterCommitments('completed', this)">Completed</button>
                            </div>
                        </div>
                        <div id="commitmentsList"></div>
                    </div>
                </div>

                <!-- History Tab -->
                <div id="tab-history" style="display:none;">
                    <div class="glass-panel p-4">
                        <h5 class="fw-bold mb-4">📜 Conversation History</h5>
                        <div id="conversationsList"></div>
                    </div>
                </div>
            </div>

            <!-- RIGHT INTELLIGENCE BRIEF PANEL -->
            <div class="col-md-3 p-4">
                <div class="glass-panel p-3 mb-4">
                    <h5 class="h6 fw-bold mb-3 text-uppercase code-font" style="color:#10b981;">Intelligence Brief</h5>
                    
                    <div class="p-3 glass-panel rounded-3 mb-3 text-center">
                        <small class="text-muted d-block code-font" style="font-size:0.7rem;">UNRESOLVED CLARIFICATIONS</small>
                        <span class="h2 fw-bold mb-0" id="briefClarificationCount" style="color:#06b6d4;">0</span>
                    </div>

                    <div class="p-3 glass-panel rounded-3 mb-3 text-center">
                        <small class="text-muted d-block code-font" style="font-size:0.7rem;">TOTAL COMMITMENTS</small>
                        <span class="h2 fw-bold text-white mb-0" id="briefTotalCommitments">0</span>
                    </div>
                </div>

                <div class="glass-panel p-3">
                    <h6 class="small fw-bold text-muted code-font mb-3">RECENT ACTIVE DELIVERABLES</h6>
                    <div id="briefRecentDeliverables" class="d-flex flex-column gap-2">
                        <small class="text-muted">No active commitments.</small>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Status Update Modal -->
    <div class="modal fade" id="statusModal" tabindex="-1">
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content glass-panel text-white">
                <div class="modal-header border-bottom-0">
                    <h5 class="modal-title fw-bold">Update Commitment Status</h5>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <input type="hidden" id="modalCommId">
                    <div class="mb-3">
                        <label class="form-label small text-muted">NEW STATUS:</label>
                        <select id="modalNewStatus" class="form-select">
                            <option value="completed">Completed</option>
                            <option value="renegotiated">Renegotiated</option>
                            <option value="cancelled">Cancelled</option>
                        </select>
                    </div>
                    <div class="mb-3">
                        <label class="form-label small text-muted">NEW DEADLINE (IF RENEGOTIATED):</label>
                        <input type="text" id="modalNewDeadline" class="form-control" placeholder="e.g. 2026-10-20">
                    </div>
                    <div class="mb-3">
                        <label class="form-label small text-muted">REASON / NOTES:</label>
                        <textarea id="modalReason" class="form-control" rows="2"></textarea>
                    </div>
                </div>
                <div class="modal-footer border-top-0">
                    <button type="button" class="btn btn-outline-light" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn btn-emerald" onclick="submitStatusUpdate()">Confirm Change</button>
                </div>
            </div>
        </div>
    </div>

    <div class="spinner-overlay" id="spinnerOverlay">
        <div class="text-center">
            <div class="spinner-border mb-3" style="width: 3rem; height: 3rem; color:#10b981;"></div>
            <h5 class="fw-bold text-white code-font" id="spinnerMessage">Processing...</h5>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function setEngineState(state) {
            document.getElementById('engineStateLabel').innerText = state.toUpperCase();
        }
        function showSpinner(msg) {
            document.getElementById('spinnerMessage').innerText = msg;
            document.getElementById('spinnerOverlay').style.display = 'flex';
        }
        function hideSpinner() {
            document.getElementById('spinnerOverlay').style.display = 'none';
        }

        function showTab(name) {
            ['workspace', 'commitments', 'history'].forEach(t => {
                document.getElementById('tab-' + t).style.display = (t === name) ? 'block' : 'none';
            });
            document.querySelectorAll('.sidebar .nav-btn').forEach(b => b.classList.remove('active'));
            if (event && event.target) event.target.classList.add('active');
        }

        async function runTextAnalysis() {
            const context = document.getElementById('textContext').value.trim();
            const conversation = document.getElementById('textTranscript').value.trim();
            if (!conversation) return alert('Enter transcript.');

            setEngineState('ANALYZING');
            showSpinner('Extracting conversation intelligence...');
            try {
                const res = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ context, conversation })
                });
                const data = await res.json();
                hideSpinner();
                setEngineState('COMPLETE');
                if (data.error) return alert('Error: ' + data.error);
                renderResults(data);
                updateBrief();
            } catch(e) {
                hideSpinner(); setEngineState('READY');
                alert('Failed: ' + e.message);
            }
        }

        async function transcribeAudio() {
            const fileInput = document.getElementById('audioFileInput');
            if (!fileInput.files.length) return alert('Select audio file.');
            const formData = new FormData();
            formData.append('audio', fileInput.files[0]);

            setEngineState('TRANSCRIBING');
            showSpinner('Transcribing audio with Sarvam Speech-to-Text...');
            try {
                const res = await fetch('/api/transcribe', { method: 'POST', body: formData });
                const data = await res.json();
                hideSpinner(); setEngineState('READY');
                if (data.error) return alert('Transcription Error: ' + data.error);
                document.getElementById('langMetadata').innerText = `Language Detected: ${data.language_code || 'Unknown'}`;
                document.getElementById('reviewedTranscript').value = data.transcript || '';
                document.getElementById('transcriptReviewArea').style.display = 'block';
            } catch(e) {
                hideSpinner(); setEngineState('READY');
                alert('Failed: ' + e.message);
            }
        }

        async function runAudioAnalysis() {
            const context = document.getElementById('audioContext').value.trim();
            const conversation = document.getElementById('reviewedTranscript').value.trim();
            if (!conversation) return alert('Transcript is empty.');

            setEngineState('ANALYZING');
            showSpinner('Analyzing transcript...');
            try {
                const res = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ context, conversation })
                });
                const data = await res.json();
                hideSpinner(); setEngineState('COMPLETE');
                if (data.error) return alert('Error: ' + data.error);
                renderResults(data);
                updateBrief();
            } catch(e) {
                hideSpinner(); setEngineState('READY');
                alert('Failed: ' + e.message);
            }
        }

        function renderResults(data) {
            document.getElementById('summaryText').innerText = data.summary || 'N/A';
            const stmtList = document.getElementById('statementsList');
            stmtList.innerHTML = '';
            (data.explicit_statements || []).forEach(s => {
                const li = document.createElement('li');
                li.className = 'list-group-item bg-transparent text-slate-200 border-secondary-subtle';
                li.style.color = '#e2e8f0';
                li.innerText = '• ' + s;
                stmtList.appendChild(li);
            });

            const commDiv = document.getElementById('commitmentsContainer');
            commDiv.innerHTML = '';
            (data.commitments || []).forEach(c => {
                const badge = getStatusBadge(c.status);
                const card = document.createElement('div');
                card.className = 'glass-panel p-3 mb-2';
                card.innerHTML = `<div>${badge} <strong class="fs-6 text-white">${c.description}</strong></div>
                                  <div class="text-muted small mt-1">Responsible: ${c.responsible_person || 'Unspecified'} | Deadline: ${c.deadline || 'None'}</div>`;
                commDiv.appendChild(card);
            });

            const ambDiv = document.getElementById('ambiguitiesContainer');
            ambDiv.innerHTML = '';
            (data.ambiguities || []).forEach(a => {
                const card = document.createElement('div');
                card.className = 'glass-panel p-3 mb-2 border-warning-subtle';
                card.innerHTML = `<div class="fw-bold text-warning mb-1">⚠️ ${a.description}</div><div class="text-muted small">Why it matters: ${a.why_it_matters}</div>`;
                ambDiv.appendChild(card);
            });

            document.getElementById('clarificationText').innerText = data.clarification_question ? `👉 "${data.clarification_question}"` : 'N/A';
            document.getElementById('resultsContainer').style.display = 'block';
            document.getElementById('resultsContainer').scrollIntoView({ behavior: 'smooth' });
        }

        function getStatusBadge(status) {
            status = (status || 'explicit').toLowerCase();
            let color = 'bg-success text-white';
            if (status === 'completed') color = 'bg-success text-white';
            else if (status === 'overdue') color = 'bg-danger text-white';
            else if (status === 'cancelled') color = 'bg-secondary text-white';
            else if (status === 'renegotiated') color = 'bg-info text-dark';
            else if (status === 'pending' || status === 'unclear') color = 'bg-warning text-dark';
            return `<span class="badge ${color} status-badge me-2">${status.toUpperCase()}</span>`;
        }

        async function loadCommitments(filter = 'all') {
            try {
                const res = await fetch(`/api/commitments?status=${filter}`);
                const data = await res.json();
                const container = document.getElementById('commitmentsList');
                container.innerHTML = '';
                if (!data.length) { container.innerHTML = '<div class="text-muted p-3">No commitments found.</div>'; return; }
                data.forEach(c => {
                    const card = document.createElement('div'); card.className = 'glass-panel p-3 mb-3';
                    card.innerHTML = `
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                ${getStatusBadge(c.status)}
                                <strong class="fs-6 text-white">${c.description}</strong>
                                <div class="text-muted small mt-1">Responsible: <b>${c.responsible_person || 'Unspecified'}</b> | Deadline: <b>${c.deadline || 'None'}</b></div>
                                ${c.evidence && c.evidence.length ? `<div class="evidence-quote small mt-2">Quote: "${c.evidence.join('", "')}"</div>` : ''}
                            </div>
                            <button class="btn btn-outline-emerald btn-sm code-font" onclick="openStatusModal('${c.id}')">Update</button>
                        </div>
                    `;
                    container.appendChild(card);
                });
            } catch(e) { console.error(e); }
        }

        function filterCommitments(filter, btn) {
            document.querySelectorAll('#tab-commitments .btn-group button').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            loadCommitments(filter);
        }

        function openStatusModal(id) {
            document.getElementById('modalCommId').value = id;
            document.getElementById('modalReason').value = '';
            document.getElementById('modalNewDeadline').value = '';
            new bootstrap.Modal(document.getElementById('statusModal')).show();
        }

        async function submitStatusUpdate() {
            const id = document.getElementById('modalCommId').value;
            const new_status = document.getElementById('modalNewStatus').value;
            const reason = document.getElementById('modalReason').value;
            const new_deadline = document.getElementById('modalNewDeadline').value;

            if (!confirm(`Confirm status change to ${new_status.toUpperCase()}?`)) return;

            showSpinner('Updating status...');
            try {
                const res = await fetch(`/api/commitments/${id}/status`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ new_status, reason, new_deadline: new_deadline || null })
                });
                const data = await res.json();
                hideSpinner();
                bootstrap.Modal.getInstance(document.getElementById('statusModal')).hide();
                if (data.error) return alert('Error: ' + data.error);
                loadCommitments();
                updateBrief();
            } catch(e) { hideSpinner(); alert('Failed: ' + e.message); }
        }

        async function loadConversations() {
            try {
                const res = await fetch('/api/conversations');
                const data = await res.json();
                const container = document.getElementById('conversationsList');
                container.innerHTML = '';
                if (!data.length) { container.innerHTML = '<div class="text-muted p-3">No recorded conversations.</div>'; return; }
                data.forEach(c => {
                    const div = document.createElement('div'); div.className = 'glass-panel p-3 mb-3';
                    div.innerHTML = `
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <span class="badge bg-secondary code-font">ID: ${c.id}</span>
                            <small class="text-muted">${new Date(c.created_at).toLocaleString()}</small>
                        </div>
                        <h6 class="fw-bold mb-1 text-white">${c.context || 'Conversation'}</h6>
                        <p class="text-muted small mb-2">${c.summary || 'No summary available.'}</p>
                        <div class="glass-panel p-2 rounded small text-break border-0 text-slate-300" style="background: rgba(15,23,42,0.9);"><strong>Transcript:</strong> ${c.transcript}</div>
                    `;
                    container.appendChild(div);
                });
            } catch(e) { console.error(e); }
        }

        async function updateBrief() {
            try {
                const resComms = await fetch('/api/commitments?status=all');
                const comms = await resComms.json();
                document.getElementById('briefTotalCommitments').innerText = comms.length || 0;

                const resConvs = await fetch('/api/conversations');
                const convs = await resConvs.json();
                let ambCount = 0;
                convs.forEach(c => { if (c.clarification_question) ambCount++; });
                document.getElementById('briefClarificationCount').innerText = ambCount;

                const briefDiv = document.getElementById('briefRecentDeliverables');
                briefDiv.innerHTML = '';
                const active = comms.filter(c => ['explicit', 'pending', 'overdue'].includes(c.status)).slice(0, 3);
                if (!active.length) { briefDiv.innerHTML = '<small class="text-muted">No active commitments.</small>'; return; }
                active.forEach(c => {
                    const div = document.createElement('div');
                    div.className = 'glass-panel p-2 rounded mb-1';
                    div.innerHTML = `<div class="small fw-bold text-white">${c.description}</div><div class="text-muted" style="font-size:0.7rem;">${c.responsible_person || 'Unassigned'} • ${c.deadline || 'No deadline'}</div>`;
                    briefDiv.appendChild(div);
                });
            } catch(e) { console.error(e); }
        }

        document.addEventListener('DOMContentLoaded', () => { updateBrief(); });
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/conversations", methods=["GET"])
def api_get_conversations():
    try:
        conversations = get_all_conversations()
        return jsonify(conversations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/conversations/<conv_id>", methods=["GET"])
def api_get_conversation_detail(conv_id):
    try:
        conv = get_conversation_by_id(conv_id)
        return jsonify(conv)
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route("/api/commitments", methods=["GET"])
def api_get_commitments():
    status = request.args.get("status", "all")
    try:
        commitments = get_all_commitments(status_filter=status)
        return jsonify(commitments)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/commitments/<comm_id>/status", methods=["POST"])
def api_update_commitment_status(comm_id):
    data = request.get_json() or {}
    new_status = data.get("new_status")
    reason = data.get("reason", "")
    new_deadline = data.get("new_deadline")

    if not new_status:
        return jsonify({"error": "new_status is required"}), 400

    try:
        updated = update_commitment_status(
            commitment_id=comm_id,
            new_status=new_status,
            reason_or_notes=reason,
            new_deadline=new_deadline,
        )
        return jsonify(updated)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided in request."}), 400

    file = request.files["audio"]
    if not file.filename:
        return jsonify({"error": "Selected file has no filename."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        file.save(temp_file.name)
        temp_file.close()

        result = transcribe_audio(temp_file.name)
        return jsonify(result)
    except (
        AudioValidationError,
        TranscriptionAuthError,
        TranscriptionProviderError,
    ) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected transcription error: {e}"}), 500
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json() or {}
    context = data.get("context", "")
    conversation = data.get("conversation", "")

    if not conversation.strip():
        return jsonify({"error": "Conversation transcript cannot be empty."}), 400

    try:
        analysis = analyze_conversation(context, conversation)
        return jsonify(analysis)
    except (
        ConfigurationError,
        ProviderError,
        ResponseParsingError,
        SchemaValidationError,
    ) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected analysis error: {e}"}), 500


def main():
    load_env()
    api_key = os.getenv("SARVAM_API_KEY")

    if not api_key:
        print("\n❌ ERROR: SARVAM_API_KEY environment variable is missing in .env")
        sys.exit(1)

    port = 5000
    url = f"http://127.0.0.1:{port}"
    print("\n==================================================")
    print("  🚀 MEAN AI Command Center Web Server Running!")
    print(f"  Open in browser: {url}")
    print("==================================================\n")

    webbrowser.open(url)
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
