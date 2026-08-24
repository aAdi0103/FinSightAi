document.addEventListener("DOMContentLoaded", () => {
    // Theme Switcher it runs when full html loaded 
    initTheme();
    initTabs();
    initDateTimePicker();
    initInputValidation();

    // these are the api calls
    initSingleInvestigation();
    initKnowledgeBase();
    checkSystemHealth();
});


function initTheme() {
    const toggleBtn = document.getElementById("theme-toggle");
    const themeIcon = document.getElementById("theme-icon");
    const html = document.documentElement;

    const savedTheme = localStorage.getItem("finsight-theme") || "dark";
    html.setAttribute("data-theme", savedTheme);
    updateThemeIcon(savedTheme);

    toggleBtn.addEventListener("click", () => {
        const currentTheme = html.getAttribute("data-theme");
        const nextTheme = currentTheme === "dark" ? "light" : "dark";
        html.setAttribute("data-theme", nextTheme);
        localStorage.setItem("finsight-theme", nextTheme);
        updateThemeIcon(nextTheme);
        showToast(`Switched to ${nextTheme.toUpperCase()} theme`);
    });

    function updateThemeIcon(theme) {
        if (theme === "dark") {
            themeIcon.className = "fa-solid fa-moon";
        } else {
            themeIcon.className = "fa-solid fa-sun";
        }
    }
}

function initTabs() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            const targetId = btn.getAttribute("data-tab");
            const targetEl = document.getElementById(targetId);
            if (targetEl) targetEl.classList.add("active");
        });
    });
}

function initDateTimePicker() {
    const tsInput = document.getElementById("txn-timestamp");
    if (tsInput && !tsInput.value) {
        const now = new Date();
        const year = now.getFullYear();
        const month = String(now.getMonth() + 1).padStart(2, "0");
        const day = String(now.getDate()).padStart(2, "0");
        const hours = String(now.getHours()).padStart(2, "0");
        const mins = String(now.getMinutes()).padStart(2, "0");
        tsInput.value = `${year}-${month}-${day}T${hours}:${mins}`;
    }
}

function initInputValidation() {
    // Decimal numeric inputs (Amount, Sender Balances, Recipient Balances)
    const decimalInputs = document.querySelectorAll("#txn-amount, #sender-old, #sender-new, #dest-old, #dest-new");
    decimalInputs.forEach(input => {
        input.setAttribute("inputmode", "decimal");

        // Prevent typing non-numeric keys
        input.addEventListener("keydown", (e) => {
            // Allow navigation and edit controls
            if (["Backspace", "Delete", "Tab", "Escape", "Enter", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(e.key)) {
                return;
            }
            // Allow copy / paste / select-all shortcuts
            if ((e.ctrlKey || e.metaKey) && ["a", "c", "v", "x"].includes(e.key.toLowerCase())) {
                return;
            }
            // Allow a single decimal point if not present
            if (e.key === "." && !input.value.includes(".")) {
                return;
            }
            // Block anything that is not a digit 0-9
            if (!/^[0-9]$/.test(e.key)) {
                e.preventDefault();
            }
        });

        // Sanitize value on paste or input
        input.addEventListener("input", () => {
            let val = input.value;
            val = val.replace(/[^0-9.]/g, "");
            const parts = val.split(".");
            if (parts.length > 2) {
                val = parts[0] + "." + parts.slice(1).join("");
            }
            if (input.value !== val) {
                input.value = val;
            }
        });
    });
}

function initSingleInvestigation() {
    const form = document.getElementById("investigate-form");
    const btnPredict = document.getElementById("btn-predict-only");
    const btnBack = document.getElementById("btn-back-to-params");
    const closeQuickModal = document.getElementById("btn-quick-modal-close");
    const quickModal = document.getElementById("quick-ml-modal");

    if (form) {
        form.addEventListener("submit", async (e) => {
            e.preventDefault();
            const payload = extractFormData();
            await runInvestigation(payload);
        });
    }

    if (btnPredict) {
        btnPredict.addEventListener("click", async () => {
            const payload = extractFormData();
            await runQuickPredict(payload);
        });
    }

    if (btnBack) {
        btnBack.addEventListener("click", () => {
            showParametersView();
        });
    }

    if (closeQuickModal && quickModal) {
        closeQuickModal.addEventListener("click", () => {
            quickModal.style.display = "none";
        });
        quickModal.addEventListener("click", (e) => {
            if (e.target === quickModal) {
                quickModal.style.display = "none";
            }
        });
    }
}

function showAnalysisView() {
    const paramsView = document.getElementById("parameters-view");
    const analysisView = document.getElementById("analysis-view");
    if (paramsView) paramsView.style.display = "none";
    if (analysisView) {
        analysisView.style.display = "block";
        window.scrollTo({ top: 0, behavior: "smooth" });
    }
}

function showParametersView() {
    const paramsView = document.getElementById("parameters-view");
    const analysisView = document.getElementById("analysis-view");
    if (analysisView) analysisView.style.display = "none";
    if (paramsView) {
        paramsView.style.display = "block";
        window.scrollTo({ top: 0, behavior: "smooth" });
    }
}

function extractFormData() {
    const tsInput = document.getElementById("txn-timestamp");
    const timestampVal = (tsInput && tsInput.value) ? tsInput.value : new Date().toISOString();

    return {
        timestamp: timestampVal,
        type: document.getElementById("txn-type").value,
        amount: parseFloat(document.getElementById("txn-amount").value) || 0.0,
        nameOrig: document.getElementById("sender-orig").value || "C_UNKNOWN",
        oldbalanceOrg: parseFloat(document.getElementById("sender-old").value) || 0.0,
        newbalanceOrig: parseFloat(document.getElementById("sender-new").value) || 0.0,
        nameDest: document.getElementById("dest-id").value || "C_UNKNOWN",
        oldbalanceDest: parseFloat(document.getElementById("dest-old").value) || 0.0,
        newbalanceDest: parseFloat(document.getElementById("dest-new").value) || 0.0
    };
}

async function runInvestigation(payload) {
    const btn = document.getElementById("btn-investigate");
    const origHtml = btn ? btn.innerHTML : "";
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...`;
    }

    try {
        const res = await fetch("/api/v1/investigate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `API returned status ${res.status}`);
        }

        const data = await res.json();
        renderDossier(data);
        showAnalysisView();
        showToast("AI Risk Investigation completed successfully!");
    } catch (err) {
        console.error(err);
        showToast(`Investigation failed: ${err.message}`, true);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = origHtml;
        }
    }
}

async function runQuickPredict(payload) {
    const btn = document.getElementById("btn-predict-only");
    const origHtml = btn ? btn.innerHTML : "";
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Scoring...`;
    }

    try {
        const res = await fetch("/api/v1/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `API status ${res.status}`);
        }
        const data = await res.json();
        showQuickMLModal(data);
    } catch (err) {
        showToast("Prediction failed: " + err.message, true);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = origHtml;
        }
    }
}

function showQuickMLModal(data) {
    const modal = document.getElementById("quick-ml-modal");
    const verdictBadge = document.getElementById("quick-verdict-badge");
    const scoreVal = document.getElementById("quick-score-val");
    const expText = document.getElementById("quick-explanation-text");

    if (!modal) return;

    const prediction = data.prediction || "Normal";
    const score = data.anomaly_score;
    const isSuspicious = prediction.toLowerCase().includes("suspicious");

    if (verdictBadge) {
        verdictBadge.textContent = prediction.toUpperCase();
        verdictBadge.className = `quick-verdict-badge ${isSuspicious ? "badge-suspicious" : "badge-normal"}`;
    }

    if (scoreVal) {
        scoreVal.textContent = typeof score === "number" ? score.toFixed(4) : "N/A";
    }

    if (expText) {
        if (isSuspicious) {
            expText.textContent = "Transaction flagged as ANOMALOUS by Isolation Forest ensemble. Full RAG investigation recommended.";
        } else {
            expText.textContent = "Transaction matches expected historical baseline behavior. No immediate anomaly detected.";
        }
    }

    modal.style.display = "flex";
}

function renderDossier(data) {
    if (!data) return;

    const emptyState = document.getElementById("dossier-empty");
    const content = document.getElementById("dossier-content");
    if (emptyState) emptyState.style.display = "none";
    if (content) content.style.display = "block";

    const riskLevel = (data.risk_stratification?.risk_level || "MEDIUM").toUpperCase();
    const riskClass = riskLevel.toLowerCase();
    const banner = document.getElementById("dossier-banner");
    if (banner) banner.className = `dossier-banner risk-${riskClass}`;

    const badge = document.getElementById("risk-badge");
    if (badge) badge.textContent = `${riskLevel} RISK`;

    const score = document.getElementById("banner-score");
    if (score) score.textContent = `Severity Score: ${data.risk_stratification?.severity_score ?? "N/A"}/100`;

    const ml = document.getElementById("banner-ml");
    if (ml) {
        const anomalyScore = data.ml_analysis?.anomaly_score;
        ml.textContent = `ML Anomaly Score: ${typeof anomalyScore === "number" ? anomalyScore.toFixed(4) : "N/A"}`;
    }

    // Executive Summary
    const execSum = document.getElementById("executive-summary-text");
    if (execSum) {
        execSum.textContent = data.investigation_report?.executive_summary || "Investigation summary generated.";
    }

    // Risk Indicators
    const indicatorsList = document.getElementById("indicators-list");
    if (indicatorsList) {
        const indicators = data.risk_stratification?.triggered_indicators || [];
        if (indicators.length > 0) {
            indicatorsList.innerHTML = indicators
                .map(ind => `<li><i class="fa-solid fa-triangle-exclamation" style="color: var(--risk-${riskClass})"></i> ${ind.replace(/^•\s*/, "")}</li>`)
                .join("");
        } else {
            indicatorsList.innerHTML = `<li><i class="fa-solid fa-circle-check" style="color: var(--risk-safe)"></i> All baseline parameters conform to standard behavior.</li>`;
        }
    }

    // Regulations
    const regsContainer = document.getElementById("regulations-container");
    if (regsContainer) {
        const regs = data.investigation_report?.regulatory_compliance_notes || [];
        if (Array.isArray(regs) && regs.length > 0) {
            regsContainer.innerHTML = regs.map(r => `
                <div class="citation-card">
                    <div class="citation-header">
                        <span class="citation-code">${r.code || "REGULATION"}</span>
                        <span style="font-size:0.75rem; color:var(--text-muted);">${r.authority || "Regulatory Body"}</span>
                    </div>
                    <div class="citation-title">${r.title || "Regulatory Directive"}</div>
                    <div style="font-size:0.78rem; color:var(--brand-primary); margin-bottom:0.35rem;">${r.clause || ""}</div>
                    <div class="citation-body">${r.relevance_summary || ""}</div>
                </div>
            `).join("");
        } else {
            regsContainer.innerHTML = `<div class="citation-card"><div class="citation-body">Standard KYC & transaction monitoring guidelines satisfied.</div></div>`;
        }
    }

    // Historical Precedents
    const casesContainer = document.getElementById("precedents-container");
    if (casesContainer) {
        const cases = data.investigation_report?.similar_precedents || [];
        if (Array.isArray(cases) && cases.length > 0) {
            casesContainer.innerHTML = cases.map(c => `
                <div class="precedent-card">
                    <div class="precedent-header">
                        <span class="precedent-id">${c.case_id || "CASE"}</span>
                        <span style="font-size:0.75rem; color:var(--text-muted);">Match Score: ${typeof c.similarity_score === "number" ? (c.similarity_score * 100).toFixed(1) + "%" : "N/A"}</span>
                    </div>
                    <div class="precedent-title">${c.title || "Precedent Case"}</div>
                    <div style="font-size:0.78rem; color:var(--risk-high); margin-bottom:0.35rem;">Typology: ${c.typology || "Uncategorized"}</div>
                    <div class="precedent-body"><strong>Resolution:</strong> ${c.historical_outcome || "Resolved"}</div>
                </div>
            `).join("");
        } else {
            casesContainer.innerHTML = `<div class="precedent-card"><div class="precedent-body">No anomalous historical matches found.</div></div>`;
        }
    }

    // Actions Checklist
    const actionsContainer = document.getElementById("actions-container");
    if (actionsContainer) {
        const actions = data.investigation_report?.recommended_actions || [];
        if (Array.isArray(actions) && actions.length > 0) {
            actionsContainer.innerHTML = actions.map((act, i) => `
                <label class="check-item">
                    <input type="checkbox" id="act-chk-${i}">
                    <span>${act}</span>
                </label>
            `).join("");
        } else {
            actionsContainer.innerHTML = `<div class="check-item"><span>Proceed with standard clearing and monitoring.</span></div>`;
        }
    }
}


function initKnowledgeBase() {
    const searchBtn = document.getElementById("btn-kb-search");
    const searchInput = document.getElementById("kb-search-input");

    searchBtn.addEventListener("click", () => searchKB(searchInput.value));
    searchInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") searchKB(searchInput.value);
    });

    // Initial load
    searchKB("account draining wire transfer mule ring");
}

async function searchKB(query) {
    if (!query) query = "rbi fraud rules";
    const container = document.getElementById("kb-results-container");
    container.innerHTML = `<div style="grid-column: 1/-1; text-align:center; padding: 2rem; color:var(--text-muted);"><i class="fa-solid fa-spinner fa-spin"></i> Searching Knowledge Base...</div>`;

    try {
        const res = await fetch(`/api/v1/knowledge/search?q=${encodeURIComponent(query)}&top_k=6`);
        const data = await res.json();

        const allDocs = [
            ...(data.regulations || []).map(r => ({ ...r, badge: "REGULATION" })),
            ...(data.historical_cases || []).map(c => ({ ...c, badge: "HISTORICAL CASE" }))
        ];

        if (allDocs.length === 0) {
            container.innerHTML = `<div style="grid-column: 1/-1; text-align:center; padding: 2rem; color:var(--text-muted);">No matching directives found.</div>`;
            return;
        }

        container.innerHTML = allDocs.map(doc => `
            <div class="kb-card">
                <div class="kb-card-meta">
                    <span class="kb-authority">${doc.authority || "Regulatory Body"}</span>
                    <span class="table-badge badge-medium">${doc.badge}</span>
                </div>
                <div class="citation-title">${doc.title}</div>
                <div class="citation-body">${doc.content || doc.summary}</div>
                ${doc.resolution ? `<div style="font-size:0.75rem; color:var(--risk-safe); margin-top:0.35rem;"><strong>Outcome:</strong> ${doc.resolution}</div>` : ""}
            </div>
        `).join("");
    } catch (err) {
        container.innerHTML = `<div style="grid-column: 1/-1; color:var(--risk-critical);">Failed to search knowledge base: ${err.message}</div>`;
    }
}

async function checkSystemHealth() {
    try {
        const res = await fetch("/health");
        const data = await res.json();
        const pill = document.getElementById("system-status-pill");
        const text = document.getElementById("status-text");

        if (data.status === "READY") {
            text.textContent = "ML & RAG Engine: Online";
        } else {
            text.textContent = "System: Degraded";
            pill.querySelector(".status-dot").style.backgroundColor = "var(--risk-medium)";
        }
    } catch {
        document.getElementById("status-text").textContent = "API: Connecting...";
    }
}

function showToast(message, isError = false) {
    const toast = document.getElementById("toast");
    toast.textContent = message;
    toast.style.borderColor = isError ? "var(--risk-critical)" : "var(--brand-primary)";
    toast.style.display = "block";

    setTimeout(() => {
        toast.style.display = "none";
    }, 3500);
}
