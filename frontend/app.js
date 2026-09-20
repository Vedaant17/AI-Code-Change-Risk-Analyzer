const form = document.getElementById("analysis-form");
const repoInput = document.getElementById("repo-url");
const shaInput = document.getElementById("commit-sha");
const submitBtn = document.getElementById("submit-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

function setLoading(loading) {
  submitBtn.disabled = loading;
  repoInput.disabled = loading;
  shaInput.disabled = loading;
  statusEl.hidden = !loading;
  if (loading) {
    statusEl.setAttribute("role", "status");
    statusEl.className = "loading";
    statusEl.textContent = "Analysing commit...";
    resultsEl.hidden = true;
  }
}

function showError(message) {
  statusEl.hidden = false;
  statusEl.className = "error";
  statusEl.setAttribute("role", "alert");
  statusEl.textContent = message;
  resultsEl.hidden = true;
}

function renderResults(data) {
  statusEl.hidden = true;
  resultsEl.hidden = false;

  const summary = resultsEl.querySelector(".summary");
  summary.innerHTML =
    `<span><strong>Repository:</strong> ${esc(data.repo_url)}</span>` +
    `<span><strong>Commit:</strong> ${esc(data.short_sha)}</span>` +
    `<span><strong>Strategy:</strong> ${esc(data.strategy)} v${esc(data.strategy_version)}</span>` +
    `<span><strong>Files analysed:</strong> ${data.files_analyzed} of ${data.total_files}</span>` +
    `<span><strong>Time:</strong> ${data.elapsed_ms.toFixed(0)} ms</span>`;

  const tbody = document.getElementById("file-tbody");
  tbody.innerHTML = "";
  data.files.forEach(function (f) {
    const pct = (f.investigation_priority_score * 100).toFixed(0);
    const scoreLabel = f.rank === 0 ? "—" : f.investigation_priority_score.toFixed(2);
    const badgeClass = f.is_binary ? "binary" : f.status;
    const lines = f.rank === 0 ? "" : `+${f.lines_added} -${f.lines_deleted}`;
    const row = document.createElement("tr");
    row.innerHTML =
      `<td class="num">${f.rank || "—"}</td>` +
      `<td class="path">${esc(f.path)}</td>` +
      `<td><div class="score-cell" aria-label="Investigation priority ${scoreLabel}">` +
        `<div class="score-bar"><div class="score-fill" style="width:${pct}%"></div></div>` +
        `<span class="score-value">${scoreLabel}</span>` +
      `</div></td>` +
      `<td><span class="badge ${badgeClass}">${esc(f.status)}</span></td>` +
      `<td class="num">${lines}</td>`;
    tbody.appendChild(row);
  });

  const warningsEl = document.getElementById("warnings");
  warningsEl.innerHTML = "";
  if (data.warnings && data.warnings.length) {
    data.warnings.forEach(function (w) {
      const div = document.createElement("div");
      div.className = "warn";
      div.textContent = w;
      warningsEl.appendChild(div);
    });
  }

  const limEl = document.getElementById("limitations-list");
  limEl.innerHTML = "";
  if (data.limitations && data.limitations.length) {
    const ul = document.createElement("ul");
    data.limitations.forEach(function (l) {
      const li = document.createElement("li");
      li.textContent = l;
      ul.appendChild(li);
    });
    limEl.appendChild(ul);
  }

  const semantics = document.createElement("p");
  semantics.style.cssText = "font-size:0.8rem;color:#666;margin-top:0.5rem;padding:0 1rem 0.75rem";
  semantics.textContent = "Investigation priority score \u2014 higher values indicate files that should be investigated earlier. This is not a probability that a file contains a defect.";
  limEl.appendChild(semantics);
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

form.addEventListener("submit", function (e) {
  e.preventDefault();
  const repo = repoInput.value.trim();
  const sha = shaInput.value.trim();
  if (!repo || !sha) return;

  setLoading(true);

  fetch("/analysis/risk", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo_url: repo, commit_sha: sha }),
  })
    .then(function (resp) {
      if (!resp.ok) {
        return resp.json().then(function (data) {
          throw new Error(data.detail || "Analysis failed (status " + resp.status + ")");
        });
      }
      return resp.json();
    })
    .then(function (data) {
      setLoading(false);
      renderResults(data);
    })
    .catch(function (err) {
      setLoading(false);
      showError(err.message || "Could not connect to the server. Ensure the backend is running.");
    });
});
