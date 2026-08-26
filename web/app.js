const form = document.getElementById("search-form");
const urlInput = document.getElementById("url");
const dialogueInput = document.getElementById("dialogue");
const urlError = document.getElementById("url-error");
const dialogueError = document.getElementById("dialogue-error");
const submitBtn = document.getElementById("submit-btn");
const progressSection = document.getElementById("progress");
const progressStage = document.getElementById("progress-stage");
const resultsSection = document.getElementById("results");

const STATUS_LABELS = {
  HIGH_CONFIDENCE: { label: "High confidence", cls: "badge-ok" },
  MEDIUM_CONFIDENCE: { label: "Medium confidence", cls: "badge-ok" },
  LOW_CONFIDENCE: { label: "Low confidence", cls: "badge-warn" },
  AMBIGUOUS_MATCH: { label: "Ambiguous match", cls: "badge-warn" },
  NO_CONFIDENT_MATCH: { label: "No confident match", cls: "badge-error" },
};

function resetUi() {
  urlError.hidden = true;
  dialogueError.hidden = true;
  progressSection.hidden = true;
  resultsSection.hidden = true;
  resultsSection.className = "card results";
  resultsSection.innerHTML = "";
}

function setStage(text) {
  progressStage.textContent = text;
}

function showFieldError(el, message) {
  el.textContent = message;
  el.hidden = false;
}

function validate(url, dialogue) {
  let ok = true;
  const urlPattern = /^https?:\/\/(www\.)?(youtube\.com|youtu\.be|ok\.ru|odnoklassniki\.ru)\//i;
  if (!urlPattern.test(url.trim())) {
    showFieldError(urlError, "Please enter a valid YouTube or OK.ru video URL.");
    ok = false;
  }
  if (!dialogue.trim()) {
    showFieldError(dialogueError, "Please enter the dialogue you're looking for.");
    ok = false;
  }
  return ok;
}

function renderError(message) {
  resultsSection.classList.add("status-error");
  resultsSection.innerHTML = `
    <span class="badge badge-error">Error</span>
    <p class="result-message">${escapeHtml(message)}</p>
  `;
  resultsSection.hidden = false;
}

function renderResult(query, result) {
  const info = STATUS_LABELS[result.status] || { label: result.status, cls: "badge-warn" };
  const hasImage = Boolean(result.image_url);
  const noMatch = result.status === "NO_CONFIDENT_MATCH";

  if (noMatch) {
    resultsSection.classList.add("status-nomatch");
  }

  const chips = [];
  if (result.timestamp) chips.push(`<span class="chip"><strong>Timestamp</strong> ${escapeHtml(result.timestamp)}</span>`);
  if (result.frame !== null && result.frame !== undefined) chips.push(`<span class="chip"><strong>Frame</strong> ${result.frame}</span>`);
  if (result.confidence !== null && result.confidence !== undefined) chips.push(`<span class="chip"><strong>Confidence</strong> ${result.confidence.toFixed(3)}</span>`);

  resultsSection.innerHTML = `
    <span class="badge ${info.cls}">${escapeHtml(info.label)}</span>
    ${hasImage ? `<img class="result-frame" src="${result.image_url}" alt="Extracted video frame" />` : ""}
    ${noMatch ? `<p class="result-message">No confident match was found for "${escapeHtml(query.dialogue)}" in this video.</p>` : ""}
    ${chips.length ? `<div class="result-meta">${chips.join("")}</div>` : ""}
    ${result.text ? `<p class="result-text">"${escapeHtml(result.text)}"</p>` : ""}
    ${result.transcript_source ? `<p class="result-source">Source: ${result.transcript_source === "captions_local_asr" ? "caption-assisted (fast path)" : "full-video ASR"}</p>` : ""}
  `;
  resultsSection.hidden = false;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

async function pollJob(jobId, query) {
  while (true) {
    const res = await fetch(`/api/search/${jobId}`);
    if (!res.ok) {
      renderError("Lost track of this search. Please try again.");
      return;
    }
    const data = await res.json();

    if (data.status === "running") {
      setStage(data.stage || "Working...");
      await new Promise((r) => setTimeout(r, 1200));
      continue;
    }

    progressSection.hidden = true;

    if (data.status === "error") {
      renderError(data.error?.message || "Something went wrong. Please try again.");
      return;
    }

    if (data.status === "done") {
      renderResult(query, data.result);
      return;
    }
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  resetUi();

  const url = urlInput.value;
  const dialogue = dialogueInput.value;
  if (!validate(url, dialogue)) return;

  const useCaptions = document.getElementById("use-captions").checked;
  const asrModel = document.getElementById("asr-model").value;

  submitBtn.disabled = true;
  progressSection.hidden = false;
  setStage("Starting...");

  try {
    const res = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url: url.trim(),
        dialogue: dialogue.trim(),
        use_captions: useCaptions,
        asr_model: asrModel || null,
      }),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      progressSection.hidden = true;
      renderError(body.detail || "Please check your input and try again.");
      return;
    }

    const { job_id } = await res.json();
    await pollJob(job_id, { url: url.trim(), dialogue: dialogue.trim() });
  } catch (err) {
    progressSection.hidden = true;
    renderError("Could not reach the server. Please try again.");
  } finally {
    submitBtn.disabled = false;
  }
});
