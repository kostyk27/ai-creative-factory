(function () {
  "use strict";

  var API_BASE = (typeof window !== "undefined" && window.API_BASE) ? window.API_BASE.replace(/\/?$/, "") : "";

  var STYLES = ["UGC", "Luxury", "Medical", "Influencer", "Studio"];

  var ideaEl = document.getElementById("idea");
  var styleBtns = document.getElementById("styleRow");
  var btnIdeas = document.getElementById("btnIdeas");
  var secIdeas = document.getElementById("secIdeas");
  var ideasOut = document.getElementById("ideasOut");
  var selectedIdeaText = document.getElementById("selectedIdeaText");
  var btnPrompts = document.getElementById("btnPrompts");
  var secPrompts = document.getElementById("secPrompts");
  var promptsOut = document.getElementById("promptsOut");
  var secSelected = document.getElementById("secSelected");
  var selectedText = document.getElementById("selectedText");
  var numEl = document.getElementById("num");
  var numVal = document.getElementById("numVal");
  var btnImages = document.getElementById("btnImages");
  var imagesOut = document.getElementById("imagesOut");
  var modal = document.getElementById("modal");
  var modalImg = document.getElementById("modalImg");
  var modalClose = document.getElementById("modalClose");

  var selectedIdea = "";
  var currentPrompts = [];
  var selectedPrompt = "";
  var activeStyle = null;

  function escapeHtml(s) {
    var div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  function getStyleForApi() {
    return (activeStyle && activeStyle.toLowerCase()) || "ugc";
  }

  function buildStyleButtons() {
    if (!styleBtns) return;
    STYLES.forEach(function (label) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "style-btn";
      btn.textContent = label;
      btn.dataset.style = label;
      btn.addEventListener("click", function () {
        activeStyle = activeStyle === label ? null : label;
        styleBtns.querySelectorAll(".style-btn").forEach(function (b) {
          b.classList.toggle("active", b.dataset.style === activeStyle);
        });
      });
      styleBtns.appendChild(btn);
    });
  }

  function showIdeasLoading() {
    if (ideasOut) ideasOut.innerHTML = "<div class=\"loader\">Generating ideas…</div>";
  }

  function showIdeasError(msg) {
    if (ideasOut) ideasOut.innerHTML = "<p class=\"err\">" + escapeHtml(msg || "Something went wrong.") + "</p>";
  }

  function prettyHttpError(r, data) {
    if (r && r.status === 429) return "Вы исчерпали лимит запросов. Пожалуйста, подождите немного";
    return (data && data.error) || (r && r.status) || "Request failed";
  }

  function renderIdeas(ideas) {
    selectedIdea = "";
    if (selectedIdeaText) {
      selectedIdeaText.textContent = "Select an idea below.";
      selectedIdeaText.classList.remove("hidden");
    }
    if (secPrompts) secPrompts.classList.add("hidden");
    if (!ideasOut) return;
    ideasOut.innerHTML = "";
    var list = document.createElement("div");
    list.className = "prompts-grid";
    ideas.forEach(function (ideaStr, i) {
      var card = document.createElement("div");
      card.className = "prompt-card";
      card.dataset.index = i;
      var textSpan = document.createElement("span");
      textSpan.className = "prompt-text";
      textSpan.appendChild(document.createTextNode(ideaStr));
      var selectBtn = document.createElement("button");
      selectBtn.type = "button";
      selectBtn.className = "btn-select";
      selectBtn.textContent = "Select";
      function selectThis() {
        selectedIdea = ideaStr;
        if (selectedIdeaText) selectedIdeaText.classList.add("hidden");
        list.querySelectorAll(".prompt-card").forEach(function (c) {
          c.classList.remove("selected");
        });
        card.classList.add("selected");
        list.classList.add("collapsed");
        if (btnShowAllIdeas) btnShowAllIdeas.classList.add("is-visible");
        if (secPrompts) secPrompts.classList.remove("hidden");
      }
      selectBtn.addEventListener("click", function (e) {
        e.stopPropagation();
        selectThis();
      });
      card.addEventListener("click", selectThis);
      card.appendChild(textSpan);
      card.appendChild(selectBtn);
      list.appendChild(card);
    });
    var btnShowAllIdeas = document.createElement("button");
    btnShowAllIdeas.type = "button";
    btnShowAllIdeas.className = "btn-show-all";
    btnShowAllIdeas.textContent = "Choose another idea";
    btnShowAllIdeas.addEventListener("click", function () {
      list.classList.remove("collapsed");
      btnShowAllIdeas.classList.remove("is-visible");
      if (selectedIdeaText) {
        selectedIdeaText.textContent = "Select an idea below.";
        selectedIdeaText.classList.remove("hidden");
      }
    });
    ideasOut.appendChild(list);
    ideasOut.appendChild(btnShowAllIdeas);
  }

  btnIdeas.addEventListener("click", async function () {
    var idea = (ideaEl && ideaEl.value.trim()) || "";
    if (!idea) {
      showIdeasError("Enter a creative idea.");
      return;
    }
    btnIdeas.disabled = true;
    if (secIdeas) secIdeas.classList.remove("hidden");
    showIdeasLoading();
    try {
      var r = await fetch(API_BASE + "/generate-ideas", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ idea: idea })
      });
      var data = await r.json().catch(function () { return {}; });
      if (!r.ok) {
        showIdeasError(prettyHttpError(r, data));
        btnIdeas.disabled = false;
        return;
      }
      renderIdeas(data.ideas || []);
    } catch (e) {
      showIdeasError(e.message || "Request failed");
    }
    btnIdeas.disabled = false;
  });

  function showPromptsLoading() {
    promptsOut.innerHTML = "<div class=\"loader\">Generating prompts…</div>";
  }

  function showPromptsError(msg) {
    promptsOut.innerHTML = "<p class=\"err\">" + escapeHtml(msg || "Something went wrong.") + "</p>";
  }

  function renderPrompts(prompts) {
    currentPrompts = prompts;
    selectedPrompt = "";
    selectedText.textContent = "Select a prompt above.";
    if (secSelected) secSelected.classList.add("hidden");

    promptsOut.innerHTML = "";
    var list = document.createElement("div");
    list.className = "prompts-grid";
    prompts.forEach(function (p, i) {
      var promptText = typeof p === "string" ? p : (p && p.prompt) || "";
      var score = typeof p === "object" && p != null && typeof p.score === "number" ? p.score : null;

      var card = document.createElement("div");
      card.className = "prompt-card";
      card.dataset.index = i;

      var textSpan = document.createElement("span");
      textSpan.className = "prompt-text";
      if (score != null) {
        var badge = document.createElement("span");
        badge.className = "prompt-score";
        badge.textContent = "[" + (Math.round(score * 10) / 10) + "] ";
        textSpan.appendChild(badge);
      }
      textSpan.appendChild(document.createTextNode(promptText));

      var selectBtn = document.createElement("button");
      selectBtn.type = "button";
      selectBtn.className = "btn-select";
      selectBtn.textContent = "Select";

      function selectThis() {
        selectedPrompt = promptText;
        selectedText.textContent = promptText;
        list.querySelectorAll(".prompt-card").forEach(function (c) {
          c.classList.remove("selected");
        });
        card.classList.add("selected");
        list.classList.add("collapsed");
        if (btnShowAll) btnShowAll.classList.add("is-visible");
        if (secSelected) secSelected.classList.remove("hidden");
      }

      selectBtn.addEventListener("click", function (e) {
        e.stopPropagation();
        selectThis();
      });
      card.addEventListener("click", function () {
        selectThis();
      });

      card.appendChild(textSpan);
      card.appendChild(selectBtn);
      list.appendChild(card);
    });

    var btnShowAll = document.createElement("button");
    btnShowAll.type = "button";
    btnShowAll.className = "btn-show-all";
    btnShowAll.textContent = "Choose another prompt";
    btnShowAll.addEventListener("click", function () {
      list.classList.remove("collapsed");
      btnShowAll.classList.remove("is-visible");
    });

    promptsOut.appendChild(list);
    promptsOut.appendChild(btnShowAll);
  }

  btnPrompts.addEventListener("click", async function () {
    if (!selectedIdea) {
      showPromptsError("Select a creative idea first.");
      return;
    }
    btnPrompts.disabled = true;
    showPromptsLoading();
    try {
      var r = await fetch(API_BASE + "/generate-prompts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ idea: selectedIdea, style: getStyleForApi() })
      });
      var data = await r.json().catch(function () { return {}; });
      if (!r.ok) {
        showPromptsError(prettyHttpError(r, data));
        btnPrompts.disabled = false;
        return;
      }
      var prompts = data.prompts || [];
      renderPrompts(prompts);
      if (secSelected) secSelected.classList.remove("hidden");
    } catch (e) {
      showPromptsError(e.message || "Request failed");
    }
    btnPrompts.disabled = false;
  });

  function showImagesLoading() {
    imagesOut.innerHTML = "<div class=\"loader\">Generating images…</div>";
  }

  function showImagesError(msg) {
    imagesOut.innerHTML = "<p class=\"err\">" + escapeHtml(msg || "Something went wrong.") + "</p>";
  }

  function openFullscreen(src) {
    if (modalImg) modalImg.src = src;
    if (modal) modal.classList.add("open");
  }

  function closeModal() {
    if (modal) modal.classList.remove("open");
  }

  if (modalClose) modalClose.addEventListener("click", closeModal);
  if (modal) modal.addEventListener("click", function (e) {
    if (e.target === modal) closeModal();
  });

  function showImages(paths) {
    if (!paths.length) {
      imagesOut.innerHTML = "<p class=\"err\">No images returned.</p>";
      return;
    }
    var grid = document.createElement("div");
    grid.className = "gallery";
    paths.forEach(function (path) {
      var src = path.replace(/"/g, "&quot;");
      var item = document.createElement("div");
      item.className = "gallery-item";
      var img = document.createElement("img");
      img.src = src;
      img.alt = "";
      var overlay = document.createElement("div");
      overlay.className = "overlay";
      var downloadBtn = document.createElement("a");
      downloadBtn.href = src;
      downloadBtn.download = path.split("/").pop() || "image.png";
      downloadBtn.className = "overlay-btn";
      downloadBtn.textContent = "Download";
      var fullscreenBtn = document.createElement("button");
      fullscreenBtn.type = "button";
      fullscreenBtn.className = "overlay-btn";
      fullscreenBtn.textContent = "Fullscreen";
      fullscreenBtn.addEventListener("click", function () {
        openFullscreen(src);
      });
      overlay.appendChild(downloadBtn);
      overlay.appendChild(fullscreenBtn);
      item.appendChild(img);
      item.appendChild(overlay);
      grid.appendChild(item);
    });
    imagesOut.innerHTML = "";
    imagesOut.appendChild(grid);
  }

  if (numEl && numVal) {
    numEl.addEventListener("input", function () {
      numVal.textContent = numEl.value;
    });
  }

  function pollResult(jobId) {
    return fetch(API_BASE + "/generate-images/result/" + encodeURIComponent(jobId))
      .then(function (r) { return r.json().then(function (data) { return { ok: r.ok, status: r.status, data: data }; }); })
      .then(function (res) {
        var data = res.data;
        if (!res.ok) throw new Error(prettyHttpError({ status: res.status }, data));
        if (data.status === "completed") return data.images;
        if (data.status === "failed") throw new Error(data.error || "Generation failed");
        return null;
      });
  }

  var isGeneratingImages = false;
  btnImages.addEventListener("click", async function () {
    if (!selectedPrompt) {
      showImagesError("Select a prompt first.");
      return;
    }
    if (isGeneratingImages) return;
    var count = Math.min(4, Math.max(1, parseInt(numEl.value, 10) || 4));
    isGeneratingImages = true;
    btnImages.disabled = true;
    showImagesLoading();
    try {
      var r = await fetch(API_BASE + "/generate-images", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: selectedPrompt, count: count })
      });
      var data = await r.json().catch(function () { return {}; });
      if (!r.ok) {
        showImagesError(prettyHttpError(r, data));
        return;
      }
      if (r.status === 202 && data.job_id) {
        var pollInterval = 2000;
        while (true) {
          await new Promise(function (resolve) { setTimeout(resolve, pollInterval); });
          var result = await pollResult(data.job_id);
          if (result !== null) {
            showImages(result);
            break;
          }
        }
      } else {
        showImages(data.images || []);
      }
    } catch (e) {
      showImagesError(e.message || "Request failed");
    } finally {
      isGeneratingImages = false;
      btnImages.disabled = false;
    }
  });

  buildStyleButtons();
})();
