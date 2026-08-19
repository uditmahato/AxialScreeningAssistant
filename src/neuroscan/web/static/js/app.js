/* Axial Screening Assistant - progressive enhancement only.
 *
 * Every feature here degrades gracefully: the upload form submits normally
 * without JavaScript, the image tabs are CSS radio buttons, the navigation
 * simply wraps instead of collapsing, and the question form falls back to a
 * standard POST. The deployment target includes old Android phones on
 * unreliable connections, so nothing essential is allowed to depend on
 * scripting.
 */
(function () {
  "use strict";

  /* Signals to the stylesheet that enhancements are active (collapsible
   * mobile navigation, gated analyse button). */
  document.documentElement.classList.add("js");

  /* ------------------------------------------------------------ navigation */

  var navToggle = document.getElementById("nav-toggle");
  var siteHeader = document.getElementById("site-header");
  if (navToggle && siteHeader) {
    navToggle.addEventListener("click", function () {
      var open = siteHeader.classList.toggle("nav-open");
      navToggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  /* ---------------------------------------------------------------- upload */

  var dropzone = document.getElementById("dropzone");
  var input = document.getElementById("scan-input");

  if (dropzone && input) {
    var prompt = document.getElementById("dropzone-prompt");
    var preview = document.getElementById("dropzone-preview");
    var previewImage = document.getElementById("preview-image");
    var previewName = document.getElementById("preview-name");
    var previewSize = document.getElementById("preview-size");
    var changeBtn = document.getElementById("change-image");
    var submitBtn = document.getElementById("submit-btn");
    var uploadError = document.getElementById("upload-error");
    var badTypeMessage = (input.form && input.form.getAttribute("data-str-bad-type")) ||
      "Unsupported file type. Please upload a JPG or PNG image.";

    /* The analyse action becomes primary only once a scan is selected.
     * Without JS the button stays enabled, so the form still works. */
    if (submitBtn && !submitBtn.disabled) submitBtn.disabled = true;

    /* The file picker's accept filter does not apply to drag-and-drop, so a
     * TIFF, DICOM or corrupt file can land here with an image/* type the
     * browser cannot actually decode. Checking the extension and then only
     * revealing the preview once the thumbnail has decoded means the user
     * either sees their scan or a clear rejection, never a broken image
     * next to an enabled Analyse button. */
    var ALLOWED_NAME = /\.(jpe?g|png)$/i;

    function formatSize(bytes) {
      if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + " MB";
      return Math.max(1, Math.round(bytes / 1024)) + " KB";
    }

    function clearInvalid() {
      dropzone.classList.remove("is-invalid");
      input.removeAttribute("aria-invalid");
      if (uploadError) { uploadError.hidden = true; uploadError.textContent = ""; }
    }

    function showInvalid(message) {
      input.value = "";
      if (preview) preview.hidden = true;
      if (prompt) prompt.hidden = false;
      if (changeBtn) changeBtn.hidden = true;
      if (submitBtn) submitBtn.disabled = true;
      dropzone.classList.add("is-invalid");
      input.setAttribute("aria-invalid", "true");
      if (uploadError) {
        /* Unhide BEFORE writing the text. role="alert" only announces a
         * mutation to a node that is already in the accessibility tree;
         * writing first and unhiding second announced nothing at all. */
        uploadError.hidden = false;
        uploadError.textContent = message || badTypeMessage;
      }
    }

    function showPreview(file) {
      clearInvalid();
      if (!file || !ALLOWED_NAME.test(file.name)) {
        showInvalid();
        return;
      }
      var maxBytes = parseInt(input.getAttribute("data-max-bytes") || "0", 10);
      if (maxBytes > 0 && file.size > maxBytes) {
        showInvalid(input.getAttribute("data-str-too-large") || undefined);
        return;
      }
      var reader = new FileReader();
      /* Without this the read simply never completes on a revoked SD card or
       * a denied storage permission: no preview, no error, Analyse stuck
       * disabled, and nothing on screen explaining why. */
      reader.onerror = function () { showInvalid(); };
      reader.onabort = function () { showInvalid(); };
      reader.onload = function (e) {
        previewImage.onload = function () {
          previewName.textContent = file.name;
          if (previewSize) previewSize.textContent = formatSize(file.size);
          if (prompt) prompt.hidden = true;
          if (preview) preview.hidden = false;
          if (changeBtn) changeBtn.hidden = false;
          /* The server's decision outranks the client's: with no model
           * loaded, a valid preview must not re-enable Analyse. */
          if (submitBtn && !submitBtn.hasAttribute("data-server-disabled")) {
            submitBtn.disabled = false;
          }
        };
        previewImage.onerror = showInvalid;
        previewImage.src = e.target.result;
      };
      reader.readAsDataURL(file);
    }

    input.addEventListener("change", function () {
      if (input.files && input.files[0]) showPreview(input.files[0]);
    });

    if (changeBtn) {
      changeBtn.addEventListener("click", function () {
        input.click();
      });
    }

    ["dragenter", "dragover"].forEach(function (name) {
      dropzone.addEventListener(name, function (e) {
        e.preventDefault();
        dropzone.classList.add("is-dragging");
      });
    });

    ["dragleave", "drop"].forEach(function (name) {
      dropzone.addEventListener(name, function (e) {
        e.preventDefault();
        dropzone.classList.remove("is-dragging");
      });
    });

    dropzone.addEventListener("drop", function (e) {
      var files = e.dataTransfer && e.dataTransfer.files;
      if (files && files.length) {
        input.files = files;
        showPreview(files[0]);
      }
    });

  }

  /* Disable the submit button during analysis and walk through the honest
   * stages. Inference plus advisory generation can take several seconds, and
   * without this users double-submit and queue a second job. */
  var uploadForm = document.getElementById("upload-form");
  var stageTimer = null;
  if (uploadForm) {
    uploadForm.addEventListener("submit", function () {
      var btn = document.getElementById("submit-btn");
      var note = document.getElementById("submit-note");
      if (btn) btn.disabled = true;
      if (note) {
        note.hidden = false;
        var stages = (note.getAttribute("data-stages") || "").split("|").filter(Boolean);
        var base = note.textContent;
        var index = 0;
        if (stages.length) {
          note.textContent = stages[0] + "... " + base;
          stageTimer = setInterval(function () {
            index = Math.min(index + 1, stages.length - 1);
            note.textContent = stages[index] + "... " + base;
          }, 2500);
        }
      }
    });

    /* Coming back from the result page restores this exact DOM from the
     * bfcache, complete with the disabled button and the running ticker, so
     * the next patient's scan could not be submitted without a full reload. */
    window.addEventListener("pageshow", function (event) {
      if (!event.persisted) return;
      if (stageTimer) { clearInterval(stageTimer); stageTimer = null; }
      var btn = document.getElementById("submit-btn");
      var note = document.getElementById("submit-note");
      if (note) note.hidden = true;
      if (btn && !btn.hasAttribute("data-server-disabled")) btn.disabled = false;
    });
  }

  /* Print needs scripting, so the button only appears when it can work. */
  var printBtn = document.getElementById("print-btn");
  if (printBtn) {
    printBtn.hidden = false;
    printBtn.addEventListener("click", function () {
      window.print();
    });
  }

  /* ----------------------------------------------------- previous scans */

  var historyTable = document.getElementById("history-table");
  var historyControls = document.getElementById("history-controls");
  if (historyTable && historyControls) {
    historyControls.hidden = false;

    var searchInput = document.getElementById("history-search");
    var filterButtons = historyControls.querySelectorAll(".filter-chips .chip");
    var rows = historyTable.querySelectorAll("tbody tr");
    var activeFilter = "all";

    function applyFilters() {
      var query = (searchInput ? searchInput.value : "").trim().toUpperCase();
      rows.forEach(function (row) {
        var matchesFilter = activeFilter === "all" ||
          row.getAttribute("data-result") === activeFilter;
        var matchesQuery = !query ||
          (row.getAttribute("data-ref") || "").toUpperCase().indexOf(query) !== -1;
        row.hidden = !(matchesFilter && matchesQuery);
      });
      var visible = 0;
      rows.forEach(function (row) { if (!row.hidden) visible += 1; });
      var none = document.getElementById("history-no-match");
      if (none) none.hidden = visible !== 0;
    }

    if (searchInput) searchInput.addEventListener("input", applyFilters);

    filterButtons.forEach(function (button) {
      button.addEventListener("click", function () {
        activeFilter = button.getAttribute("data-filter") || "all";
        filterButtons.forEach(function (other) {
          other.setAttribute("aria-pressed", other === button ? "true" : "false");
        });
        applyFilters();
      });
    });

    /* Whole row opens the scan; the View link stays for keyboard users. */
    rows.forEach(function (row) {
      row.addEventListener("click", function (e) {
        if (e.target.closest("a")) return;
        var href = row.getAttribute("data-href");
        if (href) window.location.href = href;
      });
    });
  }

  /* -------------------------------------------------- questions and answers */

  var chatForm = document.getElementById("chat-form");
  if (!chatForm) return;

  var chatInput = document.getElementById("chat-input");
  var chatLog = document.getElementById("chat-log");
  var chatSend = document.getElementById("chat-send");
  var chatStatus = document.getElementById("chat-status");
  var endpoint = chatForm.getAttribute("data-endpoint");
  var analysisId = chatForm.getAttribute("data-analysis-id");
  var pageLang = document.documentElement.lang || "en";

  var strings = {
    preparing: chatForm.getAttribute("data-str-preparing") || "Preparing answer...",
    question: chatForm.getAttribute("data-str-question") || "Your question",
    answer: chatForm.getAttribute("data-str-answer") || "Clinical information",
    sources: chatForm.getAttribute("data-str-sources") || "Sources used"
  };

  /* One exchange is one panel: the question on top, the answer below it.
   * textContent, never innerHTML: the answer is model output and must not be
   * able to inject markup into the page. */
  function makeBlock(cssClass, label, text) {
    var block = document.createElement("div");
    block.className = cssClass;
    var eyebrow = document.createElement("span");
    eyebrow.className = "eyebrow";
    eyebrow.textContent = label;
    block.appendChild(eyebrow);
    var para = document.createElement("p");
    para.textContent = text;
    block.appendChild(para);
    return block;
  }

  /* One request at a time. Concurrent POSTs append to the session transcript
   * in arrival order, and that transcript is embedded in the PDF report - two
   * rapid chip taps could attach answers to the wrong questions. */
  var chatBusy = false;
  var chipButtons = document.querySelectorAll("#chat-suggestions .chip");

  function setBusy(busy) {
    chatBusy = busy;
    if (chatSend) chatSend.disabled = busy;
    if (chatInput) chatInput.disabled = busy;
    chipButtons.forEach(function (chip) { chip.disabled = busy; });
    if (chatStatus) {
      chatStatus.hidden = !busy;
      chatStatus.textContent = busy ? strings.preparing : "";
    }
  }

  function ask(question) {
    if (!question || chatBusy) return;

    var qa = document.createElement("div");
    qa.className = "qa";
    qa.appendChild(makeBlock("qa-q", strings.question, question));
    var answerBlock = makeBlock("qa-a is-pending", strings.answer, strings.preparing);
    qa.appendChild(answerBlock);
    chatLog.appendChild(qa);
    var calmMotion = window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    qa.scrollIntoView({ behavior: calmMotion ? "auto" : "smooth", block: "nearest" });

    chatInput.value = "";
    setBusy(true);

    var answerPara = answerBlock.querySelector("p");

    var tokenMeta = document.querySelector('meta[name="csrf-token"]');
    var headers = { "Content-Type": "application/json" };
    if (tokenMeta) headers["X-CSRFToken"] = tokenMeta.getAttribute("content");

    fetch(endpoint, {
      method: "POST",
      headers: headers,
      body: JSON.stringify({
        question: question,
        analysis_id: analysisId,
        language: pageLang
      })
    })
      .then(function (response) {
        return response.json().then(function (data) {
          return { ok: response.ok, data: data };
        });
      })
      .then(function (result) {
        answerBlock.classList.remove("is-pending");
        if (!result.ok) {
          qa.classList.add("qa-error");
          answerPara.textContent = result.data && result.data.detail
            ? result.data.detail
            : (pageLang === "ne"
                ? "उत्तर दिन सकिएन। कृपया फेरि प्रयास गर्नुहोस्।"
                : "Could not answer that. Please try again.");
          return;
        }
        answerPara.textContent = result.data.answer;
        if (result.data.refused) qa.classList.add("qa-refused");

        var citations = result.data.citations;
        if (citations && citations.length) {
          var details = document.createElement("details");
          details.className = "chat-citations";
          var summary = document.createElement("summary");
          summary.textContent = strings.sources + " (" + citations.length + ")";
          details.appendChild(summary);
          var list = document.createElement("ul");
          citations.forEach(function (c) {
            var li = document.createElement("li");
            li.textContent = c;
            list.appendChild(li);
          });
          details.appendChild(list);
          answerBlock.appendChild(details);
        }
      })
      .catch(function () {
        answerBlock.classList.remove("is-pending");
        qa.classList.add("qa-error");
        answerPara.textContent = pageLang === "ne"
          ? "जडानमा समस्या भयो।"
          : "Connection problem. Please check and try again.";
      })
      .finally(function () {
        setBusy(false);
        if (chatInput) chatInput.focus();
      });
  }

  chatForm.addEventListener("submit", function (e) {
    e.preventDefault();
    ask(chatInput.value.trim());
  });

  var suggestions = document.getElementById("chat-suggestions");
  if (suggestions) {
    suggestions.addEventListener("click", function (e) {
      if (e.target && e.target.classList.contains("chip")) {
        ask(e.target.textContent.trim());
      }
    });
  }
})();
