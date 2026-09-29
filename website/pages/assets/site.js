/* pymodelvis website: copy buttons, style tabs, lightbox, play movies when visible. */
(function () {
  function init() {
    // copy buttons
    document.querySelectorAll(".nf-copy[data-copy]").forEach(function (btn) {
      if (btn.dataset.nfReady) return;
      btn.dataset.nfReady = "1";
      btn.addEventListener("click", function () {
        var el = document.querySelector(btn.getAttribute("data-copy"));
        if (!el || !navigator.clipboard) return;
        navigator.clipboard.writeText(el.innerText.trim()).then(function () {
          var t = btn.textContent;
          btn.textContent = "copied";
          btn.classList.add("is-done");
          setTimeout(function () { btn.textContent = t; btn.classList.remove("is-done"); }, 1400);
        });
      });
    });

    // style tabs
    document.querySelectorAll("[data-nf-tabs]").forEach(function (box) {
      var tabs = box.querySelectorAll("[role=tab]");
      var panels = box.querySelectorAll("[role=tabpanel]");
      tabs.forEach(function (tab, i) {
        tab.addEventListener("click", function () {
          tabs.forEach(function (t, j) { t.setAttribute("aria-selected", i === j ? "true" : "false"); });
          panels.forEach(function (p, j) { p.hidden = i !== j; });
        });
      });
    });

    // lightbox for landing-page figures and for images in docs pages
    function open(src, caption) {
      var box = document.createElement("div");
      box.className = "nf-lightbox";
      box.setAttribute("role", "dialog");
      box.innerHTML = '<button aria-label="Close">×</button><img alt=""><p></p>';
      box.querySelector("img").src = src;
      box.querySelector("p").textContent = caption || "";
      function close() { box.remove(); document.removeEventListener("keydown", onKey); }
      function onKey(e) { if (e.key === "Escape") close(); }
      box.addEventListener("click", close);
      document.addEventListener("keydown", onKey);
      document.body.appendChild(box);
    }
    document.querySelectorAll("a.nf-zoom").forEach(function (a) {
      a.addEventListener("click", function (e) {
        if (e.metaKey || e.ctrlKey) return;
        e.preventDefault();
        open(a.getAttribute("href"), a.getAttribute("data-caption"));
      });
    });
    document.querySelectorAll(".md-content .md-typeset img").forEach(function (img) {
      if (img.closest("a") || img.width < 240) return;
      img.addEventListener("click", function () { open(img.currentSrc || img.src, img.alt); });
    });

    // movies play only while on screen (saves bandwidth and CPU)
    var vids = document.querySelectorAll(".nf-home video");
    if ("IntersectionObserver" in window && vids.length) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          var v = en.target;
          if (en.isIntersecting) {
            if (v.preload === "none") { v.preload = "auto"; v.load(); }
            var p = v.play(); if (p && p.catch) p.catch(function () {});
          } else { v.pause(); }
        });
      }, { threshold: 0.35 });
      vids.forEach(function (v) { io.observe(v); });
    }
  }
  if (window.document$ && window.document$.subscribe) window.document$.subscribe(init);
  else if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
})();
