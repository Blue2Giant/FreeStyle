const labels = {
  cref: "Content",
  sref: "Style",
  ours: "Output",
};

const sectionContainers = {
  datasetSref: document.getElementById("dataset-sref"),
  datasetDual: document.getElementById("dataset-dual"),
  resultsSref: document.getElementById("results-sref"),
  resultsDual: document.getElementById("results-dual"),
};

const lightbox = document.getElementById("lightbox");
const lightboxStage = document.getElementById("lightbox-stage");
const lightboxFrame = document.getElementById("lightbox-frame");
const lightboxImage = document.getElementById("lightbox-image");
const lightboxClose = document.getElementById("lightbox-close");

function createMediaButton({ src, alt, kind }) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "media-button";
  button.dataset.src = src;
  button.dataset.alt = alt;
  if (kind) {
    button.dataset.kind = kind;
  }

  const image = document.createElement("img");
  image.src = src;
  image.alt = alt;
  image.loading = "lazy";
  image.decoding = "async";

  button.appendChild(image);
  return button;
}

function createTriptych(sample) {
  const wrapper = document.createElement("div");
  wrapper.className = "triptych";

  ["cref", "sref", "ours"].forEach((kind) => {
    const panel = document.createElement("div");
    panel.className = "triptych-panel";

    const imageButton = createMediaButton({
      src: sample.images[kind],
      alt: labels[kind],
      kind,
    });
    panel.appendChild(imageButton);

    const label = document.createElement("span");
    label.className = "triptych-label";
    label.textContent = labels[kind];
    panel.appendChild(label);

    wrapper.appendChild(panel);
  });

  return wrapper;
}

function syncTripletAspectFromOutput(card) {
  const outputImage = card.querySelector('.media-button[data-kind="ours"] img');
  if (!outputImage) return;

  const applyAspect = () => {
    if (!outputImage.naturalWidth || !outputImage.naturalHeight) return;
    const ratio = `${outputImage.naturalWidth} / ${outputImage.naturalHeight}`;
    card.style.setProperty("--triplet-ratio", ratio);
  };

  if (outputImage.complete && outputImage.naturalWidth) {
    applyAspect();
  } else {
    outputImage.addEventListener("load", applyAspect, { once: true });
  }
}

function renderTripletCard(sample) {
  const card = document.createElement("article");
  card.className = "triplet-card";

  const triptych = createTriptych(sample);
  card.append(triptych);
  syncTripletAspectFromOutput(card);

  if (sample.prompt) {
    const copy = document.createElement("div");
    copy.className = "triplet-copy";

    const prompt = document.createElement("p");
    prompt.className = "prompt-chip";
    prompt.textContent = sample.prompt;

    copy.append(prompt);
    card.append(copy);
  }

  return card;
}

// --- Lightbox: image-only with zoom & pan ---

const lightboxState = {
  scale: 1,
  fitScale: 1,
  tx: 0,
  ty: 0,
  natW: 0,
  natH: 0,
  panning: false,
  panStartX: 0,
  panStartY: 0,
  panOriginTx: 0,
  panOriginTy: 0,
};

function applyTransform() {
  lightboxFrame.style.transform = `translate(${lightboxState.tx}px, ${lightboxState.ty}px) scale(${lightboxState.scale})`;
}

function computeFit() {
  const stageRect = lightboxStage.getBoundingClientRect();
  const { natW, natH } = lightboxState;
  if (!natW || !natH || !stageRect.width || !stageRect.height) {
    return 1;
  }
  return Math.min(stageRect.width / natW, stageRect.height / natH, 1);
}

function centerAtFit() {
  lightboxState.tx = 0;
  lightboxState.ty = 0;
}

function setBaseSize() {
  lightboxFrame.style.width = `${lightboxState.natW}px`;
  lightboxFrame.style.height = `${lightboxState.natH}px`;
}

function resetLightboxView() {
  lightboxState.fitScale = computeFit();
  lightboxState.scale = lightboxState.fitScale;
  centerAtFit();
  applyTransform();
  updateZoomCursor();
}

function clampPan() {
  const stageRect = lightboxStage.getBoundingClientRect();
  const { natW, natH, scale } = lightboxState;
  const imgW = natW * scale;
  const imgH = natH * scale;
  const maxX = Math.max(0, (imgW - stageRect.width) / 2);
  const maxY = Math.max(0, (imgH - stageRect.height) / 2);

  if (imgW <= stageRect.width) {
    lightboxState.tx = 0;
  } else {
    lightboxState.tx = Math.min(maxX, Math.max(-maxX, lightboxState.tx));
  }
  if (imgH <= stageRect.height) {
    lightboxState.ty = 0;
  } else {
    lightboxState.ty = Math.min(maxY, Math.max(-maxY, lightboxState.ty));
  }
}

function updateZoomCursor() {
  if (lightboxState.scale > lightboxState.fitScale + 1e-3) {
    lightboxStage.classList.add("is-zoomed");
    lightboxStage.style.cursor = "";
  } else {
    lightboxStage.classList.remove("is-zoomed");
    lightboxStage.style.cursor = "zoom-in";
  }
}

function zoomAtPoint(targetScale, clientX, clientY) {
  const minScale = lightboxState.fitScale;
  const maxScale = Math.max(lightboxState.fitScale * 6, 4);
  const next = Math.min(maxScale, Math.max(minScale, targetScale));
  const prev = lightboxState.scale || minScale || 1;
  const stageRect = lightboxStage.getBoundingClientRect();
  const pointX =
    typeof clientX === "number" ? clientX - stageRect.left - stageRect.width / 2 : 0;
  const pointY =
    typeof clientY === "number" ? clientY - stageRect.top - stageRect.height / 2 : 0;

  if (prev > 0 && next !== prev) {
    const ratio = next / prev;
    lightboxState.tx = pointX - (pointX - lightboxState.tx) * ratio;
    lightboxState.ty = pointY - (pointY - lightboxState.ty) * ratio;
  }

  lightboxState.scale = next;
  if (Math.abs(next - minScale) <= 1e-3) {
    centerAtFit();
  }

  clampPan();
  applyTransform();
  updateZoomCursor();
}

function openLightbox({ src, alt }) {
  lightboxImage.alt = alt || "";
  lightbox.classList.add("is-open");
  lightbox.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
  lightboxFrame.style.transform = "translate(0px, 0px) scale(1)";
  lightboxFrame.style.transformOrigin = "center center";

  const finishLoad = () => {
    lightboxState.natW = lightboxImage.naturalWidth || lightboxImage.width;
    lightboxState.natH = lightboxImage.naturalHeight || lightboxImage.height;
    setBaseSize();
    resetLightboxView();
  };

  if (lightboxImage.src !== src) {
    lightboxFrame.style.transform = "translate(0px, 0px) scale(1)";
    lightboxFrame.style.transformOrigin = "center center";
    lightboxImage.removeAttribute("style");
    lightboxImage.src = src;
    if (lightboxImage.complete && lightboxImage.naturalWidth) {
      finishLoad();
    } else {
      lightboxImage.addEventListener("load", finishLoad, { once: true });
    }
  } else {
    finishLoad();
  }
}

function closeLightbox() {
  lightbox.classList.remove("is-open");
  lightbox.setAttribute("aria-hidden", "true");
  document.body.style.overflow = "";
  lightboxFrame.style.transform = "translate(0px, 0px) scale(1)";
  lightboxFrame.style.transformOrigin = "center center";
  lightboxImage.removeAttribute("style");
  lightboxImage.src = "";
  lightboxStage.classList.remove("is-zoomed", "is-panning");
}

function bindLightbox() {
  document.addEventListener("click", (event) => {
    const trigger = event.target.closest(".media-button");
    if (trigger) {
      event.preventDefault();
      openLightbox({
        src: trigger.dataset.src,
        alt: trigger.dataset.alt,
      });
      return;
    }

    if (event.target === lightbox) {
      closeLightbox();
    }
  });

  lightboxClose.addEventListener("click", closeLightbox);

  document.addEventListener("keydown", (event) => {
    if (!lightbox.classList.contains("is-open")) return;
    if (event.key === "Escape") {
      closeLightbox();
    } else if (event.key === "0") {
      resetLightboxView();
    } else if (event.key === "+" || event.key === "=") {
      const stageRect = lightboxStage.getBoundingClientRect();
      zoomAtPoint(
        lightboxState.scale * 1.25,
        stageRect.left + stageRect.width / 2,
        stageRect.top + stageRect.height / 2
      );
    } else if (event.key === "-" || event.key === "_") {
      const stageRect = lightboxStage.getBoundingClientRect();
      zoomAtPoint(
        lightboxState.scale / 1.25,
        stageRect.left + stageRect.width / 2,
        stageRect.top + stageRect.height / 2
      );
    }
  });

  lightboxStage.addEventListener(
    "wheel",
    (event) => {
      if (!lightbox.classList.contains("is-open")) return;
      event.preventDefault();
      const factor = Math.exp(-event.deltaY * 0.0015);
      zoomAtPoint(lightboxState.scale * factor, event.clientX, event.clientY);
    },
    { passive: false }
  );

  lightboxStage.addEventListener("click", (event) => {
    if (event.target === lightboxImage) {
      // Toggle between fit and 1:1 native, centered on click point.
      const target =
        lightboxState.scale > lightboxState.fitScale + 1e-3
          ? lightboxState.fitScale
          : Math.max(1, lightboxState.fitScale * 2);
      zoomAtPoint(target, event.clientX, event.clientY);
    }
  });

  lightboxStage.addEventListener("pointerdown", (event) => {
    if (lightboxState.scale <= lightboxState.fitScale + 1e-3) return;
    if (event.target !== lightboxImage) return;
    lightboxState.panning = true;
    lightboxState.panStartX = event.clientX;
    lightboxState.panStartY = event.clientY;
    lightboxState.panOriginTx = lightboxState.tx;
    lightboxState.panOriginTy = lightboxState.ty;
    lightboxStage.classList.add("is-panning");
    lightboxStage.setPointerCapture(event.pointerId);
  });

  lightboxStage.addEventListener("pointermove", (event) => {
    if (!lightboxState.panning) return;
    lightboxState.tx =
      lightboxState.panOriginTx + (event.clientX - lightboxState.panStartX);
    lightboxState.ty =
      lightboxState.panOriginTy + (event.clientY - lightboxState.panStartY);
    clampPan();
    applyTransform();
  });

  const endPan = (event) => {
    if (!lightboxState.panning) return;
    lightboxState.panning = false;
    lightboxStage.classList.remove("is-panning");
    if (event.pointerId !== undefined) {
      try {
        lightboxStage.releasePointerCapture(event.pointerId);
      } catch {}
    }
  };
  lightboxStage.addEventListener("pointerup", endPan);
  lightboxStage.addEventListener("pointercancel", endPan);
  lightboxStage.addEventListener("pointerleave", endPan);

  window.addEventListener("resize", () => {
    if (!lightbox.classList.contains("is-open")) return;
    resetLightboxView();
  });
}

function setupReveal() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );

  document.querySelectorAll(".reveal").forEach((node) => observer.observe(node));
}

function setupPlaceholderLinks() {
  document.querySelectorAll(".hero-links a[href='#']").forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
    });
  });
}

async function main() {
  const response = await fetch("./data/gallery.json");
  const data = await response.json();
  const sampleMap = new Map(data.samples.map((sample) => [sample.id, sample]));

  const renderInto = (container, ids, renderer) => {
    ids.forEach((id) => {
      const sample = sampleMap.get(id);
      if (sample) container.appendChild(renderer(sample));
    });
  };

  renderInto(sectionContainers.datasetSref, data.sections.datasetSref, renderTripletCard);
  renderInto(sectionContainers.datasetDual, data.sections.datasetDual, renderTripletCard);
  renderInto(sectionContainers.resultsSref, data.sections.resultsSref, renderTripletCard);
  renderInto(sectionContainers.resultsDual, data.sections.resultsDual, renderTripletCard);

  setupReveal();
  setupPlaceholderLinks();
  bindLightbox();
}

main().catch((error) => {
  console.error("Failed to initialize gallery", error);
});
