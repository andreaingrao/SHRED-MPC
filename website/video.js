// Start the featured silent video unless the visitor requests less motion/data.
const video = document.querySelector(".hero-video video");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
if (video) {
  video.muted = true;
  if (!reducedMotion.matches && !navigator.connection?.saveData) {
    video.play().catch(() => {}); // Native play controls remain available.
  }
  reducedMotion.addEventListener("change", event => {
    if (event.matches) video.pause();
  });
}
