const CACHE_NAME = "yutai-cross-pwa-v1";
const URLS = [
  "./",
  "app/static/manifest.json",
  "app/static/icon-192.png",
  "app/static/icon-512.png"
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(URLS)));
});

self.addEventListener("fetch", (event) => {
  event.respondWith(
    caches.match(event.request).then((resp) => resp || fetch(event.request))
  );
});
