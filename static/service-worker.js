// 最小構成：まずは登録できることを優先（オフライン対応は後で強化）
self.addEventListener("install", (event) => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

// 何もキャッシュしない（= ループ要因を排除）
self.addEventListener("fetch", (event) => {
  event.respondWith(fetch(event.request));
});
