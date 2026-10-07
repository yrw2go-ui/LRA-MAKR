// Image library kept in the browser's IndexedDB: the training images behind
// each LoRA and the images it produced. Blobs are stored directly, so nothing
// depends on Replicate's short-lived file links.

export type LibraryKind = "training" | "output";

export type LibraryImage = {
  id: string;
  jobId: string;
  kind: LibraryKind;
  name: string;
  type: string;
  blob: Blob;
  caption?: string; // training images
  prompt?: string; // outputs
  createdAt: number;
};

const DB_NAME = "lra-makr-library";
const STORE = "images";

let dbPromise: Promise<IDBDatabase> | null = null;

function openDb() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = () => {
        const store = req.result.createObjectStore(STORE, { keyPath: "id" });
        store.createIndex("jobId", "jobId");
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => {
        dbPromise = null;
        reject(req.error);
      };
    });
  }
  return dbPromise;
}

function done(tx: IDBTransaction) {
  return new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error ?? new Error("transaction aborted"));
  });
}

let persistAsked = false;

export async function addLibraryImages(images: LibraryImage[]) {
  if (images.length === 0) return;
  // ask the browser not to evict the library when space runs low
  if (!persistAsked) {
    persistAsked = true;
    navigator.storage?.persist?.().catch(() => {});
  }
  const db = await openDb();
  const tx = db.transaction(STORE, "readwrite");
  const store = tx.objectStore(STORE);
  images.forEach((img) => store.put(img));
  await done(tx);
}

export async function listLibraryImages(jobId?: string) {
  const db = await openDb();
  const tx = db.transaction(STORE, "readonly");
  const store = tx.objectStore(STORE);
  const req = jobId ? store.index("jobId").getAll(jobId) : store.getAll();
  const images = await new Promise<LibraryImage[]>((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return images.sort((a, b) => a.createdAt - b.createdAt);
}

export async function deleteLibraryImage(id: string) {
  const db = await openDb();
  const tx = db.transaction(STORE, "readwrite");
  tx.objectStore(STORE).delete(id);
  await done(tx);
}

export async function deleteJobImages(jobId: string) {
  const images = await listLibraryImages(jobId);
  const db = await openDb();
  const tx = db.transaction(STORE, "readwrite");
  const store = tx.objectStore(STORE);
  images.forEach((img) => store.delete(img.id));
  await done(tx);
}

export async function libraryUsage() {
  const estimate = await navigator.storage?.estimate?.();
  return { used: estimate?.usage ?? 0, quota: estimate?.quota ?? 0 };
}
