"use client";

import styles from "./library-view.module.scss";
import { useEffect, useMemo, useState } from "react";
import { IconButton, showToast } from "@/app/components/ui";
import { CloseIcon, DownloadIcon } from "@/app/components/icons";
import {
  LibraryImage,
  LibraryKind,
  deleteLibraryImage,
  libraryUsage,
  listLibraryImages,
} from "@/app/lib/library";
import { LoraJob, useLoraStore } from "@/app/store/lora";
import { ZipEntry, createZip } from "@/app/lib/zip";

function formatBytes(n: number) {
  if (n < 1024 * 1024) return `${Math.round(n / 1024)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(1)} GB`;
}

function slug(name: string) {
  return (
    name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "") || "lora"
  );
}

function extOf(img: LibraryImage) {
  const fromName = img.name.match(/\.(\w{2,5})$/)?.[1];
  return fromName ?? img.type.split("/")[1]?.replace("jpeg", "jpg") ?? "bin";
}

async function downloadAll(job: LoraJob, images: LibraryImage[]) {
  const encoder = new TextEncoder();
  const entries: ZipEntry[] = [];
  const prompts: string[] = [];
  let t = 0;
  let o = 0;
  for (const img of images) {
    const data = new Uint8Array(await img.blob.arrayBuffer());
    if (img.kind === "training") {
      const base = `training/${String(++t).padStart(3, "0")}`;
      entries.push({ name: `${base}.${extOf(img)}`, data });
      if (img.caption) {
        entries.push({ name: `${base}.txt`, data: encoder.encode(img.caption) });
      }
    } else {
      const file = `outputs/${String(++o).padStart(3, "0")}.${extOf(img)}`;
      entries.push({ name: file, data });
      prompts.push(`${file}\t${img.prompt ?? ""}`);
    }
  }
  if (prompts.length) {
    entries.push({
      name: "outputs/prompts.txt",
      data: encoder.encode(prompts.join("\n") + "\n"),
    });
  }
  const url = URL.createObjectURL(createZip(entries));
  const a = document.createElement("a");
  a.href = url;
  a.download = `${slug(job.name)}_library.zip`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

function Tile(props: { image: LibraryImage; onDelete: () => void }) {
  const { image } = props;
  const url = useMemo(() => URL.createObjectURL(image.blob), [image.blob]);
  useEffect(() => () => URL.revokeObjectURL(url), [url]);
  const text = image.kind === "training" ? image.caption : image.prompt;

  return (
    <div className={styles["tile"]}>
      <a href={url} target="_blank" rel="noreferrer">
        {image.type.startsWith("video/") ? (
          <video src={url} muted loop autoPlay playsInline />
        ) : (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={url} alt={text ?? image.name} />
        )}
      </a>
      <button
        className={styles["tile-delete"]}
        title="Delete from library"
        onClick={props.onDelete}
      >
        <CloseIcon />
      </button>
      {text && (
        <div className={styles["tile-text"]} title={text}>
          {text}
        </div>
      )}
    </div>
  );
}

export function LibraryView(props: {
  onReuse: (job: LoraJob, training: LibraryImage[]) => void;
}) {
  const store = useLoraStore();
  const [images, setImages] = useState<LibraryImage[] | null>(null);
  const [usage, setUsage] = useState({ used: 0, quota: 0 });
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [kind, setKind] = useState<LibraryKind>("training");

  useEffect(() => {
    let cancelled = false;
    listLibraryImages()
      .then((all) => !cancelled && setImages(all))
      .catch((e) => {
        showToast(`Couldn't open the library: ${e.message}`);
        setImages([]);
      });
    libraryUsage().then((u) => !cancelled && setUsage(u));
    return () => {
      cancelled = true;
    };
  }, [store.libraryVersion]);

  const byJob = useMemo(() => {
    const map = new Map<string, LibraryImage[]>();
    images?.forEach((img) => {
      const list = map.get(img.jobId) ?? [];
      list.push(img);
      map.set(img.jobId, list);
    });
    return map;
  }, [images]);

  const jobs = store.jobs.filter((j) => byJob.has(j.id));
  const selected = jobs.find((j) => j.id === selectedId);

  if (images === null) return <div className={styles["empty"]}>Loading…</div>;

  if (jobs.length === 0) {
    return (
      <div className={styles["empty"]}>
        <h3>Your library is empty</h3>
        <p>
          Training images and generated images are saved here automatically
          when you train a LoRA or make test images.
        </p>
      </div>
    );
  }

  if (!selected) {
    return (
      <div className={styles["library"]}>
        <div className={styles["usage"]}>
          {images.length} images · {formatBytes(usage.used)} used in this
          browser
          {usage.quota ? ` of ${formatBytes(usage.quota)} available` : ""}
        </div>
        <div className={styles["cards"]}>
          {jobs.map((job) => {
            const list = byJob.get(job.id)!;
            const training = list.filter((i) => i.kind === "training");
            const cover = list.find((i) => i.kind === "output") ?? list[0];
            return (
              <button
                key={job.id}
                className={styles["card"]}
                onClick={() => {
                  setSelectedId(job.id);
                  setKind(training.length ? "training" : "output");
                }}
              >
                <CardCover image={cover} />
                <div className={styles["card-title"]}>{job.name}</div>
                <div className={styles["card-meta"]}>
                  {job.baseModel} · {training.length} training ·{" "}
                  {list.length - training.length} outputs
                </div>
              </button>
            );
          })}
        </div>
      </div>
    );
  }

  const list = byJob.get(selected.id)!;
  const training = list.filter((i) => i.kind === "training");
  const outputs = list.filter((i) => i.kind === "output");
  const shown = kind === "training" ? training : outputs;

  async function remove(img: LibraryImage) {
    await deleteLibraryImage(img.id);
    store.bumpLibrary();
  }

  return (
    <div className={styles["library"]}>
      <div className={styles["detail-header"]}>
        <span className={styles["back"]} onClick={() => setSelectedId(null)}>
          ← All LoRAs
        </span>
        <h3>{selected.name}</h3>
        <div className={styles["card-meta"]}>
          {selected.baseModel} · trigger: <code>{selected.triggerWord}</code>
        </div>
      </div>

      <div className={styles["actions"]}>
        <IconButton
          text="Retrain with these images"
          type="primary"
          disabled={training.length === 0}
          onClick={() => props.onReuse(selected, training)}
        />
        <IconButton
          icon={<DownloadIcon />}
          text="Download all (.zip)"
          bordered
          onClick={() =>
            downloadAll(selected, list).catch((e) => showToast(e.message))
          }
        />
      </div>

      <div className={styles["tabs"]}>
        <button
          className={kind === "training" ? styles["active"] : ""}
          onClick={() => setKind("training")}
        >
          Training images ({training.length})
        </button>
        <button
          className={kind === "output" ? styles["active"] : ""}
          onClick={() => setKind("output")}
        >
          Outputs ({outputs.length})
        </button>
      </div>

      {shown.length === 0 ? (
        <div className={styles["empty"]}>
          {kind === "training"
            ? "No training images saved for this LoRA."
            : "No outputs yet. Generate test images from the Studio tab."}
        </div>
      ) : (
        <div className={styles["grid"]}>
          {shown.map((img) => (
            <Tile key={img.id} image={img} onDelete={() => remove(img)} />
          ))}
        </div>
      )}
    </div>
  );
}

function CardCover(props: { image: LibraryImage }) {
  const url = useMemo(
    () => URL.createObjectURL(props.image.blob),
    [props.image.blob],
  );
  useEffect(() => () => URL.revokeObjectURL(url), [url]);
  return props.image.type.startsWith("video/") ? (
    <video className={styles["card-cover"]} src={url} muted />
  ) : (
    // eslint-disable-next-line @next/next/no-img-element
    <img className={styles["card-cover"]} src={url} alt="" />
  );
}
