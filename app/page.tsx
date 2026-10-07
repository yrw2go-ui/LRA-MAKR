"use client";

import dynamic from "next/dynamic";

// the studio keeps its state in localStorage, so render it in the browser only
const LoraStudio = dynamic(
  async () => (await import("./components/lora-studio")).LoraStudio,
  { ssr: false },
);

export default function Page() {
  return (
    <main className="app">
      <LoraStudio />
    </main>
  );
}
