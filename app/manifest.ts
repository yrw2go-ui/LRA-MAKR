import type { MetadataRoute } from "next";

// lets phones install LRA-MAKR from "Add to Home Screen" as a full-screen app
export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "LRA-MAKR",
    short_name: "LRA-MAKR",
    description: "Train image and video LoRAs in a few clicks.",
    start_url: "/",
    display: "standalone",
    background_color: "#ffffff",
    theme_color: "#1d93ab",
    icons: [
      { src: "/icon-192.png", sizes: "192x192", type: "image/png" },
      { src: "/icon-512.png", sizes: "512x512", type: "image/png" },
      {
        src: "/icon-maskable-512.png",
        sizes: "512x512",
        type: "image/png",
        purpose: "maskable",
      },
    ],
  };
}
