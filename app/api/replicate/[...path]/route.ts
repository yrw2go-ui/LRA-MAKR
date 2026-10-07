import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";

const REPLICATE_BASE_URL = "https://api.replicate.com";

// only the endpoints LRA-MAKR needs are forwarded, so the server's token
// can't be used to reach the rest of the Replicate API
const ALLOWED_PATHS = [
  /^v1\/account$/,
  /^v1\/files$/,
  /^v1\/models$/,
  /^v1\/models\/[\w.-]+\/[\w.-]+$/,
  /^v1\/models\/[\w.-]+\/[\w.-]+\/versions\/\w+\/trainings$/,
  /^v1\/trainings\/\w+(\/cancel)?$/,
  /^v1\/predictions(\/\w+)?$/,
];

function error(status: number, message: string) {
  return NextResponse.json({ error: true, message }, { status });
}

// Picks the Replicate token for this request: the user's own token if they
// sent one, otherwise the server's REPLICATE_API_TOKEN (guarded by
// ACCESS_CODE when that's set).
function resolveToken(req: NextRequest): string | NextResponse {
  const userToken = (req.headers.get("Authorization") ?? "")
    .replace(/^Bearer\s+/i, "")
    .trim();
  if (userToken) return userToken;

  const accessCode = process.env.ACCESS_CODE;
  if (accessCode && req.headers.get("X-Access-Code") !== accessCode) {
    return error(401, "Wrong or missing access code");
  }
  const serverToken = process.env.REPLICATE_API_TOKEN;
  if (!serverToken) {
    return error(
      401,
      "No Replicate token: add yours in the app, or set REPLICATE_API_TOKEN on the server",
    );
  }
  return serverToken;
}

async function handle(
  req: NextRequest,
  { params }: { params: { path: string[] } },
) {
  const path = params.path.join("/");
  if (path === "download") return download(req);
  if (!ALLOWED_PATHS.some((re) => re.test(path))) {
    return error(403, `path not allowed: ${path}`);
  }

  const token = resolveToken(req);
  if (token instanceof NextResponse) return token;

  const headers: Record<string, string> = { Authorization: `Bearer ${token}` };
  const contentType = req.headers.get("Content-Type");
  if (contentType) headers["Content-Type"] = contentType;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 10 * 60 * 1000);
  try {
    const res = await fetch(`${REPLICATE_BASE_URL}/${path}${req.nextUrl.search}`, {
      headers,
      method: req.method,
      body: req.method === "GET" ? undefined : req.body,
      redirect: "manual",
      // @ts-ignore
      duplex: "half",
      signal: controller.signal,
    });
    const out = new Headers(res.headers);
    out.delete("www-authenticate");
    out.set("X-Accel-Buffering", "no");
    return new Response(res.body, {
      status: res.status,
      statusText: res.statusText,
      headers: out,
    });
  } finally {
    clearTimeout(timeoutId);
  }
}

// Streams a trained model's output file from Replicate's file host, for
// browsers that can't fetch it cross-origin. Only replicate.delivery URLs
// are allowed, so this can't be used as an open proxy.
async function download(req: NextRequest) {
  const token = resolveToken(req);
  if (token instanceof NextResponse) return token;

  let url: URL;
  try {
    url = new URL(req.nextUrl.searchParams.get("url") ?? "");
  } catch {
    return error(400, "bad url");
  }
  const host = url.hostname;
  if (
    url.protocol !== "https:" ||
    !(host === "replicate.delivery" || host.endsWith(".replicate.delivery"))
  ) {
    return error(403, "only replicate.delivery files can be downloaded");
  }

  const res = await fetch(url, { redirect: "follow" });
  const headers = new Headers();
  for (const h of ["content-type", "content-length"]) {
    const v = res.headers.get(h);
    if (v) headers.set(h, v);
  }
  return new Response(res.body, { status: res.status, headers });
}

export const GET = handle;
export const POST = handle;
