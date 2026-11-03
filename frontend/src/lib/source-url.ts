/** Resolve only public URLs or canonical local original-evidence endpoints. */
export function sourceUrl(value: string | null | undefined): string | null {
  if (!value) return null;
  try {
    const local =
      /^\/api\/sources\/(?:chunks|documents)\/[A-Za-z0-9-]+(?:\/pdf)?$/.test(
        value
      );
    const url = local
      ? new URL(
          value,
          process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
        )
      : new URL(value);
    return ["https:", "http:"].includes(url.protocol) &&
      !url.username &&
      !url.password
      ? url.href
      : null;
  } catch {
    return null;
  }
}
