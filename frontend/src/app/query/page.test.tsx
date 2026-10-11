import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import { submitQuery } from "@/lib/api";
import QueryPage from "./page";

vi.mock("@/lib/api", () => ({
  listAllPapers: vi.fn().mockResolvedValue([]),
  submitQuery: vi.fn().mockResolvedValue({
    session_id: "session-1",
    response: "Source excerpts are shown below.",
    sources: [],
    confidence: "low",
    follow_up_suggestions: [],
    evidence: [],
    retrieval_mode: "bm25",
  }),
}));

beforeEach(() => {
  sessionStorage.clear();
  Element.prototype.scrollIntoView = vi.fn();
  vi.mocked(submitQuery).mockClear();
});

it("sends the optional rewrite setting with a question", async () => {
  render(<QueryPage />);
  fireEvent.click(screen.getByText("Search scope and retrieval"));
  fireEvent.click(screen.getByRole("checkbox", { name: /rewrite query/i }));
  const input = screen.getByPlaceholderText("Ask a question about your research...");
  fireEvent.change(input, { target: { value: "What about its latency?" } });
  fireEvent.keyDown(input, { key: "Enter" });

  await waitFor(() =>
    expect(submitQuery).toHaveBeenCalledWith(
      "What about its latency?",
      undefined,
      expect.objectContaining({ rewrite_query: true }),
    ),
  );
});
