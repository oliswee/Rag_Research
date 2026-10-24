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

it("rewrites questions automatically without showing a rewrite control", async () => {
  render(<QueryPage />);
  expect(screen.queryByText("Rewrite query with AI")).not.toBeInTheDocument();
  expect(screen.queryByRole("combobox", { name: "Mode" })).not.toBeInTheDocument();
  const input = screen.getByPlaceholderText("Ask a question about your research...");
  fireEvent.change(input, { target: { value: "How does active retrieval work?" } });
  fireEvent.keyDown(input, { key: "Enter" });

  await waitFor(() =>
    expect(submitQuery).toHaveBeenCalledWith(
      "How does active retrieval work?",
      undefined,
      expect.objectContaining({ rewrite_query: true, retrieval_mode: "hybrid" }),
    ),
  );

  await waitFor(() => expect(screen.getByText("Source excerpts are shown below.")).toBeInTheDocument());
  fireEvent.change(input, { target: { value: "What about its latency?" } });
  fireEvent.keyDown(input, { key: "Enter" });

  await waitFor(() =>
    expect(submitQuery).toHaveBeenCalledWith(
      "What about its latency?",
      "session-1",
      expect.objectContaining({ rewrite_query: true, retrieval_mode: "hybrid" }),
    ),
  );
});
