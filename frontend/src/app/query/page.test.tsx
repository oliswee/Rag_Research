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

it("uses automatic retrieval while retaining scope filters", async () => {
  render(<QueryPage />);
  expect(screen.queryByText("Rewrite query with AI")).not.toBeInTheDocument();
  expect(screen.queryByText("Mode")).not.toBeInTheDocument();
  fireEvent.click(screen.getByText("Search filters"));
  const input = screen.getByPlaceholderText("Ask a question about your research...");
  fireEvent.change(input, { target: { value: "What about its latency?" } });
  fireEvent.keyDown(input, { key: "Enter" });

  await waitFor(() =>
    expect(submitQuery).toHaveBeenCalledWith(
      "What about its latency?",
      undefined,
      expect.any(Object),
    ),
  );
  const options = vi.mocked(submitQuery).mock.calls[0][2];
  expect(options).not.toHaveProperty("rewrite_query");
  expect(options).not.toHaveProperty("retrieval_mode");
});
