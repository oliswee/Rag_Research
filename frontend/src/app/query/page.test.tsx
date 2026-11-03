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

it("retains privacy detected from the question in the next turn and after reload", async () => {
  vi.mocked(submitQuery).mockResolvedValueOnce({
    session_id: "private-session",
    response: "Local evidence only.",
    sources: [],
    confidence: "low",
    follow_up_suggestions: [],
    retrieval_mode: "hybrid",
    routing: {
      intent: "factual",
      action: "answer_local",
      sufficiency: "uncertain",
      reasons: ["external_access_prohibited"],
      local_retry_count: 0,
      local_only: true,
      only_uploaded: true,
    },
  });
  const view = render(<QueryPage />);
  const input = screen.getByPlaceholderText(
    "Ask a question about your research..."
  );
  fireEvent.change(input, {
    target: { value: "Only use uploaded local papers" },
  });
  fireEvent.keyDown(input, { key: "Enter" });
  await waitFor(() =>
    expect(screen.getByText("Local evidence only.")).toBeInTheDocument()
  );
  expect(screen.getByLabelText("Local only")).toBeChecked();
  expect(screen.getByLabelText("Only uploaded papers")).toBeChecked();
  view.unmount();
  render(<QueryPage />);
  expect(screen.getByLabelText("Local only")).toBeChecked();
  const next = screen.getByPlaceholderText(
    "Ask a question about your research..."
  );
  fireEvent.change(next, { target: { value: "latest memory papers" } });
  fireEvent.keyDown(next, { key: "Enter" });
  await waitFor(() => expect(submitQuery).toHaveBeenCalledTimes(2));
  expect(vi.mocked(submitQuery).mock.calls.at(-1)?.[2]).toMatchObject({
    local_only: true,
    only_uploaded: true,
  });
});

it("rewrites questions automatically without showing a rewrite control", async () => {
  render(<QueryPage />);
  expect(screen.queryByText("Rewrite query with AI")).not.toBeInTheDocument();
  expect(
    screen.queryByRole("combobox", { name: "Mode" })
  ).not.toBeInTheDocument();
  const input = screen.getByPlaceholderText(
    "Ask a question about your research..."
  );
  fireEvent.change(input, {
    target: { value: "How does active retrieval work?" },
  });
  fireEvent.keyDown(input, { key: "Enter" });
  await waitFor(() =>
    expect(submitQuery).toHaveBeenCalledWith(
      "How does active retrieval work?",
      undefined,
      expect.objectContaining({ rewrite_query: true, retrieval_mode: "hybrid" })
    )
  );
  await waitFor(() =>
    expect(
      screen.getByText("Source excerpts are shown below.")
    ).toBeInTheDocument()
  );
  fireEvent.change(input, { target: { value: "What about its latency?" } });
  fireEvent.keyDown(input, { key: "Enter" });
  await waitFor(() =>
    expect(submitQuery).toHaveBeenCalledWith(
      "What about its latency?",
      "session-1",
      expect.objectContaining({ rewrite_query: true, retrieval_mode: "hybrid" })
    )
  );
});

it("sends privacy controls and explicit scope resets", async () => {
  render(<QueryPage />);
  fireEvent.click(screen.getByLabelText("Local only"));
  fireEvent.click(screen.getByLabelText("Only uploaded papers"));
  fireEvent.click(screen.getByLabelText("Find latest papers"));
  const input = screen.getByPlaceholderText(
    "Ask a question about your research..."
  );
  fireEvent.change(input, { target: { value: "memory" } });
  fireEvent.keyDown(input, { key: "Enter" });
  await waitFor(() => expect(submitQuery).toHaveBeenCalled());
  expect(vi.mocked(submitQuery).mock.calls.at(-1)?.[2]).toMatchObject({
    local_only: true,
    only_uploaded: true,
    latest: true,
    paper_ids: null,
    year_from: null,
    year_to: null,
    provider: null,
  });
});
