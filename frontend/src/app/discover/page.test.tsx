import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import { discoverPapers, startImport } from "@/lib/api";
import DiscoverPage from "./page";

vi.mock("@/lib/api", () => {
  const paper = {
    provider: "openalex", provider_id: "W1", title: "Retrieval paper",
    authors: ["A. Author"], year: 2025, abstract: "Retrieval improves accuracy.",
    abstract_summary: "Retrieval improves accuracy.",
    source_url: "https://openalex.org/W1", pdf_url: null, open_access: false,
    doi: null, citation_count: 2, tags: ["Information Retrieval"],
    relevance_reason: "Matches topic terms: retrieval.", rank_score: 1,
    possible_duplicate_ids: [],
  };
  return {
  discoverPapers: vi.fn().mockResolvedValue({
    errors: ["Semantic Scholar: 429 rate limit reached"],
    candidates: [paper, { ...paper, provider_id: "W2", title: "Another retrieval paper" }],
  }),
  startImport: vi.fn().mockResolvedValue({ job_id: "job-1", status: "queued" }),
  getImport: vi.fn(), retryImport: vi.fn(),
};
});

beforeEach(() => vi.mocked(startImport).mockClear());

it("explains a provider outage and sends one chosen paper to the library", async () => {
  render(<DiscoverPage />);
  fireEvent.change(screen.getByLabelText("Research topic or question"), {
    target: { value: "retrieval" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Search papers" }));
  await screen.findByText("Retrieval paper");
  expect(screen.getByText(/429 rate limit reached/)).toBeInTheDocument();
  expect(screen.getAllByText("Information Retrieval")).toHaveLength(2);
  fireEvent.click(screen.getAllByRole("button", { name: "Add to library" })[0]);
  await waitFor(() => expect(startImport).toHaveBeenCalledWith([
    expect.objectContaining({ provider: "openalex", provider_id: "W1" }),
  ]));
  expect(discoverPapers).toHaveBeenCalled();
});

it("imports multiple selected papers in one job", async () => {
  render(<DiscoverPage />);
  fireEvent.change(screen.getByLabelText("Research topic or question"), {
    target: { value: "retrieval" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Search papers" }));
  await screen.findByText("Another retrieval paper");
  screen.getAllByLabelText("Select for batch").forEach((checkbox) => fireEvent.click(checkbox));
  fireEvent.click(screen.getByRole("button", { name: "Add 2 selected to library" }));
  await waitFor(() => expect(startImport).toHaveBeenCalledWith([
    expect.objectContaining({ provider_id: "W1" }),
    expect.objectContaining({ provider_id: "W2" }),
  ]));
});
