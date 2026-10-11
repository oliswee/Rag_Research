import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import { discoverPapers, startImport } from "@/lib/api";
import type { DiscoveryResult } from "@/types";
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
    role_labels: ["Recent"], cited_by_results: [],
  };
  return {
  discoverPapers: vi.fn().mockResolvedValue({
    errors: ["Semantic Scholar: 429 rate limit reached"],
    themes: [{ label: "Information Retrieval", count: 2 }],
    candidates: [paper, { ...paper, provider_id: "W2", title: "Another retrieval paper" }],
    next_offsets: { semantic_scholar: 0, openalex: 2, pubmed: 0 },
    has_more: false,
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
  expect(screen.getByText("Related themes:")).toBeInTheDocument();
  expect(screen.getAllByText("Recent")).toHaveLength(2);
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

it("loads the next batch without duplicating papers and filters by a selected tag", async () => {
  const first: DiscoveryResult = {
    errors: [], themes: [], has_more: true,
    next_offsets: { semantic_scholar: 0, openalex: 1, pubmed: 0 },
    candidates: [{ provider: "openalex", provider_id: "W1", title: "First paper", authors: [], year: 2024,
      abstract: "First abstract", abstract_summary: "First abstract", source_url: null, pdf_url: null,
      open_access: false, doi: "10.1234/first", citation_count: 1, tags: ["Biology"],
      relevance_reason: "Matches biology.", rank_score: 1, possible_duplicate_ids: [], role_labels: [], cited_by_results: [] }],
  };
  const second: DiscoveryResult = {
    ...first, has_more: false,
    next_offsets: { semantic_scholar: 0, openalex: 2, pubmed: 1 },
    candidates: [first.candidates[0], { ...first.candidates[0], provider: "pubmed", provider_id: "123",
      doi: "10.1234/second", title: "Second paper", tags: ["Genetics"] }],
  };
  vi.mocked(discoverPapers).mockResolvedValueOnce(first).mockResolvedValueOnce(second);
  render(<DiscoverPage />);
  fireEvent.change(screen.getByLabelText("Research topic or question"), { target: { value: "biology" } });
  fireEvent.click(screen.getByRole("button", { name: "Search papers" }));
  await screen.findByText("First paper");
  fireEvent.click(screen.getByRole("button", { name: "Load more papers" }));
  await screen.findByText("Second paper");
  expect(screen.getAllByText("First paper")).toHaveLength(1);
  expect(discoverPapers).toHaveBeenLastCalledWith(expect.objectContaining({ openalex_offset: 1 }));
  fireEvent.click(screen.getAllByRole("button", { name: "Filter by Genetics" })[0]);
  expect(screen.queryByText("First paper")).not.toBeInTheDocument();
  expect(screen.getByText("Second paper")).toBeInTheDocument();
});

it("combines topic filters that differ only by case", async () => {
  const paper = {
    provider: "openalex" as const, provider_id: "W1", title: "First genetics paper",
    authors: [], year: 2024, abstract: null, abstract_summary: null,
    source_url: null, pdf_url: null, open_access: false, doi: null,
    citation_count: null, tags: ["Genetics"], relevance_reason: "Source result.",
    rank_score: 1, possible_duplicate_ids: [], role_labels: [], cited_by_results: [],
  };
  vi.mocked(discoverPapers).mockResolvedValueOnce({
    candidates: [paper, { ...paper, provider_id: "W2", title: "Second genetics paper", tags: ["genetics"] }],
    errors: [], themes: [], has_more: false,
    next_offsets: { semantic_scholar: 0, openalex: 2, pubmed: 0 },
  });
  render(<DiscoverPage />);
  fireEvent.change(screen.getByLabelText("Research topic or question"), { target: { value: "genetics" } });
  fireEvent.click(screen.getByRole("button", { name: "Search papers" }));
  await screen.findByText("First genetics paper");
  expect(screen.getByText("Genetics · 2")).toBeInTheDocument();
  fireEvent.click(screen.getAllByRole("button", { name: "Filter by Genetics" })[0]);
  expect(screen.getByText("Second genetics paper")).toBeInTheDocument();
});
