import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { discoverPapers, startImport } from "@/lib/api";
import DiscoverPage from "./page";

vi.mock("@/lib/api", () => ({
  discoverPapers: vi.fn().mockResolvedValue({
    errors: ["Semantic Scholar: 429 rate limit reached"],
    candidates: [{
      provider: "openalex", provider_id: "W1", title: "Retrieval paper",
      authors: ["A. Author"], year: 2025, abstract: "Retrieval improves accuracy.",
      abstract_summary: "Retrieval improves accuracy.",
      source_url: "https://openalex.org/W1", pdf_url: null, open_access: false,
      doi: null, citation_count: 2, tags: ["Information Retrieval"],
      relevance_reason: "Matches topic terms: retrieval.", rank_score: 1,
      possible_duplicate_ids: [],
    }],
  }),
  startImport: vi.fn().mockResolvedValue({ job_id: "job-1", status: "queued" }),
  getImport: vi.fn(), retryImport: vi.fn(),
}));

it("explains a provider outage and sends one chosen paper to the library", async () => {
  render(<DiscoverPage />);
  fireEvent.change(screen.getByLabelText("Research topic or question"), {
    target: { value: "retrieval" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Search papers" }));
  await screen.findByText("Retrieval paper");
  expect(screen.getByText(/429 rate limit reached/)).toBeInTheDocument();
  expect(screen.getByText("Information Retrieval")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Add to library" }));
  await waitFor(() => expect(startImport).toHaveBeenCalledWith([
    expect.objectContaining({ provider: "openalex", provider_id: "W1" }),
  ]));
  expect(discoverPapers).toHaveBeenCalled();
});
