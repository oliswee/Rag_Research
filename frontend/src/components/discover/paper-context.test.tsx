import { fireEvent, render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { getPaperContext } from "@/lib/api";
import PaperContext from "./paper-context";

vi.mock("@/lib/api", () => ({
  getPaperContext: vi.fn().mockResolvedValue({
    metadata: { source: "Crossref", venue: "Research Journal", publisher: "Example Press", type: "journal-article", published: "2024-02-03", reference_count: 6 },
    citing_papers: [{ source: "Semantic Scholar", title: "Citing study", year: 2025, url: "https://www.semanticscholar.org/paper/abc123" }],
    references: [{ source: "OpenAlex", title: "Referenced study", year: 2020, url: "https://openalex.org/W456" }],
    errors: [],
  }),
}));

it("loads source-attributed metadata only after opening a paper", async () => {
  render(<PaperContext provider="openalex" providerId="W123" />);
  expect(getPaperContext).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Explore metadata and citations" }));
  expect(await screen.findByText("Research Journal")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Citing study" })).toHaveAttribute("href", "https://www.semanticscholar.org/paper/abc123");
  expect(screen.getByRole("link", { name: "Referenced study" })).toHaveAttribute("href", "https://openalex.org/W456");
  expect(getPaperContext).toHaveBeenCalledWith("openalex", "W123");
});
