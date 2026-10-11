import { fireEvent, render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { getResearcher, searchResearchers } from "@/lib/api";
import ResearchersPage from "./page";

vi.mock("@/lib/api", () => ({
  searchResearchers: vi.fn().mockResolvedValue({
    total: 1,
    resolved_filters: { topic: "Machine Learning" },
    researchers: [{ id: "A1", name: "Ada Researcher", institutions: ["University of Sydney"],
      topics: ["Machine Learning"], fields: ["Computer Science"], works_count: 10,
      cited_by_count: 20, orcid: null, source_url: "https://openalex.org/A1" }],
  }),
  getResearcher: vi.fn().mockResolvedValue({
    id: "A1", name: "Ada Researcher", institutions: ["University of Sydney"],
    topics: [], fields: [], works_count: 10, cited_by_count: 20, orcid: null,
    source_url: "https://openalex.org/A1",
    publications: [{ id: "W1", title: "A recent paper", year: 2025,
      citation_count: 2, topic: "Machine Learning", source_url: "https://openalex.org/W1" }],
    coauthors: [{ id: "A2", name: "Bea Researcher" }],
  }),
}));

it("searches Australian researchers and opens sourced publications", async () => {
  render(<ResearchersPage />);
  fireEvent.change(screen.getByLabelText("Research topic"), {
    target: { value: "machine learning" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Search researchers" }));
  await screen.findByText("Ada Researcher");
  expect(searchResearchers).toHaveBeenCalledWith(expect.objectContaining({
    topic: "machine learning",
  }));
  fireEvent.click(screen.getByRole("button", { name: "View publications and coauthors" }));
  await screen.findByText(/A recent paper/);
  expect(screen.getByText("Bea Researcher")).toBeInTheDocument();
  expect(getResearcher).toHaveBeenCalledWith("A1");
});
