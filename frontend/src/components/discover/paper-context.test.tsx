import { fireEvent, render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { getPaperContext } from "@/lib/api";
import PaperContext from "./paper-context";

vi.mock("@/lib/api", () => ({
  getPaperContext: vi.fn().mockResolvedValue({
    metadata: { source: "Crossref", venue: "Research Journal", publisher: "Example Press", type: "journal-article", published: "2024-02-03", reference_count: 6 },
    errors: [],
  }),
}));

it("loads source-attributed metadata only after opening a paper", async () => {
  render(<PaperContext doi="10.1234/example" />);
  expect(getPaperContext).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Explore DOI metadata" }));
  expect(await screen.findByText("Research Journal")).toBeInTheDocument();
  expect(screen.queryByText(/Citing papers/)).not.toBeInTheDocument();
  expect(getPaperContext).toHaveBeenCalledWith("10.1234/example");
});
