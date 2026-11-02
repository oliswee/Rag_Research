import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { comparePapers } from "@/lib/api";
import ComparePage from "./page";

vi.mock("@/lib/api", () => ({
  listAllPapers: vi.fn().mockResolvedValue([
    { id: "a", title: "Study A", import_source: null },
    { id: "b", title: "Study B", import_source: "openalex" },
  ]),
  comparePapers: vi.fn().mockResolvedValue({
    papers: [
      { id: "a", title: "Study A" },
      { id: "b", title: "Study B" },
    ],
    rows: [],
    notices: [],
  }),
}));

it("sends explicit privacy controls for the chosen study identities", async () => {
  render(<ComparePage />);
  await waitFor(() =>
    expect(screen.getByLabelText(/Study A/)).toBeInTheDocument()
  );
  fireEvent.click(screen.getByLabelText(/Study A/));
  fireEvent.click(screen.getByLabelText(/Study B/));
  fireEvent.click(screen.getByLabelText("Local only"));
  fireEvent.click(screen.getByLabelText("Only uploaded papers"));
  fireEvent.click(
    screen.getByRole("button", { name: "Compare selected papers" })
  );
  await waitFor(() =>
    expect(comparePapers).toHaveBeenCalledWith(
      ["a", "b"],
      undefined,
      expect.objectContaining({
        local_only: true,
        only_uploaded: true,
        latest: false,
      })
    )
  );
});
