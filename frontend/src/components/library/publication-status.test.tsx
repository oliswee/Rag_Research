import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { getPublicationStatus, refreshPublicationStatus } from "@/lib/api";
import { PublicationStatusPanel } from "./publication-status";

vi.mock("@/lib/api", () => ({
  getPublicationStatus: vi
    .fn()
    .mockResolvedValue({
      paper_id: "p",
      doi: "10.1234/a",
      status: "unknown",
      relations: [],
      updates: [],
      notices: [],
      checked_at: null,
    }),
  refreshPublicationStatus: vi
    .fn()
    .mockResolvedValue({
      paper_id: "p",
      doi: "10.1234/a",
      status: "retracted",
      relations: [],
      updates: [],
      notices: [],
      checked_at: "2026-11-03T00:00:00Z",
      source_url: "https://api.crossref.org/works/10.1234/a",
    }),
}));

it("loads cached status only on demand and makes unknown explicit", async () => {
  render(<PublicationStatusPanel paperId="p" />);
  expect(getPublicationStatus).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Publication status" }));
  await screen.findByText("Retraction status unknown");
  expect(refreshPublicationStatus).not.toHaveBeenCalled();
  fireEvent.click(
    screen.getByRole("button", { name: "Check public DOI metadata" })
  );
  await screen.findByText("Retraction notice reported");
  expect(
    screen.getByRole("link", { name: "Public status source" })
  ).toHaveAttribute("href", "https://api.crossref.org/works/10.1234/a");
});

it("keeps a previous verified status visible when a refresh fails", async () => {
  vi.mocked(getPublicationStatus).mockResolvedValueOnce({
    paper_id: "p",
    doi: "10.1234/a",
    status: "retracted",
    relations: [],
    updates: [],
    notices: [],
    checked_at: "2026-11-03T00:00:00Z",
  });
  vi.mocked(refreshPublicationStatus).mockRejectedValueOnce(
    new Error("Unavailable")
  );
  render(<PublicationStatusPanel paperId="p" />);
  fireEvent.click(screen.getByRole("button", { name: "Publication status" }));
  await screen.findByText("Retraction notice reported");
  fireEvent.click(
    screen.getByRole("button", { name: "Check public DOI metadata" })
  );
  await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
  expect(screen.getByText("Retraction notice reported")).toBeInTheDocument();
});
