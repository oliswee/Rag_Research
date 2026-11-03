import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import { getFeedbackRun, submitFeedback } from "@/lib/api";
import { AnswerFeedback } from "./answer-feedback";

vi.mock("@/lib/api", () => ({
  getFeedbackRun: vi.fn().mockResolvedValue({run_id:"r",config_version:"config",sources:[
    {source_id:"local",paper_id:"p",source_origin:"local",evidence_type:"full_text",source_url:"/api/sources/chunks/local"},
    {source_id:"web",paper_id:"p",source_origin:"web",evidence_type:"abstract_only",source_url:"https://example.org/a",doi:"10.1234/a"},
  ]}),
  submitFeedback: vi.fn().mockResolvedValue({id:"f",review_status:"needs_review"}),
}));
beforeEach(() => vi.clearAllMocks());

it("saves a source-bound Web feedback category locally", async () => {
  render(<AnswerFeedback runId="r" />);
  expect(getFeedbackRun).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", {name:"Report an issue"}));
  await screen.findByLabelText("Error type");
  fireEvent.change(screen.getByLabelText("Error type"), {target:{value:"web_result_irrelevant"}});
  fireEvent.change(screen.getByLabelText("Evidence"), {target:{value:"web"}});
  fireEvent.click(screen.getByRole("button", {name:"Save feedback"}));
  await screen.findByText("Saved locally for review.");
  expect(submitFeedback).toHaveBeenCalledWith({run_id:"r",reason:"web_result_irrelevant",source_id:"web"},expect.anything());
});

it("does not fabricate a saved result when the run cannot be loaded", async () => {
  vi.mocked(getFeedbackRun).mockRejectedValueOnce(new Error("Unknown run"));
  render(<AnswerFeedback runId="old" />);
  fireEvent.click(screen.getByRole("button", {name:"Report an issue"}));
  await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
  expect(screen.queryByRole("button", {name:"Save feedback"})).not.toBeInTheDocument();
  expect(submitFeedback).not.toHaveBeenCalled();
});
