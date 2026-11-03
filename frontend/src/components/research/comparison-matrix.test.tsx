import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import { ComparisonMatrix } from "./comparison-matrix";

it("separates verified findings, missing evidence and scientific non-comparability", () => {
  render(
    <ComparisonMatrix
      result={{
        run_id: "r",
        papers: [
          { id: "a", title: "Study A" },
          { id: "b", title: "Study B" },
        ],
        rows: [
          {
            label: "Common findings",
            cells: [
              {
                paper_id: "a",
                text: "Memory accuracy is 95 percent.",
                support_status: "supported",
                claims: [],
                evidence: [],
                citations: [
                  {
                    citation_id: "c",
                    source_id: "source",
                    paper_id: "a",
                    source_origin: "web",
                    evidence_type: "abstract_only",
                    source_url: "https://example.org/a",
                    raw_quote: "Memory accuracy is 95 percent.",
                    span: [],
                    pdf_page: null,
                    version: "v1",
                    support_status: "supported",
                  },
                ],
              },
              {
                paper_id: "b",
                text: "Unverified accuracy is 99 percent.",
                support_status: "unknown",
                claims: [],
                citations: [],
                evidence: [],
              },
            ],
            comparisons: [
              {
                paper_ids: ["a", "b"],
                status: "not_comparable",
                reasons: ["effect_measure_mismatch"],
              },
            ],
          },
        ],
        coverage: {},
        notices: [],
        web_records: [],
        verification: { calls: 4 },
      }}
    />
  );
  expect(screen.getByText("Evidence insufficient")).toBeInTheDocument();
  expect(screen.getByText("Not comparable")).toBeInTheDocument();
  expect(screen.getByText("Effect measures differ")).toBeInTheDocument();
  expect(screen.getByText("Web · Abstract only")).toBeInTheDocument();
  expect(
    screen.queryByText("Unverified accuracy is 99 percent.")
  ).not.toBeInTheDocument();
  expect(screen.queryByText(/Page \d/)).not.toBeInTheDocument();
});
