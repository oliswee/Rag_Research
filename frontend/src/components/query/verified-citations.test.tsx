import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import { VerifiedCitations } from "./verified-citations";

it("links canonical local evidence and suppresses fake abstract page numbers", () => {
  render(
    <VerifiedCitations
      citations={[
        {
          citation_id: "l",
          source_id: "chunk-1",
          paper_id: "a",
          source_origin: "local",
          evidence_type: "full_text",
          source_url: "/api/sources/chunks/chunk-1",
          raw_quote: "A verified original statement.",
          span: [],
          pdf_page: 2,
          version: "sha256:known",
          support_status: "supported",
        },
        {
          citation_id: "w",
          source_id: "web:1",
          paper_id: "a",
          source_origin: "web",
          evidence_type: "abstract_only",
          source_url: "javascript:alert(1)",
          raw_quote: "An abstract statement without PDF pages.",
          span: [],
          pdf_page: 99,
          version: "sha256:other",
          support_status: "supported",
        },
      ]}
    />
  );
  expect(
    screen.getByRole("link", { name: /open original source/i })
  ).toHaveAttribute("href", "http://localhost:8000/api/sources/chunks/chunk-1");
  expect(screen.getByText("Page 2")).toBeInTheDocument();
  expect(screen.queryByText("Page 99")).not.toBeInTheDocument();
  expect(screen.getAllByRole("link")).toHaveLength(1);
});
