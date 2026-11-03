"use client";

import type { CompareResult, ComparisonAssessment } from "@/types";
import { VerifiedCitations } from "@/components/query/verified-citations";
import { sourceUrl } from "@/lib/source-url";

const STATUS: Record<ComparisonAssessment["status"], string> = {
  comparable_agreement: "Comparable agreement",
  comparable_conflict: "Comparable conflict",
  not_comparable: "Not comparable",
  insufficient_evidence: "Insufficient comparison evidence",
};
const REASONS: Record<string, string> = {
  effect_measure_mismatch: "Effect measures differ",
  population_mismatch: "Populations differ",
  study_design_mismatch: "Study designs differ",
  outcome_mismatch: "Outcomes differ",
  exposure_mismatch: "Exposures differ",
  adjustment_mismatch: "Adjustment differs",
  same_study_not_independent: "These sources describe the same study",
  same_study_different_version: "Different versions of the same study",
  matched_supported_characteristics: "Supported study characteristics match",
  study_independence_unknown: "Independent study identity is unknown",
};

export function ComparisonMatrix({ result }: { result: CompareResult }) {
  const titles = new Map(result.papers.map((paper) => [paper.id, paper.title]));
  return (
    <div className="space-y-4">
      <p className="text-sm leading-6 text-muted-foreground">
        Findings appear only after evidence checks. Different populations,
        methods or effect measures can make studies incomparable; missing
        details remain unknown.
      </p>
      {(result.notices || []).length > 0 && (
        <details className="rounded-lg border px-4 py-3 text-xs text-muted-foreground">
          <summary className="cursor-pointer font-medium">
            Evidence access and coverage notes
          </summary>
          <ul className="mt-2 space-y-2">
            {result.notices?.map((notice, i) => (
              <li key={i}>{notice}</li>
            ))}
          </ul>
        </details>
      )}
      <div className="overflow-x-auto rounded-xl border">
        <table className="w-full min-w-[680px] border-collapse text-left text-sm">
          <thead>
            <tr className="bg-muted">
              <th className="w-44 border-b p-4">Dimension</th>
              {result.papers.map((paper) => (
                <th className="min-w-64 border-b p-4 align-top" key={paper.id}>
                  {paper.title}
                  {result.coverage?.[paper.id] && (
                    <p className="mt-2 text-xs font-normal text-muted-foreground">
                      {result.coverage[paper.id].context_count || 0} excerpts ·{" "}
                      {result.coverage[paper.id].supported_count || 0} verified
                      claims
                    </p>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {result.rows.map((row) => (
              <tr key={row.label} className="align-top">
                <th className="border-b bg-muted/40 p-4 font-semibold">
                  {row.label}
                </th>
                {row.cells.map((cell) => {
                  const verified =
                    cell.support_status === "supported" &&
                    !!cell.citations?.length;
                  return (
                    <td key={cell.paper_id} className="border-b p-4">
                      <span
                        className={`inline-block rounded px-2 py-1 text-xs font-medium ${verified ? "bg-emerald-50 text-emerald-800" : "bg-muted text-muted-foreground"}`}
                      >
                        {verified
                          ? "Verified finding"
                          : cell.support_status === "unsupported" ||
                              cell.support_status === "contradicted"
                            ? "Unsupported finding"
                            : "Evidence insufficient"}
                      </span>
                      <p className="mt-3 whitespace-pre-wrap leading-6">
                        {verified
                          ? cell.text
                          : "No verified conclusion for this dimension."}
                      </p>
                      <VerifiedCitations
                        citations={verified ? cell.citations || [] : []}
                      />
                      {verified &&
                        cell.claims?.some((claim) => claim.characteristics) && (
                          <details className="mt-3 text-xs text-muted-foreground">
                            <summary className="cursor-pointer">
                              Supported study details
                            </summary>
                            {cell.claims
                              .filter(
                                (claim) => claim.support_status === "supported"
                              )
                              .map((claim) => (
                                <dl
                                  key={claim.claim_id}
                                  className="mt-2 space-y-2"
                                >
                                  {Object.entries(
                                    claim.characteristics || {}
                                  ).map(([name, feature]) =>
                                    typeof feature === "object" &&
                                    feature?.support_status === "supported" &&
                                    feature.value ? (
                                      <div key={name}>
                                        <dt className="font-medium capitalize">
                                          {name.replaceAll("_", " ")}
                                        </dt>
                                        <dd>
                                          {feature.value}
                                          <blockquote className="mt-1 border-l-2 pl-2">
                                            {feature.raw_quote}
                                          </blockquote>
                                        </dd>
                                      </div>
                                    ) : null
                                  )}
                                </dl>
                              ))}
                          </details>
                        )}
                      {!verified && cell.evidence.length > 0 && (
                        <details className="mt-3 text-xs text-muted-foreground">
                          <summary className="cursor-pointer">
                            Original excerpts · {cell.evidence.length}
                          </summary>
                          {cell.evidence.map((source) => (
                            <blockquote
                              key={source.chunk_id}
                              className="mt-2 border-l-2 pl-2"
                            >
                              <p className="whitespace-pre-wrap leading-5">
                                {source.excerpt}
                              </p>
                              {sourceUrl(source.source_url) && (
                                <a
                                  className="mt-1 inline-block text-emerald-700 hover:underline"
                                  href={sourceUrl(source.source_url)!}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                >
                                  Open source
                                </a>
                              )}
                            </blockquote>
                          ))}
                        </details>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <section className="space-y-3">
        <h2 className="text-sm font-semibold">Study comparability</h2>
        {result.rows.map((row) => (
          <details key={row.label} className="rounded-lg border p-4 text-sm">
            <summary className="cursor-pointer font-medium">
              {row.label} · {row.comparisons?.length || 0} pairs checked
            </summary>
            <div className="mt-3 space-y-3">
              {(row.comparisons || []).map((pair) => (
                <article
                  key={pair.paper_ids.join(":")}
                  className="border-t pt-3"
                >
                  <p className="text-xs text-muted-foreground">
                    {titles.get(pair.paper_ids[0])} ↔{" "}
                    {titles.get(pair.paper_ids[1])}
                  </p>
                  <p
                    className={`mt-1 font-medium ${pair.status === "comparable_conflict" ? "text-amber-700" : "text-foreground"}`}
                  >
                    {STATUS[pair.status]}
                  </p>
                  <ul className="mt-1 space-y-1 text-xs text-muted-foreground">
                    {pair.reasons.map((reason) => (
                      <li key={reason}>
                        {REASONS[reason] ||
                          (reason.endsWith("_unknown")
                            ? `${reason.replace("_unknown", "").replaceAll("_", " ")} is unknown`
                            : "Additional study details are needed")}
                      </li>
                    ))}
                  </ul>
                </article>
              ))}
            </div>
          </details>
        ))}
      </section>
      {!!result.web_records?.length && (
        <details className="rounded-lg border p-4 text-xs text-muted-foreground">
          <summary className="cursor-pointer font-medium">
            Academic access records · {result.web_records.length}
          </summary>
          <p className="mt-2">
            Bibliographic metadata helps locate a paper; it does not establish a
            finding.
          </p>
          <ul className="mt-3 space-y-3">
            {result.web_records.map((record, index) => (
              <li key={`${record.paper_id}:${index}`}>
                <p className="font-medium">{record.title}</p>
                <p>
                  {record.evidence_level.replaceAll("_", " ")} ·{" "}
                  {record.access_status || "Access unknown"}
                </p>
                {sourceUrl(record.source_url) && (
                  <a
                    href={sourceUrl(record.source_url)!}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-emerald-700 hover:underline"
                  >
                    Open academic record
                  </a>
                )}
              </li>
            ))}
          </ul>
        </details>
      )}
      {result.rows.some((row) =>
        row.cells.some((cell) => cell.support_status !== "supported")
      ) && (
        <a
          href="/discover"
          className="inline-block text-sm font-medium text-emerald-700 hover:underline"
        >
          Find papers or full-text access
        </a>
      )}
    </div>
  );
}
