/**
 * API client for FolioTrace backend.
 */

import type {
  ApiError,
  ApiRequestError as ApiRequestErrorType,
  PaperListParams,
  PaperListResponse,
  PaperResponse,
  PaperUploadResponse,
  PatternDecision,
  PatternDetailResponse,
  PatternListResponse,
  PatternResponse,
  PatternReviewResponse,
  PatternStatus,
  PatternUpdateRequest,
  QueryResponse,
  DiscoveryResult,
  Candidate,
  ImportJob,
  CompareResult,
} from "@/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ============================================================================
// Error Handling
// ============================================================================

/**
 * API request error with code and status.
 */
export class ApiRequestError extends Error implements ApiRequestErrorType {
  code: string;
  status: number;

  constructor(code: string, message: string, status: number) {
    super(message);
    this.name = "ApiRequestError";
    this.code = code;
    this.status = status;
  }
}

/**
 * Parse error response from API.
 */
async function parseError(response: Response): Promise<ApiRequestError> {
  try {
    const data = await response.json();
    const error = data.detail as ApiError | undefined;
    if (error && typeof error === "object" && "code" in error) {
      return new ApiRequestError(
        error.code,
        error.message,
        response.status
      );
    }
    // Fallback for non-standard errors
    return new ApiRequestError(
      "UNKNOWN_ERROR",
      typeof data.detail === "string" ? data.detail : "Request failed",
      response.status
    );
  } catch {
    return new ApiRequestError(
      "UNKNOWN_ERROR",
      `Request failed with status ${response.status}`,
      response.status
    );
  }
}

/**
 * Make a fetch request with error handling.
 */
async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });

  if (!response.ok) {
    throw await parseError(response);
  }

  // Handle 204 No Content
  if (response.status === 204) {
    return undefined as T;
  }

  return response.json();
}

/**
 * Make a fetch request for FormData (no Content-Type header).
 */
async function requestFormData<T>(
  path: string,
  formData: FormData
): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    method: "POST",
    body: formData,
    // Don't set Content-Type - browser will set it with boundary
  });

  if (!response.ok) {
    throw await parseError(response);
  }

  return response.json();
}

// ============================================================================
// Health
// ============================================================================

/**
 * Health check response from backend.
 */
export interface HealthStatus {
  status: "ok" | "degraded" | "error";
  database: "ok" | "error";
  qdrant: "ok" | "error";
  sessions: number;
  tracing: "enabled" | "disabled";
}

/**
 * Fetch health status from backend.
 */
export async function getHealth(): Promise<HealthStatus> {
  return request<HealthStatus>("/health");
}

// ============================================================================
// Papers
// ============================================================================

/**
 * Upload a paper (PDF or text) for processing.
 */
export async function uploadPaper(
  formData: FormData
): Promise<PaperUploadResponse> {
  return requestFormData<PaperUploadResponse>("/api/papers", formData);
}

/**
 * List papers with optional filters.
 */
export async function listPapers(
  params?: PaperListParams
): Promise<PaperListResponse> {
  const searchParams = new URLSearchParams();
  if (params?.status) searchParams.set("status", params.status);
  if (params?.type) searchParams.set("type", params.type);
  if (params?.limit) searchParams.set("limit", params.limit.toString());
  if (params?.offset) searchParams.set("offset", params.offset.toString());

  const query = searchParams.toString();
  const path = query ? `/api/papers?${query}` : "/api/papers";
  return request<PaperListResponse>(path);
}

export async function listAllPapers(): Promise<PaperResponse[]> {
  const papers: PaperResponse[] = [];
  while (true) {
    const page = await listPapers({ limit: 100, offset: papers.length });
    papers.push(...page.papers);
    if (papers.length >= page.total || page.papers.length === 0) return papers;
  }
}

/**
 * Get a single paper by ID.
 */
export async function getPaper(paperId: string): Promise<PaperResponse> {
  return request<PaperResponse>(`/api/papers/${paperId}`);
}

/**
 * Delete a paper and all its patterns.
 */
export async function deletePaper(paperId: string): Promise<void> {
  return request<void>(`/api/papers/${paperId}`, { method: "DELETE" });
}

// ============================================================================
// Patterns
// ============================================================================

/**
 * Submit pattern review decisions for a paper.
 */
export async function submitPatternReview(
  paperId: string,
  decisions: PatternDecision[]
): Promise<PatternReviewResponse> {
  return request<PatternReviewResponse>(`/api/papers/${paperId}/patterns`, {
    method: "POST",
    body: JSON.stringify({ decisions }),
  });
}

/**
 * List patterns for a paper.
 */
export async function listPatterns(
  paperId: string,
  status?: PatternStatus
): Promise<PatternListResponse> {
  const searchParams = new URLSearchParams();
  if (status) searchParams.set("status", status);

  const query = searchParams.toString();
  const path = query
    ? `/api/papers/${paperId}/patterns?${query}`
    : `/api/papers/${paperId}/patterns`;
  return request<PatternListResponse>(path);
}

/**
 * Get a single pattern by ID with full details.
 */
export async function getPattern(
  patternId: string
): Promise<PatternDetailResponse> {
  return request<PatternDetailResponse>(`/api/patterns/${patternId}`);
}

/**
 * Update a pattern.
 */
export async function updatePattern(
  patternId: string,
  data: PatternUpdateRequest
): Promise<PatternResponse> {
  return request<PatternResponse>(`/api/patterns/${patternId}`, {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

/**
 * Delete a pattern.
 */
export async function deletePattern(patternId: string): Promise<void> {
  return request<void>(`/api/patterns/${patternId}`, { method: "DELETE" });
}

// ============================================================================
// Query
// ============================================================================

/**
 * Submit a query (creates new session if sessionId not provided).
 */
export async function submitQuery(
  query: string,
  sessionId?: string,
  options?: { paper_ids?: string[]; retrieval_mode?: "dense" | "bm25" | "hybrid"; rewrite_query?: boolean; year_from?: number; year_to?: number; provider?: "semantic_scholar" | "openalex" | "pubmed" }
): Promise<QueryResponse> {
  return request<QueryResponse>("/api/query", {
    method: "POST",
    body: JSON.stringify({ query, session_id: sessionId, ...options }),
  });
}

/**
 * Submit a follow-up query (requires sessionId).
 */
export async function submitFollowUp(
  query: string,
  sessionId: string
): Promise<QueryResponse> {
  return request<QueryResponse>("/api/query/followup", {
    method: "POST",
    body: JSON.stringify({ query, session_id: sessionId }),
  });
}

export async function discoverPapers(params: { query: string; year_from?: number; year_to?: number; open_access?: boolean; semantic_offset?: number; openalex_offset?: number; pubmed_offset?: number }): Promise<DiscoveryResult> {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => { if (value !== undefined) query.set(key, String(value)); });
  return request<DiscoveryResult>(`/api/discovery/search?${query}`);
}

export async function startImport(candidates: Candidate[]): Promise<{ job_id: string; status: string; estimated_api_calls: number }> {
  return request<{ job_id: string; status: string; estimated_api_calls: number }>("/api/discovery/import", { method: "POST", body: JSON.stringify({ candidates: candidates.map(({ provider, provider_id }) => ({ provider, provider_id })) }) });
}

export async function getImport(jobId: string): Promise<ImportJob> {
  return request<ImportJob>(`/api/discovery/imports/${jobId}`);
}

export async function retryImport(jobId: string): Promise<void> {
  return request<void>(`/api/discovery/imports/${jobId}/retry`, { method: "POST" });
}

export async function comparePapers(paperIds: string[], question?: string): Promise<CompareResult> {
  return request<CompareResult>("/api/research/compare", { method: "POST", body: JSON.stringify({ paper_ids: paperIds, question }) });
}

// ============================================================================
// Convenience Export
// ============================================================================

export const api = {
  health: getHealth,
  papers: {
    upload: uploadPaper,
    list: listPapers,
    get: getPaper,
    delete: deletePaper,
  },
  patterns: {
    submitReview: submitPatternReview,
    list: listPatterns,
    get: getPattern,
    update: updatePattern,
    delete: deletePattern,
  },
  query: {
    submit: submitQuery,
    followUp: submitFollowUp,
  },
};
