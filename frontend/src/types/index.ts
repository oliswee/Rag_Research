/**
 * TypeScript types matching backend Pydantic schemas.
 */

// ============================================================================
// Enums
// ============================================================================

export type PaperType =
  | "research_paper"
  | "blog_post"
  | "article"
  | "technical_report"
  | "book_chapter";

export type PaperStatus =
  | "processing"
  | "ready_for_review"
  | "active"
  | "archived";

export type PatternStatus = "pending" | "approved" | "rejected";

export type Confidence = "high" | "medium" | "low";

// ============================================================================
// Paper Types
// ============================================================================

/**
 * Full paper response from GET endpoints.
 */
export interface PaperResponse {
  id: string;
  title: string;
  authors: string[];
  date: string | null;
  source: string | null;
  published_date: string | null;
  type: PaperType;
  status: PaperStatus;
  pattern_count: number;
  created_at: string;
  updated_at: string;
}

/**
 * Paper list response with pagination.
 */
export interface PaperListResponse {
  papers: PaperResponse[];
  total: number;
  limit: number;
  offset: number;
}

/**
 * Response from paper upload.
 */
export interface PaperUploadResponse {
  paper_id: string;
  metadata: {
    title: string;
    authors: string[];
    source: string | null;
    published_date: string | null;
    type: PaperType;
  };
  patterns: ExtractedPatternResponse[];
  extraction_warnings: string[];
}

/**
 * Parameters for listing papers.
 */
export interface PaperListParams {
  status?: PaperStatus;
  type?: PaperType;
  limit?: number;
  offset?: number;
}

// ============================================================================
// Pattern Types
// ============================================================================

/**
 * Pattern extracted during upload (before persistence).
 */
export interface ExtractedPatternResponse {
  temp_id: string;
  name: string;
  claim: string | null;
  evidence: string | null;
  context: string | null;
  tags: string[];
  confidence: Confidence;
}

/**
 * Full pattern response from GET endpoints.
 */
export interface PatternResponse {
  id: string;
  paper_id: string;
  name: string;
  claim: string | null;
  evidence: string | null;
  context: string | null;
  tags: string[];
  custom_fields: Record<string, unknown> | null;
  status: PatternStatus;
  created_at: string;
  updated_at: string;
}

/**
 * Pattern response with paper details (from GET /api/patterns/{id}).
 */
export interface PatternDetailResponse extends PatternResponse {
  paper_title: string;
}

/**
 * Single pattern review decision.
 */
export interface PatternDecision {
  temp_id: string;
  approved: boolean;
}

/**
 * Response from pattern review submission.
 */
export interface PatternReviewResponse {
  saved_count: number;
  rejected_count: number;
  patterns: PatternResponse[];
}

/**
 * Request body for updating a pattern.
 */
export interface PatternUpdateRequest {
  name?: string;
  claim?: string;
  evidence?: string;
  context?: string;
  tags?: string[];
}

/**
 * Response from listing patterns.
 */
export interface PatternListResponse {
  patterns: PatternResponse[];
}

// ============================================================================
// Query Types
// ============================================================================

/**
 * Reference to a pattern used in a response.
 */
export interface SourceReference {
  pattern_id: string;
  summary: string | null;
  relevance_score: number | null;
}

/**
 * Response from query endpoint.
 */
export interface QueryResponse {
  session_id: string;
  response: string;
  sources: SourceReference[];
  confidence: Confidence;
  follow_up_suggestions: string[];
}

// ============================================================================
// Error Types
// ============================================================================

/**
 * API error response.
 */
export interface ApiError {
  code: string;
  message: string;
}

/**
 * Error with full response details.
 */
export class ApiRequestError extends Error {
  code: string;
  status: number;

  constructor(code: string, message: string, status: number) {
    super(message);
    this.name = "ApiRequestError";
    this.code = code;
    this.status = status;
  }
}
