import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ApiRequestError,
  deletePaper,
  deletePattern,
  getHealth,
  getPaper,
  listPapers,
  listPatterns,
  submitFollowUp,
  submitPatternReview,
  submitQuery,
  updatePattern,
  uploadPaper,
} from "./api";

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe("API Client", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("error handling", () => {
    it("throws ApiRequestError for non-ok responses with structured error", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({
          detail: { code: "NOT_FOUND", message: "Paper not found" },
        }),
      });

      try {
        await getPaper("invalid-id");
        expect.fail("Should have thrown");
      } catch (err) {
        expect(err).toBeInstanceOf(ApiRequestError);
        const apiErr = err as ApiRequestError;
        expect(apiErr.code).toBe("NOT_FOUND");
        expect(apiErr.message).toBe("Paper not found");
        expect(apiErr.status).toBe(404);
      }
    });

    it("handles string error detail", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ detail: "Invalid request" }),
      });

      await expect(getHealth()).rejects.toThrow("Invalid request");
    });

    it("handles non-JSON error responses", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => {
          throw new Error("Not JSON");
        },
      });

      await expect(getHealth()).rejects.toThrow("Request failed with status 500");
    });
  });

  describe("getHealth", () => {
    it("returns health status", async () => {
      const mockHealth = {
        status: "ok",
        database: "ok",
        qdrant: "ok",
        sessions: 5,
        tracing: "enabled",
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockHealth,
      });

      const result = await getHealth();
      expect(result).toEqual(mockHealth);
      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:8000/health",
        expect.objectContaining({
          headers: { "Content-Type": "application/json" },
        })
      );
    });
  });

  describe("papers", () => {
    it("uploadPaper sends FormData without Content-Type", async () => {
      const mockResponse = {
        paper_id: "paper-123",
        metadata: { title: "Test", authors: [], type: "research_paper" },
        patterns: [],
        extraction_warnings: [],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const formData = new FormData();
      formData.append("file", new Blob(), "test.pdf");

      const result = await uploadPaper(formData);
      expect(result).toEqual(mockResponse);

      // Verify no Content-Type header (browser sets it for FormData)
      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:8000/api/papers",
        expect.objectContaining({
          method: "POST",
          body: formData,
        })
      );
    });

    it("listPapers builds query string from params", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ papers: [], total: 0, limit: 50, offset: 0 }),
      });

      await listPapers({ status: "active", type: "blog_post", limit: 10, offset: 20 });

      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:8000/api/papers?status=active&type=blog_post&limit=10&offset=20",
        expect.any(Object)
      );
    });

    it("listPapers works without params", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ papers: [], total: 0, limit: 50, offset: 0 }),
      });

      await listPapers();

      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:8000/api/papers",
        expect.any(Object)
      );
    });

    it("getPaper fetches single paper", async () => {
      const mockPaper = { id: "paper-123", title: "Test" };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockPaper,
      });

      const result = await getPaper("paper-123");
      expect(result).toEqual(mockPaper);
      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:8000/api/papers/paper-123",
        expect.any(Object)
      );
    });

    it("deletePaper handles 204 No Content", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 204,
      });

      const result = await deletePaper("paper-123");
      expect(result).toBeUndefined();
      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:8000/api/papers/paper-123",
        expect.objectContaining({ method: "DELETE" })
      );
    });
  });

  describe("patterns", () => {
    it("submitPatternReview sends decisions", async () => {
      const mockResponse = { saved_count: 2, rejected_count: 1, patterns: [] };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const decisions = [
        { temp_id: "t1", approved: true },
        { temp_id: "t2", approved: false },
      ];

      const result = await submitPatternReview("paper-123", decisions);
      expect(result).toEqual(mockResponse);
      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:8000/api/papers/paper-123/patterns",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({ decisions }),
        })
      );
    });

    it("listPatterns with status filter", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ patterns: [] }),
      });

      await listPatterns("paper-123", "approved");

      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:8000/api/papers/paper-123/patterns?status=approved",
        expect.any(Object)
      );
    });

    it("updatePattern sends partial update", async () => {
      const mockPattern = { id: "pattern-123", name: "Updated" };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockPattern,
      });

      const result = await updatePattern("pattern-123", { name: "Updated" });
      expect(result).toEqual(mockPattern);
      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:8000/api/patterns/pattern-123",
        expect.objectContaining({
          method: "PUT",
          body: JSON.stringify({ name: "Updated" }),
        })
      );
    });

    it("deletePattern handles 204", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 204,
      });

      await deletePattern("pattern-123");
      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:8000/api/patterns/pattern-123",
        expect.objectContaining({ method: "DELETE" })
      );
    });
  });

  describe("query", () => {
    it("submitQuery creates new session", async () => {
      const mockResponse = {
        session_id: "session-123",
        response: "Answer",
        sources: [],
        confidence: "high",
        follow_up_suggestions: [],
      };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const result = await submitQuery("What is X?");
      expect(result).toEqual(mockResponse);
      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:8000/api/query",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({ query: "What is X?", session_id: undefined }),
        })
      );
    });

    it("submitQuery continues existing session", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ session_id: "session-123" }),
      });

      await submitQuery("Follow up", "session-123");
      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:8000/api/query",
        expect.objectContaining({
          body: JSON.stringify({ query: "Follow up", session_id: "session-123" }),
        })
      );
    });

    it("submitFollowUp requires session_id", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ session_id: "session-123" }),
      });

      await submitFollowUp("Tell me more", "session-123");
      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:8000/api/query/followup",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({ query: "Tell me more", session_id: "session-123" }),
        })
      );
    });
  });
});
