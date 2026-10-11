export interface Researcher {
  id: string;
  name: string;
  institutions: string[];
  topics: string[];
  fields: string[];
  works_count: number;
  cited_by_count: number;
  orcid: string | null;
  source_url: string;
}

export interface ResearcherSearchResult {
  researchers: Researcher[];
  total: number;
  resolved_filters: Record<string, string>;
}

export interface ResearcherProfile extends Researcher {
  publications: {
    id: string;
    title: string;
    year: number | null;
    citation_count: number;
    topic: string | null;
    source_url: string;
  }[];
  coauthors: { id: string; name: string }[];
}
