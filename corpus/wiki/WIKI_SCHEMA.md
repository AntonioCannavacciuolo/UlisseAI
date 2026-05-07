# LLM Wiki Schema

## Purpose
You are the maintainer of this Wiki. Your goal is to build a structured, interlinked collection of markdown files that represent the long-term memory of Ulisse. This is a compounding artifact where knowledge is synthesized, not just indexed.

## Structure
- **Pages**: Stored in `corpus/wiki/pages/`. Each page is a markdown file.
- **Index**: `corpus/wiki/index.md` - A content-oriented catalog organized by category.
- **Log**: `corpus/wiki/log.md` - A chronological record of all ingestions and updates.

## Operations
1. **Ingest**: When User shares important information, projects, facts, or insights, you must integrate them into the Wiki.
   - Extract key information.
   - Update existing entity/concept pages or create new ones.
   - Note contradictions or reinforcements.
   - Use cross-references: `[[Page Name]]`.
2. **Maintenance**: Periodically check for stale claims or missing links.
3. **Synthesis**: Combine information from multiple conversations into cohesive topic summaries.

## Formatting
- Use `# Title` for the main heading.
- Use `## Section` for sub-headings.
- Use `[[Page Title]]` for internal wiki links.
- Every page should have a "Sources" section at the bottom referencing the session ID or raw data.

## Goal
The user handles the sourcing and exploration; you handle the bookkeeping, summarizing, and cross-referencing.
