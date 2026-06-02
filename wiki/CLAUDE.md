# CLAUDE.md — Project Wiki

<!-- STABLE PREFIX START — cache this section -->

## What this wiki is

A persistent, LLM-maintained knowledge base for this project.
You write and maintain all wiki pages. The human sources, directs, and asks questions.
Raw source documents are immutable — never modify files in `raw/sources/`.

## Directory structure

```
wiki/
  index.md          ← catalog of all pages (update on every change)
  log.md            ← append-only session log
  overview.md       ← project summary
  entities/         ← people, tools, services, components (one page each)
  concepts/         ← ideas, patterns, methods (one page each)
  decisions/        ← architecture decision records (ADRs)
  syntheses/        ← analyses, comparisons, conclusions
raw/
  sources/          ← immutable source documents
```

## Page format rules

### Entity page
```markdown
# [Name]
**Type:** person | org | tool | service | component
**Summary:** [2 sentences max]
## Key Facts
- [fact]
## Connections
- [[related]] — [how]
## Sources
- [title](../../raw/sources/file.md) — [date]
```

### Concept page
```markdown
# [Name]
**Definition:** [one sentence]
**Why it matters:** [one sentence]
## How it works
[2-3 sentences]
## Related
- [[concept]] — [relation]
## Sources
- [title](../../raw/sources/file.md) — [date]
```

### Decision record
```markdown
# Decision: [what]
**Date:** YYYY-MM-DD  **Status:** decided | revisiting | superseded
## Context
[why this decision was needed, 1-2 sentences]
## Options
| Option | Pro | Con |
|--------|-----|-----|
## Decision
[what + single-sentence why]
## Consequences
- [enables]
- [constrains]
```

<!-- STABLE PREFIX END -->

## Active operations

### Ingest (`/pith ingest <file>`)
1. Read source → extract entities, concepts, claims, contradictions
2. Show user what will be updated — wait for confirmation
3. Write/update pages → update index.md → append to log.md

### Query (`/pith wiki "<question>"`)
1. Read index.md → find relevant pages
2. Load relevant pages (Pith compresses each read)
3. Synthesize answer with citations
4. Offer to file useful answers as synthesis pages

### Lint (`/pith lint`)
Check for: orphan pages, stale claims, missing cross-references, entities without pages, broken links.

## Conventions

- All cross-links use `[[PageName]]` syntax
- Dates: YYYY-MM-DD
- File names: kebab-case for concepts, TitleCase for entities
- Every page has at least one source citation
- Never verbose: if structure works, don't use prose
