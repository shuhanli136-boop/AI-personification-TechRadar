# AI Personification in TechRadar Journalism

A corpus-based pipeline for classifying how AI is syntactically constructed
as a personified agent in UK technology journalism, using Systemic
Functional Linguistics (Transitivity).

## Overview

This repository contains the data-processing pipeline used to analyse a
year-long corpus of TechRadar (UK) articles (January 2025 – January 2026,
retrieved via the Lexis+ UK database), investigating the extent to which
AI is linguistically constructed as an agentive entity — one capable of
acting, thinking, or speaking — rather than as a passive tool.

Using Halliday and Matthiessen's (2014) Transitivity framework, every
sentence containing an explicit reference to AI (e.g. "AI", "artificial
intelligence", "ChatGPT", "generative AI", "large language model") is
classified according to the syntactic role AI occupies and, where
applicable, the semantic process type of the governing verb:

- **Level 1** — AI as Goal (passive subject or direct object, e.g. *"AI
  is used by researchers"*)
- **Level 2** — AI as Actor in a Material Process (e.g. *"AI detects
  fraud"*)
- **Level 3** — AI as Senser/Sayer in a Mental or Verbal Process (e.g.
  *"AI understands context"*, *"AI suggests..."*)

Relational, Existential, and Behavioural Process clauses (e.g. *"AI is
clever"*, *"AI exists"*) are excluded from the Level classification, as
they do not assign AI an active participant role.

## Research Questions

- **RQ1**: To what extent does AI occupy an active syntactic subject
  position, as opposed to an object or passive participant, in TechRadar's
  AI-related reporting?
- **RQ2**: Where AI does occupy an active role, to what degree is it
  constructed as a Material agent (Level 2) versus a Mental/Verbal agent
  (Level 3)?

## Results

Based on the final cleaned dataset of 1,531 sentences (330 unique verbs):

| Metric | Result |
|---|---|
| RQ1 — Active Subject | 57.35% |
| RQ1 — Object | 35.86% |
| RQ1 — Passive Subject (incl. by-agent) | 6.79% |
| RQ2 — Level 1 (Goal) | 40.95% |
| RQ2 — Level 2 (Material) | 44.61% |
| RQ2 — Level 3 (Mental/Verbal) | 14.44% |

## Repository Structure

```
├── ai_personification_pipeline.py      # Core pipeline: sentence segmentation,
│                                        # exclusion rules, syntactic role
│                                        # classification, verb extraction
├── full_year_batch_and_review.py       # Full-year batch processing and
│                                        # manual review merge workflow
├── step5_finalize.py                   # Verb-level mapping and summary export
├── data_cleaning_process_log.md        # Full methodological log of all
│                                        # cleaning decisions (Steps A–H),
│                                        # documenting how the pipeline
│                                        # arrives at RQ1/RQ2 results
└── verbs_level_final.xlsx              # Final verb-to-level mapping table
```

Raw corpus text files are **not included** in this repository due to
copyright restrictions on Lexis+ retrieved content. See [Data Availability](#data-availability)
below.

## Methodology Summary

The classification pipeline was developed iteratively; the full
step-by-step rationale for every cleaning decision is documented in
`data_cleaning_process_log.md`. Key stages include:

1. **Sentence segmentation and keyword matching** (spaCy), including
   correct identification of multi-word referents (e.g. "large language
   model") by extracting the syntactic head of the matched span
2. **Fragment-completeness detection** to address ellipsis markers
   introduced by keyword-based retrieval, which conflate genuine content
   omission with line-wrap artifacts
3. **Exclusion of non-agentive constructions**: modifiers ("AI-powered"),
   possessives ("AI's decision"), prepositional instruments ("using AI"),
   and reported speech
4. **Verb-extraction validity checks**, addressing parser errors caused
   by unconventional product naming (e.g. "Ryzen AI Max+") and
   part-of-speech mistagging (e.g. plural nouns misparsed as verbs)
5. **Exclusion of Relational, Existential, and Behavioural Processes**
   (e.g. *be*, *become*, *exist*, *behave*), including dynamic
   disambiguation of verbs with dual linking/lexical functions (e.g.
   *seem*, *look*, *feel*) based on the presence of an adjectival
   complement
6. **Light verb construction handling**: verbs such as *have*, *take*,
   *make*, *give*, *do* were re-analysed to identify their true governing
   predicate, typically expressed through a deverbal noun (e.g. "have a
   **solution**" leads to *solve*) or an infinitival complement following
   an abstract noun (e.g. "has the **ability** to **identify**" leads to
   *identify*)
7. **Manual verification** of automatically repaired sentence fragments
   and ambiguous verb classifications throughout, before final RQ1/RQ2
   statistics are computed

## Tools and Dependencies

- Python 3, pandas, spaCy (`en_core_web_sm`), matplotlib
- Google Colab + Google Drive (for persistent storage across sessions)
- Corpus retrieved from the Lexis+ UK database

## Data Availability

The original corpus was retrieved from the Lexis+ UK database under
institutional access and is not redistributed here due to copyright
restrictions. Researchers wishing to replicate this study should retrieve
a comparable corpus using the search terms specified in the Methodology
section of the accompanying dissertation, and may apply the pipeline in
this repository directly to their own corpus.

## Theoretical Framework

- Halliday, M. A. K., & Matthiessen, C. M. I. M. (2014). *Halliday's
  Introduction to Functional Grammar* (4th ed.). Routledge.
- Rayson, P., Archer, D., & Wilson, A. (2002). *Introduction to the USAS
  Category System*. UCREL, Lancaster University.

## Citation

If you use this pipeline or methodology in your own research, please cite
the accompanying dissertation (details to be added upon submission).

## License

Code in this repository is released for academic and research use. Please
contact the author before reuse in other contexts.
