# Prior-Art & Novelty Assessment: Polyhedral Cardinality for Cube-Query Planning

> **Purpose.** A companion to [`exact-cardinality-optimizer.md`](./exact-cardinality-optimizer.md).
> That document proposes wiring `isl`/`barvinok` lattice-point counting into the
> optimizer as an *exact* cardinality-and-cost oracle for cube/array queries. This
> document records an **adversarial prior-art / novelty check** on that idea —
> written to *refute* the claim, not confirm it — plus an assessment of whether the
> work is publishable and what related work must be cited.

> **The claim under test.** *"Nobody has wired `isl`/`barvinok` (or Ehrhart /
> Barvinok / polyhedral lattice-point counting generally) into a database or array
> query optimizer as an exact cardinality-and-cost oracle for cube/array queries —
> with tiling as the unifying primitive for both chunk-I/O cost and memory-bounded
> streaming."*

> **Verdict: CONFIRMED WHITE SPACE** across academic literature, array-DBMS cost
> models, the TileDB primary source, and the reachable patent record. No
> counter-example surfaced. The caveats are all variants of one honest limit: the
> claim is a *universal negative* and cannot be exhaustively proven against
> paywalled venues, very recent preprints, and non-US patent filings.

---

## 1. Method

The check was run as an **adversarial gap-hunt**, not a topic survey: each search
thread was tasked with *finding the paper or patent that kills the claim*. A claim
of novelty is proven by absence, and absence is only credible if the searches
designed to break it come back empty.

Three passes, each independent:

1. **Deep-research workflow** — 5 search angles → parallel web search → fetch 20
   sources → extract 82 falsifiable claims → 3-vote adversarial verification of
   the top 25 (a claim needs 2/3 refutations to be killed). **All 25 survived
   3-0; none refuted.**
2. **TileDB primary source** — the one array-store named but not read directly in
   the workflow (VLDB 2017, *The TileDB Array Data Storage Manager*), fetched and
   full-text-searched.
3. **Patent sweep** — Google Patents / Justia / USPTO full-text (reachable);
   Espacenet / WIPO / Lens (blocked — see blind spots).

The decisive test for every near-miss: does the work count lattice/integer points
for **memory footprint / data movement / cache behavior** (well-trodden compiler
use — *not* the claim) or for **query cardinality / plan selection / cost** (the
claim)?

---

## 2. Findings — three buckets

### (a) Exact prior art that kills the claim — **NONE**

No source (paper or patent) counts lattice points for query cardinality or plan
selection. Every instance of the exact tooling (`isl`, `barvinok`, Ehrhart,
Barvinok's algorithm) lands on the compiler-memory side of the line.

The most dangerous title found — **US 8,185,519 B2, "Techniques for exact
cardinality query optimization"** — obtains "exact" cardinalities by *executing* a
chosen subset of query expressions (execution feedback + weighted set-cover), not
by counting integer points. Unrelated to the claim on inspection of the claims.

### (b) Adjacent work that must be cited and distinguished

The related-work spine splits cleanly into two literatures that never cite each
other, plus the motivation anchor:

**Compiler / polyhedral side — "same math, different target."**

- **Clauss, ICS 1996** — *Counting Solutions to Linear and Nonlinear Constraints
  Through Ehrhart Polynomials*. The **first** CS application of Ehrhart
  polynomials; motivating uses are counting cache lines, memory locations, and
  parallelism for loop nests. This is the canonical near-miss and the *origin* of
  the technique. (Ehrhart introduced the polynomials in 1962; Clauss introduced
  their algorithmic use in program analysis.)
- **Verdoolaege — `isl` / `barvinok`.** `barvinok`'s `card` op returns the exact
  count of integer points in a parametric Presburger set as a piecewise
  quasi-polynomial (Ehrhart) — *this is precisely the engine the proposal wants*.
  Yet the author's own documentation (isl paper; barvinok manual; 2022 Cerebras
  overview *Polyhedral Compilation and the Integer Set Library*) frames it
  **exclusively** for polyhedral compilation — iteration domains, scheduling,
  dataflow, AST generation — with *zero* occurrences of "database" or "query."
  This is the cleanest single piece of evidence that the crossover is unmade.
- **Ding group (Rochester), 2026 preprints** — *Fully Symbolic Analysis of Loop
  Locality* (arXiv:2603.10196) and *AutoLALA* (arXiv:2604.05066) use `isl` +
  Barvinok (via Rust bindings) for cache-miss counts and reuse-distance over
  matmul / stencil / einsum kernels. These are the strongest *"is someone doing
  this right now?"* candidates — active development of the exact tooling, pointed
  at **memory behavior, not queries**. Cite to preempt "isn't this just what the
  compiler people already do?"
- **Bhaskaracharya et al. (NVIDIA), Nov 2025** — *Modeling Layout Abstractions
  Using Integer Set Relations* (arXiv:2511.10374). Uses `isl` integer-set
  **relations** + Presburger arithmetic to formalize tensor→GPU-memory layouts
  (CUTLASS lineage). Same `isl` substrate, target is memory **layout**, not query
  cardinality — the freshest evidence the tooling is live. Its use of `isl_map`
  for layout is conceptually adjacent to this engine's nD→2D flatten layer (an
  affine access relation), so cite it to pre-empt "isn't your flattening just a
  layout relation?" — the answer being: same algebra, but they never *count* or
  drive a cost-based plan.
- **Reservoir Labs polyhedral-compiler patents** (e.g. US 9,830,133 communication
  optimization from a polyhedral representation; US 10,180,828) — the patent-side
  analog: they count/manipulate integer points in parametric Z-polytopes and use
  tiling, but for compiler loop transformation and footprint/locality, never a
  query optimizer. The narrowing risk to acknowledge; does not read on query
  cardinality.

**Database / array-DBMS side — bespoke arithmetic or runtime measurement, never polyhedral counting.**

- **SciDB** (Stonebraker et al., *The Architecture of SciDB*, SSDBM 2011) —
  **explicitly rejects** closed-form analytical cardinality estimation ("wildly
  inaccurate" over cascaded array ops) in favor of an incremental *runtime*
  strategy: execute a sub-tree, measure the actual result size ("perfect
  estimate"), plan the next. **This doubles as the proposal's motivation hook** —
  the flagship array DB conceded the analytical approach and punted to runtime;
  the proposal revives that approach made *exact* by lattice counting. (Scoped to
  the 2011 architecture; post-2011 Paradigm4 internals unverified.)
- **rasdaman** (Baumann, SIGMOD 1998; Baumann et al. survey, *J. Big Data*
  2020/21) — cost model is tile-count / spatial-index arithmetic (tiles read,
  tile location, operation count). Full-text search of the survey: 0 hits for
  Ehrhart / Barvinok / polyhedra / lattice-point / isl.
- **ArrayStore** (Soroush et al., SIGMOD 2011) — a storage *manager*
  (chunking, R-tree/kd-tree overlap via explicit range-intersection), not a
  cost-based optimizer. No cardinality estimation.
- **TileDB** (Papadopoulos et al., VLDB 2017) — **verified against the primary
  source.** A storage *manager*, no query optimizer, no cardinality/cost
  estimator. Subarray reads use bespoke *global-cell-order* range arithmetic plus
  MBR (minimum bounding rectangle) tile pruning. Full-text search: 0 hits for
  Ehrhart / Barvinok / polyhedra / lattice-point / Presburger / isl — and also 0
  for "cardinality," "cost model," "selectivity," "estimate." Critically it moves
  *opposite* to an exact oracle: for sparse arrays *"the result size may be
  unpredictable... TileDB gracefully handles buffer overflow."* Cite as related
  array-storage work (MBR-based tile lookup), distinguish as having neither a
  cost-based optimizer nor exact cardinality counting.
- **Leis et al., VLDB 2015** (*How Good Are Query Optimizers, Really?*) — the
  motivation anchor: cardinality-estimation error, compounding through joins, is
  *the* dominant source of bad plans. Establishes *why* an exact oracle is
  valuable, but never connects to polyhedral counting; it is about relational
  join cardinality, not gridded/array data.

### (c) Confirmed white space

Dense array/cube queries have exactly the affine, rectangular index structure that
Barvinok counts **exactly and cheaply** (polynomial in fixed, low dimension) — yet
no array DBMS uses it, and the polyhedral community never pointed its counting
machinery at query planning. The unclaimed intersection is: **`isl`/`barvinok` as
the exact cardinality-and-cost engine for a cube-query optimizer, with tiling as
the one primitive unifying chunk-I/O cost and memory-bounded streaming.**

---

## 3. Blind spots (state these in any write-up)

The verdict rests on a universal negative and cannot be exhaustively proven.

- **Paywalled venues and very recent preprints** were not exhaustively reachable.
  The two 2026 Ding-group preprints show this exact tooling is under active
  development, so a DB-facing application could emerge (or be in flight).
- **Some DB-side negatives are argument-from-silence** (e.g., "rasdaman does not
  use polyhedral counting" is inferred from its enumerated cost drivers). The
  *affirmative* cores (tile arithmetic, runtime measurement) are directly quoted,
  so the thesis holds, but the absence is inferential.
- **Patent search is partial, not clearance-grade.** WebSearch proxied Google
  Patents / USPTO full-text well, but the Google Patents JS search UI,
  **Espacenet, WIPO PATENTSCOPE, and Lens.org were not directly searchable** — no
  independent coverage of EP / PCT / CN / JP filings. No CPC/IPC classification
  search (e.g. G06F16/2453) was run. Keyword brittleness: a filing describing this
  as "geometric counting of grid cells" or "affine constraint volume" would evade
  every query.
- **SciDB findings scoped to 2011**; closed-source post-2011 internals unverified.

---

## 4. Publishability assessment

- **arXiv itself** is a preprint server — any coherent, non-crank write-up in
  `cs.DB` / `cs.PL` / `cs.DC` clears the bar (needs an endorsement for a first
  submission). The literal question is uninteresting; the real question is whether
  it's a *paper* a reviewer finds novel and substantiated.
- **As a vision / position paper** (a CIDR-style venue, or a scientific-data /
  array-data workshop): close to ready as-is. Those venues reward exactly this
  cross-pollination framing. Lowest-effort real publication.
- **As a full systems paper** (VLDB / SIGMOD / ICDE): not yet — it is an idea, not
  a result. Needs (1) an implementation (at least Blocks 0–4 of the design note,
  ideally through the Tier-B `barvinok` path so the coupled/non-box cases where
  the novelty lives are actually exercised), and (2) an empirical comparison of
  the exact-cardinality planner vs. an estimation baseline on real cube workloads
  (ERA5/ONI), showing plan-quality wins *and* that barvinok's counting stays cheap
  at 3–5 dimensions (the fixed-dimension-is-polynomial claim must be measured, not
  asserted).

**Central claim to make:** *cardinality estimation — the dominant source of bad
plans — is replaced by exact lattice-point counting for the coordinate-structured
fragment of a cube query, and the same tiling primitive derives both I/O cost and
streaming granularity; here is a system and measurements showing it.*

**Before committing the novelty claim in print:** for *clearance-grade* certainty,
run a classification-based search on Espacenet / Lens (CPC G06F16/2453) or a
professional patent database — outside what web tools can reach.

---

## 5. Patent vs. publish — timing and the disclosure clock

> **Not legal advice.** For anything you actually file, use a registered patent
> practitioner. This note captures the tradeoff so a decision isn't made by
> accident (e.g. by posting to arXiv first).

**The core conflict: publishing can destroy patent rights.** A public disclosure —
an arXiv preprint, a talk, a blog post, even a public repo describing the
invention — starts the clock:

- **United States:** a **12-month grace period** from the inventor's own first
  public disclosure to file. Miss it and the disclosure becomes prior art against
  your own application.
- **Most other countries (EU, China, Japan, India, …):** **absolute novelty** —
  *any* pre-filing public disclosure can bar patentability outright, with no grace
  period. So an arXiv post today can forfeit non-US rights immediately.

**Rule of thumb: if a patent is seriously on the table, file before you disclose.**
The cheap way to preserve optionality is a **provisional / priority filing** (US
provisional, or an Indian application) *before* arXiv/submission — it locks a
priority date, costs relatively little, and gives 12 months to decide whether to
pursue full prosecution or just publish.

**India-specific ordering** (from §4 and the Section 39 discussion): an
India-resident inventor must either **file in India first and wait 6 weeks**, or
obtain a **Foreign Filing License** — *before* any foreign filing. So the safe
sequence is: Indian priority filing (or FFL) → then US/PCT → then public
disclosure (arXiv, conference).

**Recommended decision order for this work:**

1. **Decide patent-or-not first** — and get a proper novelty + **non-obviousness**
   opinion. The prior-art check here is encouraging but *not clearance-grade*, and
   patentability needs non-obviousness, not just novelty. The same "two disjoint
   literatures" framing that makes the idea publishable is exactly the argument an
   examiner uses for obviousness (obvious combination of known Barvinok counting +
   known query cost models) — so this needs a real assessment, not optimism.
2. **If pursuing a patent:** file the priority application (Indian filing / FFL,
   then US provisional or PCT) **before** posting to arXiv or submitting anywhere.
3. **If publishing only:** arXiv/submit freely — but know that doing so likely
   forecloses non-US patents immediately and starts the US 12-month clock.

**Vision-paper note:** the §4 recommendation (ship a workshop/position paper) and a
patent are not mutually exclusive *if* the filing precedes the paper. If timing
forces a choice and the goal is research credit + a strong novelty story, the
publish path is lower-friction; the patent path is only worth the cost and delay if
there's a commercialization or defensive-portfolio reason.

---

## 6. Sources

**Compiler / polyhedral**
- Clauss, *Counting Solutions to Linear and Nonlinear Constraints Through Ehrhart Polynomials*, ICS 1996.
- Verdoolaege, *isl: An Integer Set Library for the Polyhedral Model*, ICMS 2010.
- Verdoolaege, *barvinok* User Guide — https://barvinok.sourceforge.io/barvinok.pdf
- Verdoolaege, *Polyhedral Compilation and the Integer Set Library*, Cerebras / trends-in-arithmetic-theories 2022.
- *Fully Symbolic Analysis of Loop Locality*, arXiv:2603.10196 (2026).
- *AutoLALA: Automatic Loop Algebraic Locality Analysis for AI and HPC Kernels*, arXiv:2604.05066 (2026).
- Bhaskaracharya, Acharya, Hagedorn, Grover (NVIDIA), *Modeling Layout Abstractions Using Integer Set Relations*, arXiv:2511.10374 (2025).
- Reservoir Labs: US 9,830,133; US 10,180,828.

**Database / array-DBMS**
- Leis et al., *How Good Are Query Optimizers, Really?*, PVLDB 9(3), 2015 — https://www.vldb.org/pvldb/vol9/p204-leis.pdf
- Stonebraker et al., *The Architecture of SciDB*, SSDBM 2011.
- Baumann, rasdaman, SIGMOD 1998 (DOI 10.1145/276305.276386); Baumann et al. survey, *J. Big Data* 2020/21.
- Soroush et al., *ArrayStore*, SIGMOD 2011.
- Papadopoulos et al., *The TileDB Array Data Storage Manager*, VLDB 2017.
- US 8,185,519 B2 — *Techniques for exact cardinality query optimization* (execution-feedback, not lattice counting).
