# AstroMedica — RAG + RL codebase

AstroMedica turns a Vedic birth chart into a short health-screening chat. The
app computes which organs/diseases the chart points to, a chat asks the patient
about matching symptoms, and an RL model (the Medical Report Enhancer) turns
the confirmed findings into a symptom-level report that keeps learning from
every finished chat.

This README is written so someone can work on the **RL side (e.g. a Siamese
network)** without any other context. Section 12 is the Siamese hand-off; section 14 is a glossary of terms.

> ### Start here
>
> **The app in plain words:** you enter your birth date, time and place. The
> app draws your birth chart and, using rules from Vedic astrology, guesses
> which 3 parts of the body may be weak (for example the bladder). It then
> chats with you in everyday language — "do you get up at night to pass
> urine?" — never naming a disease. Your yes/no answers show which guess was
> right. Finally a small AI model turns the confirmed findings into a list of
> likely symptoms, and it re-trains every hour on all the chats so far.
>
> **What to read, depending on your task:**
>
> | If you want to… | Read |
> |---|---|
> | Run the app on your PC | §1 |
> | Understand the whole system in 10 minutes | §2 (flow) → §5 (worked example) |
> | Work on the RL / Siamese network | §2 → §8 (data formats) → §10 (RL) → §12 (Siamese) |
> | Understand the astrology and the Combined Algo | §14 (glossary) → §3 → §5 |
> | Understand how the chat asks and scores | §4 → §5 |
> | Find a specific file | §6 (file map) and §7 (patient folder) |
>
> **Words you'll see everywhere:** *organ cluster* (one of the 3 guessed body
> areas), *branch* (a symptom theme inside it), *Combined Algo* (the vote of 5
> astrology algorithms that picks the 3 clusters), *RL* (the learning model).
> Everything else is in the glossary (§14).
>
> **Assumed knowledge:** basic Python, JSON and a terminal. No astrology knowledge needed.

- Stack: Python 3.13, Flask (port 5000), PyTorch (CPU), Google Firestore,
  Clerk (login), Gemini API (chat + embeddings), Prokerala API (charts).
- Everything runs in **one Flask process**: web app, chat, RAG, RL worker and
  the hourly RL training scheduler.

---

## 1. Get the code and run it

### 1.1 What you need first

- **Git** — https://git-scm.com/downloads
- **Python 3.13** (3.11+ works) — `python --version`
- **Google Cloud CLI** — https://cloud.google.com/sdk/docs/install (for Firestore)
- **Access** to the private GitHub repo (ask to be added to the
  `Ameya-Innovations` organisation) and to the Google Cloud project
  `astrohealth-509609` (role *Cloud Datastore User*).
- About 3 GB free disk (PyTorch is most of it).

### 1.2 Clone the repository

```bash
git clone https://github.com/Ameya-Innovations/Astrohealth.git
cd Astrohealth
```

The repo is private: the first time, Git opens a browser window to sign in to
GitHub (or asks for a username + personal access token). "Repository not found"
means your account hasn't been given access yet.

### 1.3 Install

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Linux / macOS / the VM**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch, much smaller
pip install -r requirements.txt
```

### 1.4 Configure

1. Copy `.env.example` to `.env` and fill it in (Clerk keys, Firestore
   project — see the table in §1.6). Get the values from the team; `.env` is
   gitignored, never commit or share it.
2. Log in to Google for Firestore (once per machine; Workspace accounts need
   this again roughly every day):

   ```bash
   gcloud auth application-default login
   gcloud auth application-default set-quota-project astrohealth-509609
   ```

   On the VM skip this — its service account is used automatically.
3. API keys for Prokerala / Gemini / Serper are currently in `config.py`
   (being moved to `.env`) — get them from the team.

### 1.5 Run

```powershell
.\.venv\Scripts\python.exe server.py        # Windows — always the venv's Python (it has torch)
```

```bash
python server.py                              # Linux / macOS with the venv activated
```

Open **http://localhost:5000** (not a 192.168.x.x address — Clerk login only
works on localhost or HTTPS). Stop with **Ctrl+C**; on Windows, if it doesn't
stop (background RL threads), close the terminal window.

Quick checks once it's running:

| Check | Expected |
|---|---|
| http://localhost:5000/health | `{"status": "ok"}` |
| http://localhost:5000/rl | the RL dashboard |
| http://localhost:5000/healthz | RL config, training and scheduler status (JSON) |
| Home page → Generate a chart | console shows `[DB] Saved chart for patient …` |

Healthy startup looks like:

```
[Auth] Clerk login: ON
[DB] Main DB: Firestore project astrohealth-509609
[DB] Firestore connection OK
[RL] Lookup tables loaded from firestore (symptom library: 45 diseases)
[Firestore] RL connected to astrohealth-509609: symptom_library (45), organ_aliases (25), branch_hints (44), chat_maps (2), stop_words (106), organ_branch_families (9), chat_logs (N)
| INFO | training | Loaded CustomANN weights from ...rl_enhancer\weights\customann_ppo.pt
[RL] Enhancer ready — model: CustomANN (PPO-trained), training: V2 (learns from chat logs)
[RL Training] Scheduler on - first cycle in 5 min, then every 60 min
```

### 1.6 `.env` settings

| Variable | Meaning |
|---|---|
| `AUTH_ENABLED` | `true` = Clerk sign-in required; `false` = everything runs as user `local-dev` |
| `CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY` | Clerk dev instance keys |
| `ADMIN_CLERK_USER_IDS` | comma-separated Clerk user ids that can see every patient |
| `FRONTEND_URL` | `http://localhost:5000` — Clerk tokens from other origins are rejected |
| `FIRESTORE_BACKEND=firestore`, `FIRESTORE_PROJECT_ID` | main DB **and** RL data. Without them the app DB is in-memory (RL still needs Firestore) |
| `TRAIN_INTERVAL_SECONDS` (3600), `TRAIN_LEADIN_SECONDS` (300), `TRAIN_SCHEDULER_ENABLED` (true) | RL training scheduler |

API keys for Prokerala / Gemini / Serper are still hard-coded in `config.py`
(to be moved to env vars) — don't share that file.

### 1.7 Sending your changes back (Git workflow)

```bash
git checkout -b feature/siamese-network      # one branch per piece of work
# ...edit code, test...
git status                                   # check nothing private is listed
git add rl_enhancer/siamese.py rl_integration.py
git commit -m "Add Siamese similarity scoring"
git push -u origin feature/siamese-network
```

Then open a **Pull Request** on GitHub into `main` for review. Pull the
latest changes before starting new work: `git checkout main && git pull`.

**Never commit:** `.env`, `patients/` (real names and birth details),
`.venv/`, `rl_enhancer/data/`, service-account `.json` keys, or API keys in code.
If `git status` shows any of these, stop and fix `.gitignore` first.

---

## 2. Data flow — from birth details to RL output

### 2.1 The steps, in order

| # | Step | Code | Input → output |
|---|---|---|---|
| 1 | Patient fills the birth form and presses Generate | `astrohealth-landing.html` → `POST /generate-chart` | name, DOB, time, place, lat/lng, gender |
| 2 | Fetch the birth chart | `jataka_fetch.fetch_and_save` | Prokerala API → planets, houses, nakshatras, dasha. New patient id `0NN` (same birth details reuse the old id) |
| 3 | Rule 1: find the disease-causing planets | `rules.py` (via `jataka_logic.run_logic`) | chart → P1…P7 + PM sources (§3.1) |
| 4 | Run the 5 algorithms | `jataka_logic.run_logic` → the five engines (§3.2) | sources → each algorithm's ranked organs/diseases |
| 5 | Combined Algo | `algorithm_comparison.build_combined_algo_summary` (§3.3) | 5 rankings → **top 3 organs, 3 diseases each** |
| 6 | Cache + de-identify | `orchestrator.handle_chart_generation`, `deidentification` | name/DOB go to the identity vault; AI side keeps only the patient id |
| 7 | Create the RL input file | `combine_output.write_rl_report` | top 3 organs → `For RL Report.json` (`conditions`, empty `chat_log`) |
| 8 | Save to the main DB | `db_layer.save_chart_generation` | inputs + all algorithm outputs → Firestore `users/{uid}/…` |
| 9 | Show results, patient starts the chat | `results.html` → `POST /start-consultation` | Combined Algo top 3 → chat plan: 3 clusters × several symptom branches (§4.1) |
| 10 | Each question | `chat_module` + `rag_pipeline` + Gemini | runtime state + RAG context → next question JSON |
| 11 | Each answer | `POST /send-message` → `symptom_mapper.classify_answer`, `chat_module.update_cluster_support` | answer text → label `yes_strong / yes_but / no_but / no_strong` → cluster/branch status (§4.2–4.3). `chat_log.json` and `For RL Report.json` updated |
| 12 | Chat ends | `chat_module` (turn 15 or early finish, §4.4) | `completed = true` → RL job queued |
| 13 | RL enhancement | `rl_integration._run_job` → `rl_enhancer` | report + this chat + the patient's older chats (Firestore) → `RL Output.json` (§10.1) |
| 14 | Final report | `POST /combine-output` → `combine_output.run_combine` | chat + Combined Algo → `final_output.json`, 0–10 disease scores on the page (§4.5) |
| 15 | Learning (hourly) | `training2` via the scheduler | symptom library + every stored chat → new model weights (§10.2) |

### 2.2 Where the data sits after each step

| After step | Disk (`patients/<id>/`) | Firestore | Memory |
|---|---|---|---|
| 2 | `profile.json`, `raw_chart.json`, `raw_dasha.json`, `shadbala.json` | — | — |
| 4–5 | `processed_logic.json`, `chart_data.json`, `diagnosis.json`, `algorithm_comparison.json` | — | orchestrator chart cache |
| 6–8 | `For RL Report.json` | `users/{uid}/inputs, prokerala_results, algorithm_results, combined_results, patients, identity_vault` | — |
| 10–11 | `chat_log.json`, `For RL Report.json` (chat_log refreshed) | — | orchestrator chat session |
| 13 | `RL Output.json` | `chat_logs/{patient}__{started_at}` | RL job store |
| 14 | `final_output.json` | — | — |
| 15 | `rl_enhancer/weights/*.pt` | `symptom_library` (append-only enrichment) | live model |

### 2.3 The same flow as a picture

```
 birth form ─► Prokerala ─► Rule 1 (P1…P7, PM) ─► 5 algorithms ─► Combined Algo top-3 organs
                                                                        │
                       ┌────────────────────────────────────────────────┤
                       ▼                                                ▼
              For RL Report.json                                   chat plan (3 clusters)
              (conditions)                                              │  Gemini asks, patient answers,
                       ▲                                                │  answers labelled yes/no,
                       └──────── chat_log refreshed every turn ◄────────┘  clusters confirmed/rejected
                                                                        │ chat ends
                                                                        ▼
                         RL job: save chat to Firestore ─► merge older chats ─► enhancer ─► RL Output.json
                                                                        │
                         final_output.json + 0–10 scores on results.html
                         hourly: retrain on library + all chats ─► new weights
```

---

## 3. The astrology engine: 5 algorithms and the Combined Algo

> New to astrology terms (house, lord, rashi, nakshatra, drishti)? Read the
> glossary in §14 first — it takes two minutes. **The short version:** the
> chart is a map of where each planet was at birth. The **6th house** of that
> map is about illness, so the planets connected to it are treated as
> "disease-causing". Every planet, star (nakshatra) and sign (rashi) has a
> traditional list of body parts and diseases, and the algorithms add these up.

### 3.1 Shared input — "Rule 1" sources

Every algorithm starts from the same planets, picked from the **6th house**
(the house of disease) by `rules.apply_rule_1`:

| Code | Source | Notes |
|---|---|---|
| **P1** | Planet(s) sitting in the 6th house (Rogkaraka 1) | if none, the 11th-house occupants take this role |
| **P2** | Dispositor of the 6th lord (Rogkaraka 3) | the lord of the sign the 6th lord sits in |
| **P3** | Planets aspecting (drishti on) the 6th house | |
| **P4** | Lord of the 6th house | |
| **P5** | Ascendant (Lagna) lord | |
| **P6** | Sign (rashi) of the 6th house | used as an organ "anchor" |
| **P7** | Ascendant sign | used as an organ "anchor" |
| **PM** | Conditional Moon | only when the Moon is linked to the above |

Each planet gives three **layers** of terms from `knowledge_base.py`:
**planet layer** (`PLANET_DISEASES`), **nakshatra layer**
(`NAKSHATRA_DISEASES` for its nakshatra + pada — the most specific) and
**rashi layer** (`RASHI_ORGANS` for the sign it sits in). Terms are mapped to
body systems with `term_to_systems.TERM_TO_SYSTEMS` and grouped for display
with `MERGE_GROUPS` (e.g. "Kidney / Urinary System"). Terms that don't apply to
the patient's gender are dropped. A "+5% confirmation" in several algorithms
comes from `eleventh_lord_confirmation.py`: if the 11th lord's nakshatra
diseases match a found disease, that score is raised by 5%.

### 3.2 The five algorithms (short)

**1. System Priority — HitList** (`hitlist_engine.py`, output `processed["hitlist"]["top4"]`)
Scores body systems from all three layers of every source.

```
term_weight  = SOURCE_WEIGHT[P] × LAYER_MULT[layer] × (1 / number of systems the term maps to)
system_score = Σ term_weight                         (over all sources, layers, terms)
             − P1 coherence penalty                  (a P1 planet hitting a system in only 2 / 1 of its
                                                      3 layers keeps 80% / 50% of that P1 score)
final        = system_score × CONVERGENCE[#layers]   (3 layers agree ×2.0, 2 layers ×1.4, 1 layer ×1.0)
group score  = Σ final over the systems in a display group  → ranked, top 4 groups
SOURCE_WEIGHT: P1 8, P2 4, P3 2.5, P4 2, P5 1.5, P7 3, PM 5/4/3.5/3
LAYER_MULT:    planet 1.5, nakshatra 6, rashi 7
```

*In plain words:* every disease/organ word linked to a disease-causing planet
gives points to the body systems it belongs to. Important planets (P1) and
specific layers (nakshatra, rashi) give more points; a vague word that fits
many systems gives less to each. A system pointed at by all three layers is
doubled. The 4 highest-scoring body-system groups win.

**2. Rashi Priority Correlation** (`rashi_correlation_engine.py`, `top3`)
Only counts evidence that agrees with the **6th-house sign** or **ascendant sign** organs.

```
for each evidence term (P1–P5, PM; layers planet/nakshatra/own rashi) and each anchor organ term
sharing a body system with it:
  score = PRIORITY[P] × LAYER[layer] × ANCHOR[P6|P7] × match_strength × (1 + 0.2 × 1/#shared systems)
  match_strength = 1.35 for the exact same term, else 1.0 + min(0.25, 0.08 × #shared systems)
best match per term is added to its system group; broad overlaps (skin/sensory/systemic, or blood/
lymph outside the nakshatra layer) don't count unless exact; +5% confirmation; top 3 groups.
PRIORITY: P1 6, P2 5, P3 3, P4 2.5, P5 2, PM 6/4/3/2.5 · LAYER: planet 0.85, nakshatra 1.25,
own rashi 0.8 · ANCHOR: P6 3.0, P7 1.5
```

*In plain words:* the 6th-house sign and the rising sign each name some body
parts. A planet's evidence only counts if it points at one of those same body
parts — like a second witness confirming the first. Exact word matches count
most; loose matches through generic systems (skin, senses) are ignored.

**3. Organ Truth Correlation** (`organ_truth_correlation_engine.py`, `top_organs`)
Starts from the anchor organs and only lets planets strengthen those organs.

```
organ starts with:  6th-house sign organs +6, ascendant sign organs +3, Moon sign organs +2
each source organ that is already an anchor organ:  + SOURCE[P] × LAYER[layer]
each nakshatra disease → its organs (disease→organ table) that are anchors:  + SOURCE[P] × 2
(+ the disease is listed under that organ);  +5% confirmation;  top 8 organs with their diseases
SOURCE: P1 6, P2 4.5, P3 3.2, P4 3, P5 2, PM 3 · LAYER: planet 1, nakshatra 2, own rashi 1
(when there is no P1, the P2 rashi organs count directly, ×1.5)
```

*In plain words:* start with the organs named by the 6th-house sign and the
rising sign. Planets can only add points to those organs, never introduce new
ones. The result is a ranked list of organs, each with the diseases that point
at it.

**4. Disease First Logic** (`disease_first_logic.py`, `top_diseases`)
Starts from each source's nakshatra **diseases** and asks which organs back them up.

```
disease score = SOURCE[P]                                   for each source naming the disease
              + PLANET_LAYER[P] for each disease organ also among the planet's own organs
              + RASHI_LAYER[P_RASHI] for each disease organ among the source sign's organs
              + 6th sign 3 / ascendant 1.5 / Moon sign 1.5 for each disease organ in those signs
              + 5% confirmation                              → top 8 diseases with supporting organs
SOURCE: P1 6, P2 4.5, P3 3.5, P4 2.8, PM 2.5, P5 2 · PLANET_LAYER: P1 3, P2 2.2, P3 1.5, P4 1.2,
PM 1.5, P5 1.2 · RASHI_LAYER: P1 3, P2 2
```

*In plain words:* the other way round — start from the diseases each planet's
star (nakshatra) names, then give a disease extra points for every other part
of the chart that points at the same organ. The best-supported diseases win.

**5. Disease Compare Logic** (`disease_compare_logic.py`, `top_diseases`)
Same scoring as Disease First, but a source disease also counts for **similarly
worded** diseases in the master list (exact, substring, text similarity ≥ 0.9,
≥ 0.82 with a shared medical word, or ≥ 0.72 with 75% word overlap), provided
they share an organ. Catches spelling/wording variants of the same disease.

### 3.3 Combined Algo — the median consensus (`algorithm_comparison.build_combined_algo_summary`)

This is what the chat, RAG and RL use. For every organ mentioned by any algorithm:

```
1. Organ names are normalised (heart/blood → "Heart / Blood", liver/stomach → "Digestive / Liver", …).

2. Rank score — an organ found in an algorithm's item at rank r (1 = that algorithm's top item),
   where raw = that item's own score:
       S(r, raw) = min( 100,  max(35, 115 − 15·r)  +  min(10, 2·log10(|raw| + 1)) )
       r = 1 → 100   r = 2 → 85 + boost   r = 3 → 70 + boost   r = 4 → 55 + boost   r ≥ 6 → 35 + boost
       In plain words: an algorithm's 1st place = 100 points, each lower place loses 15
       (never below 35), plus up to 10 bonus points for a big raw score.
   For HitList/Rashi the organs come from the group label, matched organs and trail;
   for Organ Truth from the organ itself; for the two disease algorithms from each disease's
   supporting organs. Per algorithm the organ keeps its best (max) score; 0 if not mentioned.

3. consensus_score = MEDIAN of the five algorithm scores (sorted, middle = 3rd value)
   algorithm_support_count = number of algorithms with score > 0
   In plain words: line up the 5 scores from low to high and take the middle one —
   like a jury vote, one very excited or silent algorithm can't decide alone.

4. Drop body regions that aren't organs (back, abdomen, face, feet …).

5. Rank organs by (consensus_score, support_count, sum of the 5 scores), descending → TOP 3.

6. Diseases per organ: each mention adds that item's raw score; sorted by
   (number of algorithms naming it, summed score) → top 3 diseases per organ.
```

The median means an organ needs **at least 3 of the 5 algorithms** behind it to
score well — one algorithm alone can't push it up. Combined Algo **v2**
(`algorithm_comparison_v2.py`) only feeds the unused "New Priority Engine" page.

---

## 4. How the chat decides

### 4.1 Chat plan

`combined_algo_clusters` turns the Combined Algo top 3 organs into **clusters**
(rank 1–3). `symptom_mapper` splits each cluster's diseases into **branches** —
symptom themes such as `kidney`, `prostate`, `micturition`, `piles` — each with
lay-language **screening**, **drilldown** and **context** questions. The
patient never hears organ or disease names.

### 4.2 One turn

1. Gemini gets the system prompt (rules), the runtime state (cluster/branch
   scores) and RAG context, and returns JSON:
   `{chat_message, reasoning, cluster_rank, question_type, branch_id}`.
   The backend checks `branch_id` exists; if Gemini fails it uses
   `fallback_question_meta` (fixed order: cluster 1 screening → drilldown → next clusters).
2. The patient answers. `symptom_mapper.classify_answer` labels it (first rule that matches wins):

| Label | Matches | Cluster points |
|---|---|---|
| `no_strong` | "never", "not at all", "absolutely not", or the word "no" | −3 |
| `no_but` | "not really", "rarely", "hardly", "only once", "no but" | −1 |
| `yes_but` | "sometimes", "mild", "a little", "occasionally", "yes but" | +1 |
| `yes_strong` | "definitely", "daily", "often", "a lot", the word "yes", or a symptom word (pain, swelling, tired…) | +3 |

### 4.3 Branch and cluster status (`chat_module.update_cluster_support`)

```
branch:  confirmed   ≥ 2 yes_strong, or 1 yes_strong + ≥ 2 yes_but
         rejected    ≥ 2 no_strong,  or 1 no_strong + ≥ 2 no_but
         drilldown   any yes_but, or yes_strong mixed with a no
         suspected   only yes_strong so far
         screening   only no answers so far / not asked
cluster: confirmed if any branch is confirmed; rejected if ≥ min(2, #branches) branches are
         rejected and none is suspected; drilldown if a branch is suspected/drilldown;
         suspected if any yes; else screening.
```

*In plain words:* two clear "yes" answers about the same symptom theme confirm
it; two clear "no" answers rule it out; a mix means "ask more". A body area
counts as confirmed as soon as one of its symptom themes is confirmed.

Once a cluster is confirmed, the chat screens **every unasked branch** of that
cluster, then asks severity/context follow-ups.

### 4.4 When the chat ends

- At turn **15**, or
- **early**: from turn 6, when a confirmed cluster has no unasked branch left
  and its main branch has ≥ 2 follow-up answers (severity / red flag / context).

Then `completed = true`, the RL job is queued and the closing message is shown.

### 4.5 The 0–10 scores on the results page (`combine_output.build_confirmed_findings`)

```
organ    = the confirmed cluster with the most yes_strong answers (then highest score)
diseases = all Combined Algo diseases of that organ (asked or not)
score    = average answer points × 10, over the answers on that disease's branch
           points: yes_strong 1.0, yes_but 0.5, no_but 0, no_strong 0
           no answers on its branch → "Not asked"
```

*In plain words:* for each disease, look at the questions asked about its
symptoms. If all were answered "yes" the score is 10; all "no" gives 0; a
"yes, but" counts as half a yes. Example: yes, yes-but, no → (1 + 0.5 + 0) / 3 × 10 = **5.0**.

---

## 5. Worked example — patient 002 (test chat, de-identified)

**Rule 1:** 6th house sign Vrishabha · 6th lord Venus · ascendant lord Jupiter ·
no planet in the 6th (so the 11th-house fallback applies) · P2 = Saturn ·
drishti on the 6th from Moon, Mars, Saturn.

**The five algorithms (top results):**

| Algorithm | Top results (score) |
|---|---|
| HitList | Kidney / Urinary 548.05 · Skin / Immunity 360 · Digestive / Liver 356.23 · Heart / Blood 261.28 |
| Rashi correlation | Digestive / Liver 497.62 · Reproductive 294.03 · Respiratory / Lungs 231.06 |
| Organ truth | Blood 140.44 · Flesh 54.14 · Bladder 50.19 · Urinary tracts 50.19 |
| Disease first | Prostate gland enlargement 22.0 · Brain fever 21.0 · Leprosy 20.9 · Constipation 19.32 |
| Disease compare | Fracture in foot 22.57 · Prostate gland enlargement 22.0 · Brain fever 21.0 · Constipation 19.32 |

**Combined Algo for Bladder:** algorithm scores HitList 100, Rashi 0, Organ
truth 73.42, Disease first 100, Disease compare 87.72 → sorted
`[0, 73.42, 87.72, 100, 100]` → **median 87.72**, support 4.

| Rank | Organ | Consensus | Support | Top diseases (# algorithms naming it) |
|---|---|---|---|---|
| 1 | Bladder | 87.72 | 4 | Urinary trouble (4), Prostate gland enlargement (4), Frequent micturition (3) |
| 2 | Genitals | 87.72 | 3 | Prostate gland enlargement (3), Problems in kidney & bladder (1), Excess of urine and kidney problem (1) |
| 3 | Heart | 87.68 | 4 | Leprosy (3), Brain fever (3), Dropsy (2) |

Bladder beats Genitals on the tie because it has more algorithms supporting it.

**`For RL Report.json` → `conditions`:** `[{"Bladder": […]}, {"Genitals": […]}, {"Heart": […]}]`.

**Chat** (the tester answered "yes" to everything):

| Turn | Question (lay wording) | Cluster · type · branch | Answer → label |
|---|---|---|---|
| 1 | swelling around the ankles or puffiness around the eyes? | 1 · screening · kidney | yes → yes_strong |
| 2 | is the urine stream weaker or slower than before? | 1 · screening · prostate | yes → yes_strong |
| 3 | passing urine much more often during the day? | 1 · screening · micturition | yes → yes_strong |
| 4 | does the urge come on suddenly? | 1 · drilldown · micturition | yes → yes_strong |
| 5 | is this affecting work, travel or sleep? | 1 · severity · micturition | yes → yes_strong |
| … | follow-ups until the early-finish rule triggers | | |

Result: cluster 1 (Bladder) **confirmed** — 7 × yes_strong × 3 = score 21;
clusters 2 and 3 never asked (status screening).

**Results page:** Bladder — Urinary trouble 10/10, Prostate gland enlargement
10/10, Frequent micturition 10/10 (every answer on their branches was yes).

**RL:** the chat is saved to Firestore `chat_logs`, merged with any earlier
chats for 002, and `RL Output.json` lists symptoms per disease with an evidence
block (`endorsed` = the questions answered yes). At the next hourly training
the endorsed questions become new (disease → symptom) training pairs.

---

## 6. Folder and file map

### 6.1 Root — web server, chat, RAG, glue

| File | What it does |
|---|---|
| `server.py` | Flask app: all page and API routes, Gemini call (`call_gemini`, JSON mode, retries), Prokerala token/current-positions, registers the RL blueprint, calls `init_rl()` on start. Loads `.env` first. |
| `orchestrator.py` | Single coordination layer: owns chat sessions (in memory), patient chart cache (Combined Algo + priority ladder), de-identification, RAG call, safety prompt. |
| `chat_module.py` | The consultation chat ("Module B"): builds the chat plan from Combined Algo clusters, system prompt rules, parses Gemini JSON, classifies answers (`yes_strong / yes_but / no_but / no_strong`), tracks cluster/branch scores, decides when to finish, saves turns, syncs the RL report, queues RL on completion. |
| `combine_output.py` | "Module C": final report. Also writes **`For RL Report.json`** (`write_rl_report`, `ensure_rl_report`, `refresh_rl_report_chat`) and builds the 0–10 confirmed findings (`build_confirmed_findings`, `ANSWER_POINTS`). |
| `combined_algo_clusters.py` | Combined Algo v1 top organs reshaped into the cluster format the chat understands. This is the chat plan source. |
| `symptom_mapper.py` | Turns organ/disease names into lay symptom questions (the patient never hears disease names). |
| `rag_pipeline.py` | RAG steps 1, 2, 3A, 4: query builder, Gemini embedding (gemini-embedding-001, 768-dim), retrieval via `db_layer.query_similar_chunks` (stub, no knowledge base yet), prompt augmentation. |
| `llm1_search.py` | RAG step 3B "LLM1" web search (Serper — key currently returns 403; planned switch to Gemini grounding). |
| `deidentification.py` | Splits name/DOB/place/parents into the identity vault; AI/RAG/RL see only an HMAC token or patient_id. |
| `db_layer.py` | The only place that talks to the main DB: identity vault, chart saves (`save_chart_generation`), RAG `kb_chunks` (stub). |
| `app_auth.py` | Glue to the login/DB code in `app_core/`: `login_required`, `check_patient_access`, admin list, `AUTH_ENABLED` switch, Firestore error handling (fails in ~10 s with a clear message). |
| `rl_integration.py` | Runs the RL enhancer inside Flask (replaces the standalone FastAPI `main.py`): `/rl` page, `/api/...` routes, worker thread, patient RL jobs, Firestore logs, training start + hourly scheduler. |
| `config.py` | API keys and model names (Gemini model, embedding model/dim). |
| `jataka_fetch.py` / `jataka_logic.py` | Module A: Prokerala fetch + patient folder/ID allocation (`001`, `002` …) / runs the astrology rule set. |
| `rules.py`, `knowledge_base.py`, `term_to_systems.py`, `parhora_table.py` | Astrology rules and lookup tables (planets/rashis/nakshatras → organs/diseases). |
| `hitlist_engine*.py`, `rashi_correlation_engine*.py`, `organ_truth_correlation_engine*.py`, `disease_*_logic*.py`, `*_weighted_disease_priority.py`, `priority_cascade.py`, `algorithm_comparison*.py`, `zone_engine.py`, `second_ascendant.py`, `imp_rashi_distance.py`, `dasha_chat_priority.py`, `eleventh_lord_confirmation.py`, `related_house_severity_engine.py`, `rashi_truth_*.py`, `common_findings_engine.py`, `chart_specific_overrides.py` | The individual astrology algorithms whose outputs feed the Combined Algo. `*_v2` versions only feed the unused "New Priority Engine" page. |
| `disease_diagnose.py`, `chat_ai_context_builder.py` | Diagnosis file and chat context file per patient. |
| `auth.py`, `login.html`, `seed_users.py` | Old email/password login — **no longer used** (Clerk replaced it). |
| `test_*.py`, `verify_fix.py`, `check_dasha.py`, `dict_dump.py`, `temp_kb_output.py`, `parse_excel.py`, `*_out.txt` | Old scripts and outputs from earlier development. |

### 6.2 Front end

| File | Page |
|---|---|
| `astrohealth-landing.html` | Home page: birth form, recent patients list, Sign in button |
| `results.html` | **The final patient page**: chart, analysis (Public/Dev view toggle), chat panel, "What Our Conversation Showed" |
| `static/session.js` | Clerk sign-in; adds the login token to every same-origin `fetch` |
| `priority6.html`, `results_staged.html`, `print_report.*` | Unused / legacy pages |

### 6.3 `app_core/` — login + main DB (from the Google-Cloud version of the app)

| File | Role |
|---|---|
| `auth.py` | Clerk token verification (issuer + subject checks) |
| `http.py` | `require_auth`, `enforce_patient_access` (patient must belong to the user) |
| `repository.py` | Firestore (or in-memory) store, every doc under `users/{internal_user_id}/…`; 10 s timeout per call |
| `persistence.py` | `save_generation`: writes inputs, Prokerala result, one doc per algorithm, combined record, patient ownership |
| `settings.py`, `runtime.py`, `errors.py`, `validation.py`, `identifiers.py`, `ratelimit.py`, `logging_config.py` | Settings from env, service wiring, error types, input validation, ids, rate limits, JSON logs |

### 6.4 `rl_enhancer/` — Abhishek's Medical Report Enhancer (RL)

Merged in with **only import paths changed** (relative imports) plus marked
`Patched:` fixes in `config_db.py`. Replace files here on update.

| File | Role |
|---|---|
| `parser.py` | Validates the input wrapper `{patient_id, conditions:[{Organ:[diseases]}], chat_log:[…]}` |
| `chat_log_parser.py` | Turns chat-log documents into a clinical summary: organs, diseases, `symptom_evidence {question: {status, question_type, branch}}`, endorsed/denied lists; merges a patient's sessions (`persist_and_merge_chat_logs`, `parse_chat_log_list`) |
| `enhancer.py` | `build_output`: per organ → disease → symptom list + evidence block. `symptoms_for_disease` asks the model (`training2.generate_symptoms`) first, then the symptom library + endorsed chat evidence, then a generic fallback. Evidence scoring uses keyword overlap / Jaccard / rarity weights |
| `training2.py` | **V2 (used)**: CustomANN language model (4 transformer blocks, 192-dim, 6 heads, max 96 tokens). Corpus = symptom library ∪ chat-log pairs (disease → endorsed questions). Pretraining (cross-entropy, 20 epochs) then PPO-style fine-tuning (20 epochs, reward = token recall vs. reference symptoms + length/redundancy terms, KL to a frozen reference). Also API enrichment and `generate_symptoms` |
| `training.py` | V1 (library only) — kept, not used |
| `config_db.py` | Firestore access for RL: seeding, loaders, append-only symptom library (`ArrayUnion`), chat-log upsert/load/delete. Every call has a 10 s limit (`CALL`) |
| `config_loader.py` | In-memory cache of the lookup tables, reloaded from Firestore every 30 min (falls back to `bootstrap_data.py`) |
| `bootstrap_data.py` | Built-in starting values: 45-disease symptom library, organ aliases, branch hints, organ/disease regex maps, stop words, organ→branch families |
| `medical_api.py` | MedlinePlus + Wikipedia symptom fetch for enrichment (cached in `data/api_cache/`) |
| `jobs.py` | Thread-safe in-memory job store (enhancement + training jobs) |
| `weights/` | `customann_pretrained.pt`, `customann_ppo.pt` — overwritten by every training cycle |
| `templates/index.html`, `static/` | The `/rl` page |
| `data/` | `api_cache/`, `uploads/`, `outputs/` (manual uploads from the `/rl` page) |

### 6.5 Other folders

| Folder | Contents |
|---|---|
| `patients/<id>/` | One folder per patient (see §7). **Contains real names and birth details — never commit or share.** |
| `charts_cache/` | Cached Prokerala responses (same birth details → no new API call) |
| `_db_stub/` | JSON stand-in for DB tables when there's no request user (scripts) |
| `.venv/` | Python environment (not in git) |

---

## 7. Files in each patient folder

| File | Written by | Contents |
|---|---|---|
| `profile.json`, `raw_chart.json`, `raw_dasha.json`, `shadbala.json` | jataka_fetch | Birth details + Prokerala data |
| `processed_logic.json`, `chart_data.json`, `diagnosis.json`, `algorithm_comparison.json` | generate-chart | Algorithm outputs (Combined Algo is computed from `processed_logic.json`) |
| `chat_log.json` | chat_module | Every turn of every session (appends) |
| **`For RL Report.json`** | combine_output | **RL input** — created at Generate, `chat_log` refreshed every turn |
| **`RL Output.json`** | rl_integration | **RL output** for the latest finished chat (admin/dev only) |
| `final_output.json` | combine_output | Final report + confirmed findings |
| `run_history.json`, `chat_ai_context.json` | various | History / chat context |

---

## 8. Data formats the RL works with

### 8.1 RL input — `patients/<id>/For RL Report.json`

```json
{
  "patient_id": "002",
  "conditions": [
    {"Bladder": ["Urinary trouble", "Diabetes", "Kidney stone"]},
    {"Neck":    ["...", "..."]},
    {"Throat":  ["...", "..."]}
  ],
  "chat_log": [ <one chat session, see 8.2> ]
}
```

`conditions` = the **Combined Algo top-3 organ clusters** and their diseases
(disease names are Vedic/astrological terms, e.g. "Urinary trouble",
"Dropsy", "Leprosy" — most are *not* in the 45-disease bootstrap library).
No patient name is ever sent to RL.

### 8.2 Chat session (inline in the report, and stored in Firestore `chat_logs`)

```json
{
  "patient_id": "002",
  "session_id": "session_1791277445040",
  "started_at": "2026-10-06T14:34:11.491723",
  "completed": true,
  "turns": [
    {"turn": 1, "role": "assistant", "timestamp": "...",
     "message": "{\"chat_message\": \"Have you noticed ...?\", \"reasoning\": \"...\", ...}",
     "metadata": {"cluster_rank": 1, "question_type": "screening", "branch_id": "kidney",
                  "session_id": "...", "backend_reasoning": {...}}},
    {"turn": 1, "role": "user", "message": "yes, mostly at night",
     "metadata": {"answer_result": "yes_strong", "answered_branch_id": "kidney",
                  "answered_cluster_rank": 1, "answered_question_type": "screening",
                  "session_id": "..."}}
  ],
  "hitlist_cluster_scores": {
    "1": {"rank": 1, "system": "...", "status": "confirmed|suspected|drilldown|...",
          "score": 0, "yes_strong": 0, "yes_but": 0, "no_but": 0, "no_strong": 0,
          "confirmed_organs": [], "possible_organs": [], "confirmed_diseases": [],
          "possible_diseases": [], "dominant_branch": "kidney",
          "branches": {"kidney": {"id": "kidney", "label": "...", "asked_count": 2, "status": "...",
                                  "yes_strong": 1, "...": "...", "possible_diseases": []}},
          "answers": []},
    "2": {}, "3": {}
  }
}
```

- `answer_result` per user turn: **`yes_strong`, `yes_but`, `no_but`, `no_strong`** —
  this is the label signal. `question_type`: `screening`, `drilldown`, `context`.
- The assistant `message` is the raw Gemini JSON; the question text is its
  `chat_message` field (`chat_log_parser._assistant_question` extracts it).
- Firestore doc id: `{patient_id}__{started_at}` (slugged), so re-saving the
  same session overwrites instead of duplicating.

### 8.3 RL output — `patients/<id>/RL Output.json`

```json
{
  "patient_id": "002", "generated_at": "...", "model": "CustomANN (PPO-trained)",
  "lookup_tables": "firestore", "chat_session_id": "session_...", "chat_started_at": "...",
  "chat_completed": true, "chat_logs_used": 1,
  "enhanced": [
    {"Bladder": [
      {"Urinary Trouble": ["symptom 1", "symptom 2", "..."],
       "evidence": {"confidence": "high|medium|low", "organ": "Bladder",
                    "match_method": "branch_or_keyword", "match_score": 0.7, "matched_by": [],
                    "endorsed": ["questions answered yes"], "denied": ["questions answered no"]}}
    ]}
  ]
}
```

`enhanced` is exactly `enhancer.build_output(...)`; the wrapper fields are
added by `rl_integration._save_patient_output`.

---

## 9. Firestore layout (project `astrohealth-509609`, database `(default)`)

```
clerk_index/{clerk_user_id}                     login → internal user id
users/{internal_user_id}                        app data, one subtree per user
    inputs/  chart_inputs/  prokerala_results/  algorithm_results/  combined_results/
    patients/{patient_id}                       "this patient belongs to this user" (hidden: true after a correction)
    account/chart_allowance                     {charts_used, active_patient_id, history} — 1 chart + 1 correction
    chart_sources/{patient_id}                  raw Prokerala chart + dasha (JSON text), profile, planet positions
                                                of generation day, algo_version — reopen/re-run without Prokerala
    identity_vault/{patient_id}                 name, DOB, place (PII)
    audit_events/                               denied access attempts

# RL (shared, not per user) — rl_enhancer/config_db.py
symptom_library/{disease_slug}                  {disease, symptoms[]}   append-only
organ_aliases/{alias_slug}                      {alias, canonical}
branch_hints/{disease_slug}                     {disease, branch}
chat_maps/organ_map | disease_map               {kind, map | list of {pattern, canonical}}
stop_words/{word}                               {word}
organ_branch_families/{organ_slug}              {organ, branches[]}
chat_logs/{patient_id}__{started_at}            one finished/ongoing chat session (8.2)
```

Rules: browsers never access Firestore directly (`firestore.rules` denies
all); only the server does. RL collections must not hold names or birth details.

---

## 10. RL subsystem in detail

### 10.1 When the RL runs

1. Chat completes → `chat_module.queue_rl(id)` → `rl_integration.queue_patient_rl(id)` creates a job.
2. Worker thread (`rl_integration._worker_loop` → `_run_job`):
   - `parser.parse_input(For RL Report.json)`
   - `chat_log_parser.persist_and_merge_chat_logs(patient_id, inline)` — saves this
     session to Firestore and merges it with the patient's older sessions
   - `chat_log_parser.parse_chat_log_list(...)` → summary
   - `enhancer.build_output(conditions, chat_summary=summary)`
   - written to `patients/<id>/RL Output.json`
3. Console: `[RL] Chat completed …` → `[Firestore] Saved chat …` → `[RL] Job … done … chat logs used: N`.

### 10.2 Training (V2, `training2.py`)

- Started by the scheduler in `rl_integration` (first cycle 5 min after start,
  then hourly) or manually from `/rl` / `POST /api/training/pretrain|ppo`.
- Cycle: **API enrichment** (append-only to `symptom_library`) → **corpus**
  (library + `_chat_log_corpus()` from all Firestore chat logs) →
  **pretrain** → **PPO**. Weights saved to `rl_enhancer/weights/`; the live
  model is swapped in-process (`training2.STATE`).
- Busy flag `STATE.busy` prevents overlapping jobs.
- Logs: `[RL Training] …` (ours) and `| INFO | training | Corpus built: N samples (45 library + M chat-log)` (his).

### 10.3 RL HTTP endpoints (`rl_integration.py`, no login yet — admin/dev only)

| Route | Purpose |
|---|---|
| `GET /rl` | His RL dashboard page |
| `POST /api/upload` | Run the enhancer on an uploaded JSON (same format as 8.1) |
| `GET /api/jobs`, `/api/jobs/<id>`, `/api/jobs/<id>/result`, `/download`, `POST /cancel` | Job status/results |
| `POST /api/chat_logs`, `/api/chat_logs/batch` | Store chat logs |
| `GET /api/chat_logs/<patient_id>` | List a patient's stored sessions |
| `DELETE /api/chat_logs/<patient_id>/<session_id>` | Delete one session (use for test chats) |
| `GET /api/training/status`, `POST /api/training/<pretrain|ppo>` | Training |
| `GET/POST /api/scheduler/status|config|run-now` | Scheduler |
| `GET /api/config/status`, `POST /api/config/reload` | Lookup-table cache |
| `GET /api/symptom_library/<disease>` | One disease's symptoms |
| `GET /healthz` | Health + config + training + scheduler state |

---

## 11. Login, roles, privacy (short)

- Clerk login (`AUTH_ENABLED=true`). Patients only see charts they generated;
  ids in `ADMIN_CLERK_USER_IDS` see all. Role handling is planned to move into
  a `role` field on `users/{id}`.
- RL output is clinical and **admin/dev only** — never on the patient page.
- RL never receives patient names; chat logs carry only `patient_id`.

### 11.1 One chart per account (`chart_access.py`)

So nobody can resell readings from one account:

- A signed-in (non-admin) account can **generate 2 charts in total**: its first
  chart + **one correction**. Chat is unlimited. Admins and login-off testing
  are unlimited. Accounts without an allowance record start fresh.
- `/generate-chart` checks the allowance **before** calling Prokerala and
  returns `403 {"code": "chart_limit"}` when it's used up.
- Home page: an account with a chart sees "Your birth chart is ready —
  continue your health chat" instead of the form (`GET /my-chart` tells the page).
- Results page: "Typed your birth details wrong? → Correct my birth details"
  below the chat (only on the active chart, until used) → confirmation →
  `/?correct=1` shows the form once. The previous chart is then hidden from
  the user (`patients/{id}.hidden`), kept for admins with its chats.

### 11.2 Reopening a chart and algorithm updates

- Opening a chart (`/patient-data/<id>` → `server._prepare_patient_chart`)
  loads the account's saved chart from Firestore `chart_sources/{id}` into
  `patients/<id>/` — **no Prokerala call**. Console:
  `[DB] Patient 015 chart loaded from Firestore: users/…/chart_sources/015 (raw chart + dasha, 92 KB)`.
- `chart_access.ALGO_VERSION` — **bump it whenever the astrology algorithms
  change.** On the next open, results with an older version are re-run on the
  saved chart (`[Algo] Patient 015: results v1 -> v2 - re-running …`); the old
  results stay as `processed_logic.v1.json` for admins. Re-runs don't count
  against the allowance. "Current planet positions" stay those of the day the
  chart was made.

---

## 12. Siamese network — hand-off notes

### 12.1 Goal options (pick with the team)

A Siamese network = one shared encoder applied to two texts, trained so
matching pairs land close and non-matching pairs far apart (contrastive /
triplet loss). Where it can help here:

| Use | Replaces / improves | Where in code |
|---|---|---|
| **a. Patient ↔ disease scoring** — similarity between what the patient endorsed and each candidate disease's symptom profile | keyword overlap in `enhancer._evidence_for` / `_question_score`, and the yes-count 0–10 score | `enhancer.py` (new score in the evidence block) |
| **b. Filter generated symptoms** — keep CustomANN outputs that are close to the disease's real symptoms | today `symptoms_for_disease` trusts `generate_symptoms` if it returns ≥2 fragments (gives wrong symptoms for Vedic names) | `enhancer.symptoms_for_disease` |
| **c. PPO reward** — semantic similarity instead of token recall | `training2._reward_for_generation` | `training2.py` |
| **d. Synonym matching** — "frequent urination" = "micturition" | `_canonical_question`, `chat_log_parser._canonical_disease` regexes | `enhancer.py`, `chat_log_parser.py` |

### 12.2 Training data already available

- **Positive pairs:** two symptoms of the same disease (`symptom_library`, 45
  diseases, ~5–6 each, growing via enrichment); an endorsed question
  (`answer_result` = `yes_strong`/`yes_but`) + the cluster's disease/branch;
  MedlinePlus/Wikipedia phrases (`data/api_cache/`).
- **Negatives:** symptoms of other diseases; questions answered
  `no_strong`/`no_but` for that disease/branch.
- **Volume is small** (a handful of real chat logs so far) → start from a
  pretrained text encoder (CustomANN blocks, a small open model such as
  MiniLM on CPU, or Gemini embeddings — 768-dim, already used in
  `rag_pipeline.embed_query` (async), but an API call per text) and train only a small
  projection head.
- Training text is English; the Vedic disease names in `conditions` are the
  hard case — most have no library entry.

### 12.3 Suggested shape (fits the existing code)

- New file `rl_enhancer/siamese.py`: `SiameseEncoder`, `train_siamese(job_id)`,
  `similarity(a, b) -> float`, `load_siamese_weights()`; weights in
  `rl_enhancer/weights/siamese.pt`.
- Training: add a third step to the cycle after PPO in
  `rl_integration._run_one_training_cycle` (same `STORE` job + `STATE.busy`
  pattern as `training2.start_pretraining`).
- Loading: call `load_siamese_weights()` in `rl_integration.init_rl()` next to
  `load_existing_weights()`.
- Output: add fields under each disease's `evidence` block in
  `enhancer.build_output` (e.g. `"similarity": 0.82`,
  `"filtered_symptoms": [...]`) — they flow into `RL Output.json` unchanged.
- Optional vector store: Firestore supports vector search (e.g. a
  `rl_disease_vectors` collection) if precomputed embeddings should persist.
- Keep it CPU-friendly (the VM has 4 vCPUs shared with the web app); cap
  torch threads during training.

### 12.4 Rules of the road

- Work inside `rl_enhancer/`; use relative imports (`from .enhancer import …`).
- Every Firestore call should pass `**config_db.CALL` (10 s limit) — without it
  an expired Google login hangs for 5 minutes.
- Don't put names/birth details in any RL collection or file.
- V2 learns from **every** stored chat — delete test chats afterwards:
  `DELETE /api/chat_logs/<patient_id>/<session_id>`.
- Test against a copy of a patient folder; don't overwrite real `patients/`
  data. Don't commit `.env`, `patients/`, `rl_enhancer/data/`, weights.
- Running standalone (without Flask): `from rl_enhancer import config_loader, training2` after
  `load_dotenv(".env")`; `config_loader.reload_all()` loads tables from Firestore.

---

## 13. Known open items

- LLM1 web search returns 403 (Serper) — plan: Gemini with Google Search grounding.
- RAG knowledge base (`kb_chunks`) is empty — retrieval returns 0 chunks.
- Combined Algo organ headings sometimes show the group name instead of the strongest organ.
- `001`-style patient ids: two users with identical birth details share one folder.
- `/rl` and `/api/*` are not behind login yet.
- API keys in `config.py` should move to environment variables.

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **Rashi** | Zodiac sign (12). Each rashi maps to body organs (`RASHI_ORGANS`), e.g. Vrishabha → face, throat, neck. |
| **Nakshatra / pada** | One of 27 lunar mansions, each split into 4 padas. Nakshatra + pada map to specific diseases (`NAKSHATRA_DISEASES`) — the most specific layer. |
| **House (bhava)** | 12 areas of the chart. The **6th house** is disease; the **11th** is used for confirmation and as a fallback. |
| **Lagna / ascendant** | The rising sign; its lord is P5, its sign is P7. |
| **Lord** | The planet that rules a sign/house (e.g. 6th lord = ruler of the 6th house's sign). |
| **Dispositor** | The lord of the sign a planet sits in (P2 = dispositor of the 6th lord). |
| **Drishti** | A planet's aspect (influence) on a house (P3 = planets aspecting the 6th). |
| **Rogkaraka** | "Disease-giving" planets found by Rule 1 (RK1 = 6th house occupants, RK3 = dispositor). |
| **Dasha** | Planetary time periods (Vimshottari); used for timing/forecast, not by the Combined Algo. |
| **Rule 1** | `rules.apply_rule_1` — picks the P1…P7 + PM sources every algorithm starts from (§3.1). |
| **Layer** | The three term sources per planet: planet diseases, nakshatra diseases, rashi organs. |
| **HitList** | Algorithm 1 — scores body systems from all layers; `top4` groups (§3.2). |
| **Convergence** | How many of the 3 layers point at the same system (HitList multiplier). |
| **Anchor** | The 6th-house sign and ascendant sign organs that Rashi/Organ-truth evidence must agree with. |
| **Confirmation boost** | +5% when the 11th lord's nakshatra diseases match a found disease. |
| **Combined Algo (v1)** | Median of the 5 algorithms' rank scores per organ → top 3 organs (§3.3). Drives the chat, RAG and RL. |
| **Consensus score / support count** | The median score / how many algorithms mention the organ. |
| **Cluster** | One of the 3 Combined Algo organs inside the chat (rank 1–3). |
| **Branch** | A symptom theme inside a cluster (`kidney`, `prostate`, `micturition` …) with its own questions. |
| **Screening / drilldown / context / severity** | Question types: first check / dig deeper after a yes / background / impact on daily life. |
| **yes_strong / yes_but / no_but / no_strong** | Answer labels (§4.2) — the main training signal for RL. |
| **Confirmed / suspected / drilldown / rejected / screening** | Branch and cluster statuses (§4.3). |
| **Chat plan source** | Where the chat's clusters come from — `combined_algo`. |
| **RAG** | Retrieval-augmented generation: build query → embed → retrieve knowledge → add to the prompt (`rag_pipeline.py`). |
| **LLM1 / LLM2** | LLM1 = web-search grounding (Step 3B, currently failing); LLM2 = Gemini that writes the chat questions. |
| **Orchestrator** | The coordination layer that owns sessions and calls chat, RAG, de-identification. |
| **De-identification / identity vault** | Name/DOB/place stored separately; AI, RAG and RL only see the patient id. |
| **For RL Report.json / RL Output.json** | RL input / RL output files in the patient folder (§8). |
| **Symptom library** | Disease → symptom list the RL uses (45 diseases at start, grows via API enrichment). |
| **Enrichment** | Append-only adding of MedlinePlus/Wikipedia symptoms to the library before training. |
| **CustomANN** | Abhishek's small transformer language model that generates symptom lists. |
| **Pretraining / PPO** | Training stage 1 (learn the corpus) / stage 2 (reinforcement fine-tuning with a reward). |
| **V1 / V2 training** | `training.py` (library only, unused) / `training2.py` (library + chat logs, used). |
| **Siamese network** | One shared encoder applied to two texts, trained so matching pairs score as similar (§12). |
