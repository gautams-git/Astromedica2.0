"""Executive and technical reports rendered from an actual benchmark bundle."""

from __future__ import annotations

from benchmark.architecture import ALGORITHM_MAP


def _cell_line(cell: dict | None) -> str:
    cell = cell or {}
    if not cell.get("ok"):
        return "failed: " + str(cell.get("error") or "unknown")
    top = (cell.get("rankings") or [])[:1]
    if not top:
        return "ok, empty ranking"
    return f"ok, top {top[0].get('label')} ({top[0].get('score')})"


def executive_report(bundle: dict) -> str:
    runs = bundle.get("runs") or []
    lines = [
        "# Executive benchmark report",
        "",
        "This report describes a forensic comparison of the existing algorithms. It does not change their formulas, and it does not claim clinical accuracy.",
        "",
        "## What was tested",
        "",
        f"{bundle.get('patient_count', len(runs))} synthetic patients were frozen and sent through three tracks:",
        "",
        "- Track A runs the original engine.",
        "- Track B runs the existing agent with engine-default weights and with the LLM switched off.",
        "- Track C, for Rashi Correlation only, may apply a validated weight proposal. The other four algorithms have no Track C. That gap is reported rather than filled in.",
        "",
        "When no live model key is configured, the Rashi weight proposal is a scripted selector. It is labelled as such and is not a live model result.",
        "",
        "## Matrix",
        "",
        "| Algorithm | Track A | Track B | Track C |",
        "|---|---|---|---|",
    ]
    for row in ALGORITHM_MAP:
        lines.append(
            f"| {row['algorithm']} | {'available' if row['track_a'] else 'gap'} | {'available' if row['track_b'] else 'gap'} | {'available' if row['track_c'] else 'not implemented'} |"
        )
    lines.extend(["", "## Patient outcomes", ""])
    lines.append("| Patient | Case | Rashi A | Rashi B | Rashi C | A vs B | A vs C |")
    lines.append("|---|---|---|---|---|---|---|")
    preserved = 0
    weight_moves = 0
    for run in runs:
        cells = run["cells"].get("Rashi Correlation", {})
        diff = run["differentials"].get("Rashi Correlation", {})
        ab = diff.get("A_vs_B", {})
        ac = diff.get("A_vs_C", {})
        if ab.get("status") == "equivalent":
            preserved += 1
        if ac.get("status") == "explained_weight_change":
            weight_moves += 1
        lines.append(
            f"| {run['patient_id']} | {run['case']} | {_cell_line(cells.get('A'))} | {_cell_line(cells.get('B'))} | {_cell_line(cells.get('C'))} | {ab.get('status', '')} | {ac.get('status', '')} |"
        )
    lines.extend([
        "",
        "## What changed",
        "",
        f"Rashi Track B matched Track A on {preserved} of {len(runs)} patients in this run (status `equivalent`). A match means the agent, with default weights and no LLM, preserved the original ranking on that input.",
        "",
        f"Rashi Track C showed an explained weight change on {weight_moves} of {len(runs)} patients. A weight change is not automatically an improvement. There is no adjudicated clinical label for these fixtures, so benefit is not claimed.",
        "",
        "## Limitations",
        "",
        "- The patients are synthetic. They exercise the code. They are not a clinical cohort.",
        "- Independent verification can only be strict where the arithmetic is reconstructable. Other algorithms are marked limited when their internals are not exposed.",
        "- A scripted weight selector is repeatable. A live model call would be statistically repeatable only, and only when a key is configured.",
        "- Learned weights from the pilot search are not written into the engines.",
        "",
    ])
    return "\n".join(lines)


def technical_report(bundle: dict) -> str:
    lines = [
        "# Technical forensic report",
        "",
        f"Commit: `{bundle.get('source_commit', '')}`",
        f"Created: {bundle.get('created_at', '')}",
        "",
        "## Architecture",
        "",
        "| Algorithm | Original | Agentic | Track C |",
        "|---|---|---|---|",
    ]
    for row in ALGORITHM_MAP:
        lines.append(f"| {row['algorithm']} | `{row['original']}` | `{row['agentic']}` | {row['track_c_reason']} |")
    lines.append("")
    for run in bundle.get("runs") or []:
        lines.append(f"## {run['patient_id']} — {run['case']}")
        lines.append("")
        lines.append(f"- Input hash: `{run['input']['input_hash']}`")
        lines.append(f"- Missing fields: {', '.join(run['input']['missing']) or 'none'}")
        lines.append(f"- Record hash: `{run['record_hash']}`")
        lines.append("")
        for algorithm, tracks in run["cells"].items():
            lines.append(f"### {algorithm}")
            lines.append("")
            for track, cell in tracks.items():
                verification = cell.get("verification") or {}
                lines.append(
                    f"- Track {track}: ok={cell.get('ok')} elapsed_ms={cell.get('elapsed_ms')} "
                    f"verification={verification.get('status')} error={cell.get('error') or ''}"
                )
                top = (cell.get("rankings") or [])[:3]
                if top:
                    lines.append("  - Top: " + ", ".join(f"{item['label']}={item['score']}" for item in top))
                audit = cell.get("weight_audit") or {}
                if audit.get("model_identifier"):
                    lines.append(
                        f"  - Weight model: {audit.get('model_identifier')} live={audit.get('live_model')} "
                        f"prompt={audit.get('prompt_version')}"
                    )
                changed = ((audit.get("validation") or {}).get("changed")) or []
                if changed:
                    lines.append("  - Weight edits: " + ", ".join(
                        f"{item['table']}.{item['key']} {item['baseline']}→{item['proposed']}" for item in changed
                    ))
                for ablation in audit.get("ablation") or []:
                    lines.append(
                        f"  - Ablation {ablation['table']}.{ablation['key']}: "
                        f"top score delta {ablation.get('top_score_delta')}, label changed={ablation.get('top_label_changed')}"
                    )
            diff = run["differentials"].get(algorithm, {})
            for pair, item in diff.items():
                lines.append(f"- {pair}: {item.get('status')} — {item.get('explanation')}")
            lines.append("")
    lines.append("Failures above are retained. A failed cell does not remove the other cells.")
    lines.append("")
    return "\n".join(lines)
