"""LLM interpretation, tool-enabled Q&A, and report generation."""

from __future__ import annotations

from datetime import datetime

from src.diagnosis.engine import DiagnosisResult
from src.llm.gemini_client import GeminiClient, GeminiResponse
from src.llm.prompts import (
    DIAGNOSIS_PROMPT,
    QA_PROMPT,
    REPORT_PROMPT,
    SYSTEM_INSTRUCTION,
    TOOL_CHAT_PROMPT,
)
from src.llm.tools import ToolExecutionContext
from src.logger import get_logger

logger = get_logger("diagnosis.interpreter")


def interpret_diagnosis(gemini: GeminiClient, result: DiagnosisResult, context: str = "") -> str:
    """Ask Gemini to interpret the signal-processing evidence."""
    prompt = DIAGNOSIS_PROMPT.format(
        evidence=result.to_evidence_text(), context=context or "No reference documents available."
    )
    return gemini.generate(prompt, system_instruction=SYSTEM_INSTRUCTION).text


def ask_with_signal_tools(
    gemini: GeminiClient,
    question: str,
    tool_context: ToolExecutionContext,
    rag_context: str = "",
    previous_interaction_id: str | None = None,
) -> GeminiResponse:
    """Answer a natural-language question by letting Gemini invoke Python signal-processing tools.

    Maintains server-side conversation state across follow-up turns via
    ``previous_interaction_id``.
    """
    prompt = TOOL_CHAT_PROMPT.format(
        session_state=tool_context.describe_state(),
        rag_context=rag_context or "No reference documents available.",
        question=question,
    )
    return gemini.generate_with_tools(
        prompt,
        tool_context=tool_context,
        system_instruction=SYSTEM_INSTRUCTION,
        previous_interaction_id=previous_interaction_id,
    )


def answer_question(
    gemini: GeminiClient,
    question: str,
    context: str = "",
    diagnosis: DiagnosisResult | None = None,
    previous_interaction_id: str | None = None,
    tool_context: ToolExecutionContext | None = None,
) -> GeminiResponse:
    """Answer a technical question, using Gemini Function Calling when ``tool_context`` is supplied."""
    if tool_context is not None:
        return ask_with_signal_tools(
            gemini,
            question=question,
            tool_context=tool_context,
            rag_context=context,
            previous_interaction_id=previous_interaction_id,
        )

    prompt = QA_PROMPT.format(
        context=context or "No reference documents available.",
        diagnosis=diagnosis.to_evidence_text() if diagnosis else "None",
        question=question,
    )
    return gemini.generate(
        prompt, system_instruction=SYSTEM_INSTRUCTION, previous_interaction_id=previous_interaction_id
    )


def _build_deterministic_report_sections(
    result: DiagnosisResult,
    asset: dict,
    signal_info: dict,
    rag_hits: list[dict],
) -> tuple[str, str]:
    """Build deterministic Markdown sections (1–6 and 9) from actual measurements and RAG hits."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sig_name = signal_info.get("signal_name", "Uploaded Signal")
    sig_source = signal_info.get("source_type", "File / Built-in")
    sig_channel = signal_info.get("channel", "Default")
    n_samples = int(result.features.get("samples", len(result.processed_signal)))
    duration_s = float(result.features.get("duration_s", n_samples / result.fs))
    rpm_val = result.shaft_hz * 60.0

    geom = result.geometry
    geom_str = (
        f"N = {geom.n_balls} elements, d = {geom.ball_diameter:.3f} mm, "
        f"D = {geom.pitch_diameter:.3f} mm, φ = {geom.contact_angle_deg:.1f}°"
        if geom is not None
        else "Explicitly provided"
    )

    f_psd, p_psd = result.psd
    psd_peak_idx = int(p_psd.argmax()) if len(p_psd) > 0 else 0
    psd_peak_hz = float(f_psd[psd_peak_idx]) if len(f_psd) > 0 else 0.0
    psd_peak_val = float(p_psd[psd_peak_idx]) if len(p_psd) > 0 else 0.0
    dom_hz = result.dominant_frequency["frequency_hz"] if result.dominant_frequency else 0.0
    dom_amp = result.dominant_frequency["amplitude"] if result.dominant_frequency else 0.0

    feats = result.features
    ff = result.fault_frequencies

    comp_lines = [
        "| Fault Code | Harmonic | Theoretical (Hz) | Observed FFT Peak (Hz) | Diff (Hz) | Rel Diff (%) | Screening Match | Envelope 1x SNR (dB) |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    if result.fft_comparison is not None:
        for code, comp in result.fft_comparison.comparisons.items():
            fm = result.fault_matches.get(code, {}).get("harmonics", [])
            env_snr = fm[0].get("snr_db", 0.0) if fm else 0.0
            for h in comp.harmonics_checked:
                obs_str = f"{h.matched_observed_hz:.2f}" if h.matched_observed_hz is not None else "—"
                diff_str = f"{h.frequency_diff_hz:.2f}" if h.frequency_diff_hz is not None else "—"
                rel_str = f"{h.relative_diff_pct:.2f}%" if h.relative_diff_pct is not None else "—"
                match_str = "Yes (screening)" if h.matched else "No"
                snr_str = f"{env_snr:.1f}" if h.harmonic_order == 1 else "—"
                comp_lines.append(
                    f"| **{code}** | {h.harmonic_order}x | {h.theoretical_harmonic_hz:.2f} | "
                    f"{obs_str} | {diff_str} | {rel_str} | {match_str} | {snr_str} |"
                )
    comp_table = "\n".join(comp_lines)

    asset_rows = "\n".join(f"- **{k}**: {v}" for k, v in asset.items()) if asset else "- **Asset**: Standard Test Rig"
    suspected_str = (
        ", ".join(result.suspected_faults)
        if result.suspected_faults
        else "None (No strong bearing defect signature above screening threshold)"
    )

    header_md = f"""# VibraMind AI — Industrial Bearing Diagnostic Report

**Generated:** {timestamp}  
**Screening Summary:** {suspected_str}

---

## 1. Uploaded Signal Information
- **Signal Identifier:** `{sig_name}`
- **Data Source:** {sig_source}
- **Selected Channel / Variable:** `{sig_channel}`
- **Sample Count:** {n_samples:,} samples
- **Signal Duration:** {duration_s:.4f} s
- **Preprocessing Applied:** Detrend = `{result.detrend}`, Normalization = `{result.normalization}`
{asset_rows}

## 2. Sampling Frequency & Operating Conditions
- **Sampling Frequency ($f_s$):** {result.fs:,.2f} Hz (Nyquist frequency: {result.fs / 2.0:,.2f} Hz)
- **Shaft Rotational Speed:** {rpm_val:.2f} RPM ({result.shaft_hz:.3f} Hz)
- **Bearing Geometry:** {geom_str}

## 3. Time-Domain Statistical Measurements
| Indicator | Value | Diagnostic Significance |
|---|---:|---|
| **Mean** | {feats['mean']:.6f} | DC offset / static baseline |
| **Variance** | {feats['variance']:.6f} | Overall signal power dispersion |
| **Standard Deviation (Std)** | {feats['std']:.6f} | AC vibration fluctuation level |
| **RMS** | {feats['rms']:.6f} | Overall vibration energy severity |
| **Kurtosis (excess)** | {feats['kurtosis']:.3f} | Impulsiveness / spikiness (> 1.5 suggests repetitive impacts) |
| **Crest Factor** | {feats['crest_factor']:.3f} | Peak-to-RMS ratio (> 4.5 suggests early-stage impacting) |
| **Peak Amplitude** | {feats['peak']:.6f} | Maximum absolute excursion |
| **Peak-to-Peak** | {feats['peak_to_peak']:.6f} | Total dynamic swing |
| **Skewness** | {feats['skewness']:.3f} | Waveform asymmetry |

## 4. FFT & Welch PSD Spectral Findings
- **Dominant FFT Frequency:** **{dom_hz:.2f} Hz** (Amplitude: {dom_amp:.5f})
- **Welch PSD Peak Density Frequency:** **{psd_peak_hz:.2f} Hz** (Power Spectral Density: {psd_peak_val:.6e})
- **Rule-Based Spectral Screening Notes:**
""" + "\n".join(f"  - {finding}" for finding in result.findings) + f"""

## 5. Theoretical Bearing Characteristic Frequencies
Calculated from kinematic zero-slip equations at shaft frequency $f_r = {result.shaft_hz:.3f}\\text{{ Hz}}$ ({rpm_val:.1f} RPM):

| Characteristic Frequency | Code | Theoretical Value (Hz) | Order ($\\times f_r$) | Physical Defect Location |
|---|---|---:|---:|---|
| **Fundamental Train Frequency** | `FTF` | **{ff['FTF']:.2f} Hz** | {ff['FTF'] / result.shaft_hz:.3f}$\\times$ | Cage / retainer defect |
| **Ball Pass Frequency Outer Race** | `BPFO` | **{ff['BPFO']:.2f} Hz** | {ff['BPFO'] / result.shaft_hz:.3f}$\\times$ | Stationary outer raceway defect |
| **Ball Pass Frequency Inner Race** | `BPFI` | **{ff['BPFI']:.2f} Hz** | {ff['BPFI'] / result.shaft_hz:.3f}$\\times$ | Rotating inner raceway defect |
| **Ball Spin Frequency** | `BSF` | **{ff['BSF']:.2f} Hz** | {ff['BSF'] / result.shaft_hz:.3f}$\\times$ | Rolling element (ball/roller) defect |

## 6. Observed vs. Theoretical Frequency Comparison
> **Important Engineering Note:** Theoretical bearing frequencies assume pure rolling contact without slip. An observed FFT peak matching a theoretical frequency is a screening indicator only and is **never** definitive proof of a bearing fault without envelope demodulation and time-domain corroboration.

{comp_table}
"""

    if rag_hits:
        ref_lines = ["## 9. Retrieved Technical References (ChromaDB RAG)"]
        for idx, hit in enumerate(rag_hits, start=1):
            doc_name = hit.get("document_name", "Unknown Document")
            page_num = hit.get("page_number", "?")
            cat = hit.get("category", "Reference")
            score = hit.get("similarity_score", 0.0)
            snippet = str(hit.get("text", "")).strip().replace("\n", " ")
            if len(snippet) > 320:
                snippet = snippet[:317] + "..."
            ref_lines.append(
                f"{idx}. **`[{doc_name}, Page {page_num}]`** (*{cat}*, similarity `{score:.3f}`): \"{snippet}\""
            )
        refs_md = "\n".join(ref_lines)
    else:
        refs_md = (
            "## 9. Retrieved Technical References (ChromaDB RAG)\n"
            "- No external manual or research paper chunks were retrieved for this report."
        )

    return header_md, refs_md


def generate_report(
    gemini: GeminiClient,
    result: DiagnosisResult,
    interpretation: str = "",
    asset: dict | None = None,
    signal_info: dict | None = None,
    rag_hits: list[dict] | None = None,
) -> str:
    """Generate a complete 10-section Markdown diagnostic report combining actual measurements, RAG citations, and Gemini analysis."""
    asset = asset or {}
    signal_info = signal_info or {}
    rag_hits = rag_hits or []

    header_md, refs_md = _build_deterministic_report_sections(
        result=result,
        asset=asset,
        signal_info=signal_info,
        rag_hits=rag_hits,
    )

    asset_text = "\n".join(f"- {k}: {v}" for k, v in {**asset, **signal_info}.items()) or "- Not specified"
    asset_text += f"\n- Report generated: {datetime.now():%Y-%m-%d %H:%M}"

    if rag_hits:
        references_text = "\n\n".join(
            f"[{h.get('document_name')}, Page {h.get('page_number')}] ({h.get('category', 'Reference')}):\n{h.get('text', '')}"
            for h in rag_hits
        )
    else:
        references_text = "No reference documents available."

    prompt = REPORT_PROMPT.format(
        asset=asset_text,
        evidence=result.to_evidence_text(),
        references=references_text,
        interpretation=interpretation or "Not available.",
    )
    logger.info("Generating diagnostic report")
    llm_narrative = gemini.generate(prompt, system_instruction=SYSTEM_INSTRUCTION).text.strip()

    return f"{header_md}\n\n{llm_narrative}\n\n{refs_md}\n"
