#!/usr/bin/env python3
"""Print the paper's result numbers (macros, Tables 1, 3, Fig. 2, MusicGen case study) from results/.

Usage
    python3 make_paper_numbers.py            # macros + tables, tab-separated
    python3 make_paper_numbers.py --macros   # only the LaTeX macros

Each macro line is  <macro> TAB <value as typeset> TAB <where in the paper>.
The value is formatted from a number read out of results/; nothing is typed in.
Table blocks follow the macros: Table 1 and the counts of its caption, the
counts of the Table 2 caption (main-map checkpoints in the rate check), the
MusicGen case study (Sec. 3.3), the roster count of the Table 3 caption,
Table 3 and the 40 main-map cells of Fig. 2. Table 2 itself (capture specification) is config/checkpoints.yaml.

Counting rule: the main map is the sixteen checkpoints minus MusicGen-S/M/L
(thirteen checkpoints; ten checkpoint/front-end pairings).  Every main-map
count, median, maximum and Spearman correlation is taken over it; MusicGen's
values are printed separately (case-study macros and block).  Within the main
map every per-cue count uses all ten pairings.  The dagger of Fig. 2 and
Table 3 marks a coarse frame grid; it is read from
results/frame_diagnostic.json and excludes nothing from any count.

Checks (each stops the script with a message and exit status 1): the stored
interval flags and counts agree with the intervals, the coarse-frame-grid flags
agree with their rule and with the shipped extraction/grid_common.py (sha256 of
its text with line endings normalised to LF), all nine trained-minus-random
paired intervals lie below zero, and the files agree with each other where
they share a number.  Subset checks: before any main-map number is formed,
the same computation over the full population must reproduce the file's own
record (per-cue lists of primary_map.json, the D summary of
content_replacement.json, the offset counts of offset_counts.json, the
Spearman values of rate_rescore.json and rate_common12.json); and per cue the
main-map lists plus the MusicGen lists must be exactly the full lists.

Runs on CPU with the standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def fail(msg):
    raise SystemExit("make_paper_numbers.py: " + msg)


def dig(obj, path):
    cur = obj
    for key in path:
        cur = cur[key]
    return cur


# ---------------------------------------------------------------- vocabulary
CUES = ["freq", "harm", "onset", "temp"]
CUE_KEY = {"freq": "freq_proximity", "harm": "harmonicity",
           "onset": "onset_sync", "temp": "temp_proximity"}
CUE_ROW = {"freq": "Frequency prox.", "harm": "Harmonicity",
           "onset": "Onset sync.", "temp": "Temporal prox."}
PAIRED_ORDER = ["musicgen_small", "musicgen_medium", "musicgen_large",
                "magnet_small", "magnet_medium", "vampnet", "acestep15",
                "acestep15_xl", "diffrhythm12", "pupujepa_large",
                "magenta_rt2_small", "magenta_rt2_base", "stableaudio3_medium"]
DISPLAY = {
    "musicgen_small": "MusicGen-S", "musicgen_medium": "MusicGen-M",
    "musicgen_large": "MusicGen-L", "magnet_small": "MAGNeT-S",
    "magnet_medium": "MAGNeT-M", "vampnet": "VampNet",
    "acestep15": "ACE-Step 1.5", "acestep15_xl": "ACE-Step 1.5 XL",
    "diffrhythm12": "DiffRhythm-v1.2", "pupujepa_large": "PupuM2D-L",
    "magenta_rt2_small": "Magenta-RT2-S", "magenta_rt2_base": "Magenta-RT2-B",
    "stableaudio3_medium": "Stable Audio 3",
    "mert": "MERT-330M", "mert_95m": "MERT-95M", "muq": "MuQ",
}
MG = ["musicgen_small", "musicgen_medium", "musicgen_large"]
# the MusicGen extraction case study; the main map is everything else
CASE = list(MG)
MAIN_PAIRED = [k for k in PAIRED_ORDER if k not in CASE]
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
RANDINIT_SEEDS = ["s4242", "s4243", "s4244"]
# rows and arms of Table 3 (content replacement); MusicGen rows below the median
CR_MAIN_ROWS = [("fe_encodec32_pre", "EnCodec pre-q."),
                ("fe_encodec32_rvq", "EnCodec RVQ"),
                ("magnet_small", "MAGNeT-S"), ("magnet_medium", "MAGNeT-M"),
                ("mert", "MERT-330M"), ("muq", "MuQ"), ("pupujepa_large", "PupuM2D-L"),
                ("acestep15", "ACE-Step 1.5"), ("acestep15_xl", "ACE-Step XL"),
                ("stableaudio3_medium", "Stable Audio 3")]
CR_CASE_ROWS = [("musicgen_small", "MusicGen-S"), ("musicgen_medium", "MusicGen-M"),
                ("musicgen_large", "MusicGen-L")]
CR_ROWS = CR_MAIN_ROWS + CR_CASE_ROWS
CR_ARMS = [("orig", "orig."), ("cd_noise", "noise"), ("cd_shuffle", "shuffle"),
           ("cd_silence", "clicks")]
CLOCK_MAX_HZ = 0.333   # clock frequencies below this keep f*r*tau < 1/2 on every grid


class Book:
    def __init__(self):
        self.rows = []
        self.value = {}

    def add(self, name, raw, kind, where):
        if name in self.value:
            fail("macro %s defined twice" % name)
        if kind == "int":
            if int(raw) != raw:
                fail("%s is not an integer: %r" % (name, raw))
            text = "%d" % int(raw)
        elif kind == "word":
            if int(raw) != raw or not 0 <= raw < len(WORDS):
                fail("%s has no number word: %r" % (name, raw))
            text = WORDS[int(raw)]
        elif kind == "signed":
            text = "%+.3f" % raw
        elif kind == "plain":
            text = "%.3f" % raw
        elif kind == "text":
            text = str(raw)
        elif kind == "ci":
            lo, hi = raw
            text = "[$%+.3f$, $%+.3f$]" % (lo, hi)
        else:
            fail("unknown kind %r" % kind)
        self.value[name] = text
        self.rows.append((name, text, where))


def load_coarse_grid(primary):
    """results/frame_diagnostic.json -> {(read-out key, cue): coarse frame grid}.

    The flag is extraction/grid_common.frame_diagnostics' flag_coarse_frame_rate:
    an annotated onset pair falls on one read-out frame, fewer than 10 frames,
    or a degenerate-row share >= 1 %.  It is a mark only; no count excludes a
    flagged cell.
    """
    diag = load("frame_diagnostic.json")
    with open(os.path.join(HERE, "extraction", "grid_common.py"), "rb") as fh:
        # LF-normalised, so that a checkout with CRLF line endings hashes the same
        shipped = hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()
    if diag["grid_common_sha256"] != shipped:
        fail("frame_diagnostic.json was made by a grid_common.py (sha256 %s) other "
             "than the shipped one (%s)" % (diag["grid_common_sha256"], shipped))
    pc = diag["positive_controls"]
    if pc["n_ok"] != pc["n"] or sum(c["ok"] for c in pc["cells"].values()) != pc["n"]:
        fail("frame diagnostic positive controls not all reproduced")
    out = {}
    fields = ("n_same_frame_onset_pairs", "n_frames_min", "degenerate_row_share",
              "flag_coarse_frame_rate")
    for key, cell in diag["cells"].items():
        readout, cue_key = key.split("|", 1)
        absent = [f for f in fields if f not in cell]
        if absent:
            fail("frame_diagnostic.json %s lacks %s" % (key, ", ".join(absent)))
        rule = (cell["n_same_frame_onset_pairs"] > 0 or cell["n_frames_min"] < 10
                or cell["degenerate_row_share"] >= 0.01)
        if rule != cell["flag_coarse_frame_rate"]:
            fail("frame_diagnostic.json %s: flag %r disagrees with its own fields"
                 % (key, cell["flag_coarse_frame_rate"]))
        for cue, ck in CUE_KEY.items():
            if ck == cue_key:
                out[(readout, cue)] = bool(cell["flag_coarse_frame_rate"])
    # every cell the paper marks must be in the diagnostic
    needed = [(k, c) for k in PAIRED_ORDER for c in CUES] + [(k, "temp") for k, _ in CR_ROWS]
    missing = sorted("%s|%s" % (k, CUE_KEY[c]) for k, c in needed if (k, c) not in out)
    if missing:
        fail("frame_diagnostic.json has no cell for %s" % ", ".join(missing))
    # the pairs of primary_map.json carry the flag of the earlier, partial
    # diagnostic; every cell it flagged must still be flagged here
    for key in PAIRED_ORDER:
        for cue in CUES:
            if primary["pairs"]["%s|%s" % (key, CUE_KEY[cue])].get(
                    "flag_coarse_frame_rate") and not out[(key, cue)]:
                fail("%s|%s: flagged in primary_map.json, not in frame_diagnostic.json"
                     % (key, cue))
    return out


def spearman(x, y):
    """Spearman correlation: Pearson correlation of average ranks."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                out[order[t]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return out
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (c - my) for a, c in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((c - my) ** 2 for c in ry)) ** 0.5
    return num / den


def median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def primary_models(primary):
    out = []
    for key in primary["cells"]:
        model = key.split("|", 1)[0]
        if model not in out:
            out.append(model)
    if len(out) != 16:
        fail("expected 16 checkpoints in the primary map, got %d" % len(out))
    return out


def build():
    primary = load("primary_map.json")
    cr = load("content_replacement.json")
    traj = load("fixed_trajectory.json")
    zero = load("zero_waveform.json")
    offset = load("offset_counts.json")
    offset_paired = load("offsets.json")
    fbi = load("first_block_input.json")
    rr = load("rerender_random.json")
    rr_ci = load("rerender_random_intervals.json")
    rr_ls = load("rerender_size_contrast.json")
    rate = load("rate_rescore.json")
    common12 = load("rate_common12.json")
    static = load("static_tables.json")
    clock = load("clock_reference.json")
    sel = load("index_selectivity.json")
    nat = load("musicgen_native_schedule.json")
    seln = load("index_selectivity_native.json")

    b = Book()
    tables = {}

    # coarse frame grid (the daggers): a presentation mark, never an exclusion
    coarse = load_coarse_grid(primary)

    # fixed DSP / learned front-end columns of Table 1: largest |rho|
    fixed_learned = {}
    for cue in CUES:
        s = static[CUE_KEY[cue]]
        fixed_learned[cue] = (max(abs(v) for v in s["fixed_all"].values()),
                              max(abs(v) for v in s["learned_all"].values()))

    # ------------------------------------ static inputs of Sec. 2 and Sec. 3.1
    fixed = {c: static[CUE_KEY[c]]["fixed_all"] for c in CUES}
    for cue, stem in (("freq", "NCeilFreq"), ("harm", "NCeilHarm"),
                      ("onset", "NCeilOnset")):
        b.add(stem, max(fixed[cue].values()), "plain",
              "Sec. 3.1, best fixed transform, %s" % CUE_KEY[cue])
    b.add("NFixedTempMax", fixed_learned["temp"][0], "plain",
          "Sec. 3.1, fixed DSP temporal |rho| bound")
    b.add("NLearnedTempMax", fixed_learned["temp"][1], "plain",
          "Sec. 3.1, learned front-end temporal maximum")
    for name, m in static["stimulus_spec"]["macros"].items():
        b.add(name.lstrip("\\"), m["tex"], "text", "Sec. 2.1 Stimuli (%s)" % m["derived_from"])

    # ------------------------------------------------------ learned front-ends
    fe_freq = static["freq_proximity"]["learned_all"]
    if len(fe_freq) != 8:
        fail("expected 8 learned front-end read-outs")
    b.add("NatFeFreqLo", min(fe_freq.values()), "plain",
          "Sec. 3.1, learned front-ends on frequency proximity (low)")
    b.add("NatFeFreqHi", max(fe_freq.values()), "plain",
          "Sec. 3.1, learned front-ends on frequency proximity (high)")

    # ----------------------------------------------------------- primary map
    cells = []
    for key in PAIRED_ORDER:
        for cue in CUES:
            de = primary["pairs"]["%s|%s" % (key, CUE_KEY[cue])]["delta_E"]["auc"]
            lo, hi = de["ci_low"], de["ci_high"]
            resolves = "above" if lo > 0 else "below" if hi < 0 else "covers"
            if (resolves == "above") != de["resolves_above"] or \
               (resolves == "below") != de["resolves_below"]:
                fail("%s|%s interval sign disagrees with the stored flags" % (key, cue))
            cells.append({"key": key, "checkpoint": DISPLAY[key], "cue": cue,
                          "delta": de["delta"], "lo": lo, "hi": hi,
                          "resolves": resolves, "dagger": coarse[(key, cue)]})
    cell_of = {(r["key"], r["cue"]): r for r in cells}
    main_models = [m for m in primary_models(primary) if m not in CASE]
    # subset check: the main map is the sixteen minus exactly the three MusicGen sizes
    if len(main_models) != 13 or len(MAIN_PAIRED) != 10 or \
            sorted(set(primary_models(primary)) - set(main_models)) != sorted(CASE):
        fail("main map is not the sixteen checkpoints minus MusicGen-S/M/L")
    # per-cue lists over all thirteen pairings, rebuilt from the cells and checked
    # against the run's own interval_sign_counts; then the main-map counts
    summary = {}
    for cue in CUES:
        who = {"above": set(), "below": set(), "covers": set()}
        for key in PAIRED_ORDER:
            who[cell_of[(key, cue)]["resolves"]].add(key)
        run_counts = primary["interval_sign_counts"][CUE_KEY[cue]]["auc"]
        for bucket, field in (("above", "above_zero"), ("below", "below_zero"),
                              ("covers", "covers_zero")):
            if set(run_counts[field]) != who[bucket]:
                fail("cue %s %s: rebuilt %r, run says %r"
                     % (cue, bucket, sorted(who[bucket]), sorted(run_counts[field])))
        main_who = {k: {m for m in v if m in MAIN_PAIRED} for k, v in who.items()}
        case_who = {k: {m for m in v if m in CASE} for k, v in who.items()}
        # subset check: main-map lists + MusicGen lists = the full lists, disjointly
        for bucket in who:
            if main_who[bucket] | case_who[bucket] != who[bucket] or \
                    main_who[bucket] & set(CASE):
                fail("cue %s %s: main-map/case split does not partition the run's list"
                     % (cue, bucket))
        tally = {k: len(v) for k, v in main_who.items()}
        if sum(tally.values()) != len(MAIN_PAIRED):
            fail("cue %s: %d main-map pairings counted, not %d"
                 % (cue, sum(tally.values()), len(MAIN_PAIRED)))
        best = max(primary["cells"]["%s|%s" % (m, CUE_KEY[cue])]["point"]["auc"]
                   for m in main_models)
        summary[cue] = {"fixed": fixed_learned[cue][0], "learned": fixed_learned[cue][1],
                        "model": best, "n_pairs": sum(tally.values()),
                        "above": tally["above"], "below": tally["below"],
                        "above_keys": main_who["above"], "below_keys": main_who["below"]}
    cue_mac = {"freq": "Freq", "harm": "Harm", "onset": "Onset", "temp": "Temp"}
    for cue in CUES:
        s = summary[cue]
        b.add("NatAbove%s" % cue_mac[cue], s["above"], "int",
              "Sec. 3.1 / Table 1, resolved above (main-map pairings), %s" % CUE_KEY[cue])
        b.add("NatBelow%s" % cue_mac[cue], s["below"], "int",
              "Sec. 3.1 / Table 1, resolved below (main-map pairings), %s" % CUE_KEY[cue])
    # every cue counts all main-map pairings, so one pairing count serves all four cues
    npairs = {summary[c]["n_pairs"] for c in CUES}
    if npairs != {len(MAIN_PAIRED)}:
        fail("pairings counted per cue differ: %r" % sorted(npairs))
    b.add("NatNPairs", len(MAIN_PAIRED), "int",
          "Sec. 3.1 / Table 1, main-map checkpoint/front-end pairings per cue")
    b.add("NatNPairsWord", len(MAIN_PAIRED), "word", "Abstract, the same count as a word")
    b.add("NatNCells", 4 * len(MAIN_PAIRED), "int", "Fig. 2, main-map Delta_E cells")
    b.add("NatNMainWord", len(main_models), "word", "Abstract / Sec. 3, main-map checkpoints")
    b.add("NatNAllWord", len(primary_models(primary)), "word", "Abstract, all checkpoints")
    b.add("NatPosNonTemp", sum(summary[c]["above"] for c in ("freq", "harm", "onset")),
          "int", "Sec. 3.1, positive non-temporal contrasts (main map)")
    b.add("NatNegNonTemp", sum(summary[c]["below"] for c in ("freq", "harm", "onset")),
          "int", "Sec. 3.1, negative non-temporal contrasts (main map)")
    for cue in CUES:
        best_key, best_val = None, None
        for m in main_models:
            val = primary["cells"]["%s|%s" % (m, CUE_KEY[cue])]["point"]["auc"]
            if best_val is None or val > best_val:
                best_key, best_val = m, val
        b.add("NatBest%s" % cue_mac[cue], best_val, "plain",
              "Sec. 3.1 / Table 1, best main-map checkpoint, %s" % CUE_KEY[cue])
        if cue == "temp":
            b.add("NatBestTempArg", DISPLAY[best_key], "text",
                  "Sec. 3.1, best temporal checkpoint (main map)")
    worst_key, worst_val = None, None
    for key in MAIN_PAIRED:
        val = primary["pairs"]["%s|freq_proximity" % key]["delta_E"]["auc"]["delta"]
        if worst_val is None or val < worst_val:
            worst_key, worst_val = key, val
    b.add("NatDeMinFreq", worst_val, "signed",
          "Sec. 3.1, most negative frequency Delta_E (main map)")
    b.add("NatDeMinFreqArg", DISPLAY[worst_key], "text", "Sec. 3.1, its checkpoint")
    # MusicGen case study (Sec. 3.3): synchronised-token Delta_E on the other cues
    mg_other = {"above": 0, "below": 0, "covers": 0}
    for cue in ("freq", "harm", "onset"):
        for k, letter in zip(CASE, "SML"):
            b.add("NatDeMg%s%s" % (letter, cue_mac[cue]), cell_of[(k, cue)]["delta"],
                  "signed", "Sec. 3.3 case study, MusicGen-%s Delta_E, %s" % (letter, CUE_KEY[cue]))
            mg_other[cell_of[(k, cue)]["resolves"]] += 1
    b.add("NatMgOtherAbove", mg_other["above"], "int",
          "Sec. 3.3, MusicGen non-temporal contrasts resolved above")
    b.add("NatMgOtherBelow", mg_other["below"], "int",
          "Sec. 3.3, MusicGen non-temporal contrasts resolved below")
    if any(cell_of[(k, "temp")]["resolves"] == "above" for k in CASE):
        fail("a MusicGen synchronised-token temporal contrast resolves above")
    b.add("NatMgTempBelowList", "/".join(letter for k, letter in zip(CASE, "SML")
                                         if cell_of[(k, "temp")]["resolves"] == "below"),
          "text", "Sec. 3.3, MusicGen temporal contrasts resolved below (sizes)")
    for key, stem in zip(MG, ["NatMgSAuc", "NatMgMAuc", "NatMgLAuc"]):
        b.add(stem, primary["cells"]["%s|temp_proximity" % key]["point"]["auc"],
              "signed", "Sec. 3.3 case study, synchronised tokens, exact AUC")

    # ---------------------------------------------- fixed-trajectory control
    tcell = traj["cells"]
    if sorted(tcell) != sorted(cr["checkpoints"] + cr["front_ends"]):
        fail("fixed-trajectory roster differs from the content-replacement roster")
    enc = tcell["fe_encodec32_pre"]["cd_silence"]
    b.add("NatTrajEncSilShare", enc["share"], "plain",
          "Sec. 3.2 index-only, variance share of the shared trajectory (clicks)")
    b.add("NatTrajEncSil", enc["mean_trajectory"], "signed",
          "Sec. 3.2 index-only, EnCodec encoder shared trajectory (clicks)")
    for key, stem in zip(MG, ["NatTrajMgS", "NatTrajMgM", "NatTrajMgL"]):
        b.add(stem, tcell[key]["orig"]["mean_trajectory"], "signed",
              "Sec. 3.3 case study, shared trajectory on tones")
    fe_mt = {m: tcell[m]["orig"]["mean_trajectory"] for m in cr["front_ends"]}
    fe_top = max(fe_mt, key=fe_mt.get)
    b.add("NatTrajFeMax", fe_mt[fe_top], "signed",
          "Sec. 3.2 index-only, best front-end trajectory")
    main_ckpts = [m for m in cr["checkpoints"] if m not in CASE]
    if sorted(main_ckpts) != sorted(main_models):
        fail("content-replacement main-map roster differs from the primary map's")
    above = {m: tcell[m]["orig"]["mean_trajectory"] for m in main_ckpts
             if tcell[m]["orig"]["mean_trajectory"] > fe_mt[fe_top]}
    b.add("NatTrajNAboveFe", len(above), "int",
          "Sec. 3.2 index-only, main-map checkpoints whose trajectory exceeds every front-end")
    b.add("NatTrajAboveFeLo", min(above.values()), "signed",
          "Sec. 3.2 index-only, lowest of those")
    b.add("NatTrajNStim", traj["n_stimuli"], "int", "Sec. 3.2 index-only, stimuli")
    for src, stem in (("fe_encodec32_pre", "NatZeroEncPre"), ("fe_encodec32_rvq", "NatZeroEncRvq"),
                      ("musicgen_small", "NatZeroMgS"), ("musicgen_medium", "NatZeroMgM"),
                      ("musicgen_large", "NatZeroMgL")):
        b.add(stem, zero["cells"][src]["rho_zero"], "signed",
              ("Sec. 3.3 case study" if src in CASE else "Sec. 3.2 index-only")
              + ", all-zero input")

    # excess over the shared trajectory: Delta_I = rho9(recorded) - rho9(shared trajectory), tones
    ckpt_all = [k for k in sel["cells"] if k in cr["checkpoints"]]
    fe_keys = [k for k in sel["cells"] if k in cr["front_ends"]]
    if len(ckpt_all) != 16 or len(fe_keys) != 12:
        fail("excess over the shared trajectory, read-out split: %d/%d"
             % (len(ckpt_all), len(fe_keys)))
    ckpt_keys = [k for k in ckpt_all if k not in CASE]
    for k, r in sel["cells"].items():
        if abs(r["recorded"] - cr["cells"][k]["orig"]["rho"]) > 1e-9:
            fail("excess over the shared trajectory: recorded rho9 differs from "
                 "content_replacement for %s" % k)
    for keys, stem, who in ((ckpt_keys, "NatSel", "main-map checkpoints"), (fe_keys, "NatSelFe", "front-ends")):
        ab = [k for k in keys if sel["cells"][k]["resolves_above"]]
        be = [k for k in keys if sel["cells"][k]["resolves_below"]]
        b.add(stem + "Above", len(ab), "int", "Sec. 3.2 / Table 3, Delta_I resolves above (%s)" % who)
        b.add(stem + "Below", len(be), "int", "Sec. 3.2 / Table 3, Delta_I resolves below (%s)" % who)
        b.add(stem + "Cover", len(keys) - len(ab) - len(be), "int",
              "Sec. 3.2 / Table 3, Delta_I covers zero (%s)" % who)
        if stem == "NatSel":
            short = {"acestep15_xl": "ACE-Step~XL", "magnet_small": "MAGNeT-S", "magnet_medium": "MAGNeT-M"}
            names = sorted(short[k] for k in ab)
            if names != ["ACE-Step~XL", "MAGNeT-M", "MAGNeT-S"]:
                fail("checkpoints resolving above their shared trajectory: %r" % ab)
            b.add("NatSelAboveList", "ACE-Step~XL, MAGNeT-S/M", "text",
                  "Sec. 3.2, those checkpoints (names)")
    # "Delta_I is negative where the shared trajectory itself scores high (lo to
    # hi)": the main-map checkpoints resolving below are exactly those whose
    # shared trajectory exceeds every front-end's; hi is the top of that range
    below_main = [k for k in ckpt_keys if sel["cells"][k]["resolves_below"]]
    if set(below_main) != set(above):
        fail("main-map Delta_I-below set differs from the trajectory-above-front-end set")
    for k in below_main:
        if abs(sel["cells"][k]["trajectory"] - above[k]) > 1e-9:
            fail("index_selectivity.json trajectory differs from fixed_trajectory.json for %s" % k)
    b.add("NatTrajMainHi", max(sel["cells"][k]["trajectory"] for k in below_main), "signed",
          "Sec. 3.2, highest shared trajectory where Delta_I resolves below (main map)")
    if not all(sel["cells"][k]["resolves_below"] for k in CASE):
        fail("a MusicGen synchronised-token Delta_I no longer resolves below")

    # ------------------------------------------------- content replacement
    # paired D per checkpoint: recomputed over all sixteen it must reproduce the
    # file's own summary; the paper's statistics are over the main map
    def dstats(arm, pop):
        rows = [cr["D"][arm][m] for m in pop]
        ds = [r["delta"] for r in rows]
        return {"n": len(ds), "median": median(ds), "min": min(ds), "max": max(ds),
                "n_interval_above_zero": sum(r["lo"] > 0 for r in rows),
                "n_interval_below_zero": sum(r["hi"] < 0 for r in rows)}
    main_d = {}
    for arm, _ in CR_ARMS[1:]:
        full = dstats(arm, cr["checkpoints"])
        for field, val in full.items():
            if abs(val - cr["summary"][arm]["checkpoints"][field]) > 1e-12:
                fail("content replacement %s %s: recount %r, file %r"
                     % (arm, field, val, cr["summary"][arm]["checkpoints"][field]))
        main_d[arm] = dstats(arm, main_ckpts)
    for arm, stem in (("cd_noise", "Noise"), ("cd_shuffle", "Shuf")):
        for field, suffix in (("median", "MedD"), ("min", "MinD"), ("max", "MaxD")):
            b.add("Nat%s%s" % (stem, suffix), main_d[arm][field],
                  "signed", "Sec. 3.2 content, %s D %s (main map) / Table 3" % (arm, field))
    for m, stem in (("diffrhythm12", "NatDrDe"), ("pupujepa_large", "NatPupuDe")):
        for arm, tag in (("orig", "Orig"), ("cd_noise", "Noise")):
            b.add(stem + tag, cr["own_fe"][arm][m]["delta"], "signed",
                  "Sec. 3.2 content, own front-end contrast (%s)" % arm)
    for arm, tag in (("cd_noise", "Noise"), ("cd_shuffle", "Shuf"), ("cd_silence", "Sil")):
        for field, suffix in (("n_interval_above_zero", "DAbove"),
                              ("n_interval_below_zero", "DBelow")):
            b.add("Nat" + tag + suffix, main_d[arm][field], "int",
                  "Sec. 3.2 content, paired D intervals (%s, main map)" % arm)
    b.add("NatPupuFeTemp", cr["cells"]["fe_pupujepa_logmel"]["orig"]["rho"], "signed",
          "Sec. 3.1, PupuM2D log-spectrogram temporal score")
    for key, arm, stem in (("fe_encodec32_pre", "orig", "NatEncPreOrig"),
                           ("fe_encodec32_pre", "cd_silence", "NatEncPreSil"),
                           ("fe_encodec32_rvq", "cd_silence", "NatEncRvqSil"),
                           ("fe_encodec32_rvq", "orig", "NatEncRvqOrig"),
                           ("musicgen_small", "orig", "NatMgSOrig"),
                           ("musicgen_medium", "orig", "NatMgMOrig"),
                           ("musicgen_large", "orig", "NatMgLOrig"),
                           ("musicgen_small", "cd_silence", "NatMgSSil"),
                           ("musicgen_medium", "cd_silence", "NatMgMSil"),
                           ("musicgen_large", "cd_silence", "NatMgLSil"),
                           ("mert", "cd_silence", "NatMertSil"),
                           ("stableaudio3_medium", "cd_silence", "NatSaSil")):
        b.add(stem, cr["cells"][key][arm]["rho"], "signed",
              ("Sec. 3.3 case study" if key in CASE else "Sec. 3.2")
              + " / Table 3 (%s, %s)" % (key, arm))
    b.add("NatEncRvqSilAnchors", cr["cells"]["fe_encodec32_rvq"]["cd_silence"]["n_clusters"],
          "int", "Sec. 3.2 near-silent clicks, scorable anchors")
    for key, stem in (("magnet_small", "NatMagSSilDe"), ("stableaudio3_medium", "NatSaSilDe")):
        b.add(stem, abs(cr["own_fe"]["cd_silence"][key]["delta"]), "plain",
              "Sec. 3.2 near-silent clicks, |own front-end contrast|")

    # --------------------------------------------------------------- offsets
    rows = {(r["endpoint"], r["offset"]): r for r in offset["rows"]}
    for i, stem in enumerate(["NatMgSOffP", "NatMgMOffP", "NatMgLOffP"]):
        b.add(stem, rows[("auc", 1)]["musicgen_temporal"][i], "signed",
              "Sec. 3.3 case study, read-out +1 frame, exact AUC")
    paired_de = {}
    for off, suffix in ((0, "OffZ"), (1, "OffP")):
        for k, letter in zip(MG, "SML"):
            hits = [p for p in offset_paired["paired"]
                    if p["model"] == k and p["cue"] == "temp_proximity" and p["offset"] == off]
            if len(hits) != 1 or hits[0]["front_end"] != "fe_encodec32_rvq":
                fail("offset paired entry for %s offset %d" % (k, off))
            val = hits[0]["delta_E"]["auc"]["delta"]
            paired_de[(k, off)] = val
            b.add("NatDeMg%s%s" % (letter, suffix), val, "signed",
                  "Sec. 3.3 case study, Delta_E at offset %+d (front-end shifted likewise)" % off)
            if off == 0 and abs(val - cell_of[(k, "temp")]["delta"]) > 1e-9:
                fail("offsets.json offset-0 Delta_E differs from primary_map.json (%s)" % k)
            if off == 1 and not hits[0]["delta_E"]["auc"]["resolves_above"]:
                fail("%s offset +1 Delta_E no longer resolves positive" % k)

    # temporal contrasts resolving at each offset, per pairing (offsets.json)
    def off_resolved(off, pop):
        ab, be = [], []
        for k in pop:
            hits = [p for p in offset_paired["paired"]
                    if p["model"] == k and p["cue"] == "temp_proximity" and p["offset"] == off]
            if len(hits) != 1:
                fail("offset paired entry for %s offset %d not unique" % (k, off))
            de = hits[0]["delta_E"]["auc"]
            if de["resolves_above"] != (de["ci_low"] > 0) or \
               de["resolves_below"] != (de["ci_high"] < 0):
                fail("offsets.json flags disagree with the interval (%s, %d)" % (k, off))
            if de["resolves_above"]:
                ab.append(k)
            if de["resolves_below"]:
                be.append(k)
        return ab, be
    # subset check: over all thirteen pairings the recount must equal
    # offset_counts.json at every offset, and at offset 0 primary_map.json's lists
    if sorted(offset_paired["offsets"]) != [-2, -1, 0, 1, 2]:
        fail("offsets.json offset grid is not -2..+2")
    for off in (-2, -1, 0, 1, 2):
        ab, be = off_resolved(off, PAIRED_ORDER)
        cnt = rows[("auc", off)]["counts"]["temp_proximity"]
        if (len(ab), len(be)) != (cnt["above"], cnt["below"]):
            fail("offset %d: per-pairing recount %d/%d, offset_counts.json %d/%d"
                 % (off, len(ab), len(be), cnt["above"], cnt["below"]))
    ab0, be0 = off_resolved(0, MAIN_PAIRED)
    ab1, be1 = off_resolved(1, MAIN_PAIRED)
    if set(ab0) != summary["temp"]["above_keys"] or set(be0) != summary["temp"]["below_keys"]:
        fail("main-map offset-0 lists differ from the primary map's")
    if set(ab0) - set(ab1) or set(be0) - set(be1):
        fail("a main-map temporal contrast loses its resolution at +1")
    tex = lambda ks: ", ".join(DISPLAY[k].replace(" ", "~") for k in ks)
    b.add("NatAboveTempOffP", len(ab1), "int",
          "Sec. 3.1, main-map temporal positives at offset +1")
    b.add("NatBelowTempOffP", len(be1), "int",
          "Sec. 3.1, main-map temporal negatives at offset +1")
    b.add("NatOffPAddList", tex([k for k in ab1 if k not in ab0]), "text",
          "Sec. 3.1, main-map pairings positive at +1, not at 0")
    b.add("NatOffPBelowList", tex([k for k in be1 if k not in be0]), "text",
          "Sec. 3.1, main-map pairings negative at +1, not at 0")
    for off, suffix in ((-2, "OffNN"), (-1, "OffN"), (2, "OffPP")):
        ab, be = off_resolved(off, MAIN_PAIRED)
        b.add("NatAboveTemp" + suffix, len(ab), "int",
              "Sec. 3.1, main-map temporal positives at offset %+d" % off)
        b.add("NatBelowTemp" + suffix, len(be), "int",
              "Sec. 3.1, main-map temporal negatives at offset %+d" % off)
    # native delay-pattern token schedule (temporal cue), offset 0
    if nat["cue"] != "temp_proximity" or nat["front_end"] != "fe_encodec32_rvq":
        fail("native-schedule record is not the temporal cue against EnCodec RVQ")
    for k, letter in zip(MG, "SML"):
        for off in ("-1", "1"):
            if not nat["native"][k][off]["dE_auc_lo"] > 0:
                fail("native schedule %s offset %s: Delta_E does not resolve positive" % (k, off))
        c = nat["native"][k]["0"]
        b.add("NatMg%sNat" % letter, c["auc"], "signed",
              "Sec. 3.3 case study, native schedule, exact AUC")
        b.add("NatDeMg%sNat" % letter, c["dE_auc"], "signed",
              "Sec. 3.3 case study, native schedule, Delta_E")
    # excess over the shared trajectory, native-schedule features (tones, rho_9)
    for k, letter in zip(MG, "SML"):
        r = seln["cells"][k]
        if abs(r["recorded"] - nat["native"][k]["0"]["rho9"]) > 1e-9:
            fail("native schedule, excess over the shared trajectory: recorded rho_9 for %s "
                 "differs from the native-schedule rho_9" % k)
        b.add("NatSelMg%sNat" % letter, r["delta"], "signed",
              "Sec. 3.3 case study, Delta_I under the native schedule")

    # ---------------------------------------------------- first-block input
    post = [fbi["cells"]["first_block_input_%s|temp_proximity" % k]["point"] for k in MG]
    if len({"%.3f" % v for v in post}) != 1:
        fail("the three first-block-input scores differ at three decimals")
    b.add("NatEzeroPost", post[2], "plain", "Sec. 3.3 case study, first-block input")

    # ----------------------------------------------- re-render and random init
    trained = {k: rr["cells"]["rerender_%s|tp_rr_main" % k]["point"] for k in MG}
    for k, stem in zip(MG, ["NatRrS", "NatRrM", "NatRrL"]):
        b.add(stem, trained[k], "plain", "Sec. 3.3 size controls, re-render")
    ls = rr_ls["trained_LS"]
    if abs(ls["rho_aul_a"] - trained["musicgen_large"]) > 1e-9 or \
       abs(ls["rho_aul_b"] - trained["musicgen_small"]) > 1e-9:
        fail("re-render L-S endpoints do not match the cells")
    b.add("NatRrLS", ls["delta"], "signed", "Sec. 3.3 size controls, re-render L-S")
    b.add("NatRrLSCI", (ls["ci_low"], ls["ci_high"]), "ci", "Sec. 3.3 size controls, its interval")
    tmr = {k: v["delta_recomputed"] for k, v in rr["pairs"].items()
           if k.startswith("trained_minus_randinit_")}
    if len(tmr) != 9:
        fail("expected 9 trained-minus-random contrasts")
    # "all nine trained-minus-random paired intervals are negative": the intervals
    # (results/rerender_random_intervals.json) must match these points and lie below 0
    tmr_ci = rr_ci["pairs"]
    if sorted(k + "__" + rr_ci["cue"] for k in tmr_ci) != sorted(tmr):
        fail("rerender_random_intervals.json does not hold the nine trained-minus-random contrasts")
    for k, iv in sorted(tmr_ci.items()):
        if abs(iv["delta"] - tmr[k + "__" + rr_ci["cue"]]) > 1e-9:
            fail("%s: interval file delta %r differs from rerender_random.json %r"
                 % (k, iv["delta"], tmr[k + "__" + rr_ci["cue"]]))
        if not iv["ci_low"] <= iv["delta"] <= iv["ci_high"]:
            fail("%s: point difference outside its interval" % k)
        if not iv["ci_high"] < 0:
            fail("%s: trained-minus-random interval [%+.3f, %+.3f] is not below zero"
                 % (k, iv["ci_low"], iv["ci_high"]))
    b.add("NatTrMinusRandLo", min(tmr.values()), "signed", "Sec. 3.3 random decoders, lowest difference")
    b.add("NatTrMinusRandHi", max(tmr.values()), "signed", "Sec. 3.3 random decoders, highest difference")

    # ------------------------------------------------------------------ rate
    b.add("NatRateArms", len(rate["arms"]), "int", "Sec. 3.3 size controls, rate arms")
    hi_n = {rate["cross"]["%s|auc" % a]["n_checkpoints"] for a in ("nn75", "pool75")}
    if len(hi_n) != 1:
        fail("the two 25 Hz arms do not share a population")
    hi_all = rate["cross"]["nn75|auc"]["checkpoints"]
    if hi_all != rate["cross"]["pool75|auc"]["checkpoints"] or len(hi_all) != hi_n.pop():
        fail("the two 25 Hz arms do not list the same checkpoints")

    def arm_rho(arm, pop):
        return spearman([rate["cells"]["native|%s|auc" % k]["value"] for k in pop],
                        [rate["cells"]["%s|%s|auc" % (arm, k)]["value"] for k in pop])
    # subset check: on the files' own populations the recount reproduces their values
    for arm in ("nn75", "pool75"):
        if abs(arm_rho(arm, hi_all) - rate["cross"]["%s|auc" % arm]["spearman_vs_native"]) > 1e-9:
            fail("Spearman recount differs from rate_rescore.json (%s)" % arm)
    if sorted(common12["checkpoints"]) != sorted(hi_all):
        fail("rate_common12.json population differs from the 25 Hz arms")
    for arm in ("nn75", "pool75", "nn33", "pool33"):
        recs = [r for r in common12["records"] if r["arm"] == arm and r["endpoint"] == "auc"]
        if len(recs) != 1:
            fail("rate record for %s not unique" % arm)
        if abs(arm_rho(arm, common12["checkpoints"]) - recs[0]["rho"]) > 1e-9:
            fail("Spearman recount differs from rate_common12.json (%s)" % arm)
    hi_main = [k for k in hi_all if k not in CASE]
    b.add("NatRateEligHi", len(hi_main), "int",
          "Sec. 3.1 rate re-score, main-map checkpoints whose rates reach 25 Hz")
    # main-map temporal positives (offset 0) without a 25 Hz arm: in the native
    # arm, absent from both 25 Hz arms, present in both 11 Hz arms
    kpa = rate["keys_per_arm"]
    out = [k for k in MAIN_PAIRED
           if k in summary["temp"]["above_keys"] and k not in hi_main]
    for k in out:
        if k not in kpa["native"] or k in kpa["nn75"] or k in kpa["pool75"] \
                or k not in kpa["nn33"] or k not in kpa["pool33"]:
            fail("%s is not a below-25 Hz checkpoint in rate_rescore.json" % k)
    short = {"acestep15": "ACE-Step~1.5", "acestep15_xl": "ACE-Step~XL",
             "diffrhythm12": "DiffRhythm"}
    if any(k not in short for k in out):
        fail("no prose name for %r" % out)
    b.add("NatRateOutPos", len(out), "int",
          "Sec. 3.1 rate re-score, main-map temporal positives below 25 Hz (outside the check)")
    b.add("NatRateOutList", ", ".join(short[k] for k in out), "text",
          "Sec. 3.1 rate re-score, their names")
    for arm, stem in (("nn75", "NatRateRankNn"), ("pool75", "NatRateRankPool"),
                      ("nn33", "NatRateRankNnLoC"), ("pool33", "NatRateRankPoolLoC")):
        b.add(stem, arm_rho(arm, hi_main), "plain",
              "Sec. 3.1 rate re-score, Spearman native vs %s on those checkpoints" % arm)
    # Table 2 caption: ticked pairings (both 25 Hz arms paired) outside MusicGen,
    # unpaired encoders, and their total
    ticked = {v["model"] for v in rate["paired"].values() if v["arm"] == "nn75"}
    if ticked != {v["model"] for v in rate["paired"].values() if v["arm"] == "pool75"}:
        fail("the two 25 Hz arms pair different checkpoints")
    t2 = (len(ticked - set(CASE)), len(set(hi_all) - ticked), len(hi_main))
    if t2[0] + t2[1] != t2[2]:
        fail("Table 2 caption counts do not add up")

    # ----------------------------------------------------------------- clock
    # f in {0.05, 0.1, 0.2} Hz on the frame grids of the read-outs (frame_rates.json);
    # the file also holds f = 0.333 Hz and a T = 3000 grid, which no read-out uses
    grids = {v["t_native"] for v in load("frame_rates.json")["keys"].values()}
    band = [r["rho"] for r in clock["rows"]
            if r["rep"].startswith("pos_fourier_f") and r["defined"] and r["T"] in grids
            and float(r["rep"][len("pos_fourier_f"):]) < CLOCK_MAX_HZ]
    if len(band) != 3 * len(grids):
        fail("clock rows: %d, expected 3 frequencies x %d grids" % (len(band), len(grids)))
    b.add("NPosTempMax", max(band), "plain", "Sec. 3.2 index-only, model-free clock (high)")
    b.add("NPosTempMin", min(band), "plain", "Sec. 3.2 index-only, model-free clock (low)")

    # ---------------------------------------------------------------- tables
    t1a = []
    for cue in CUES:
        s = summary[cue]
        t1a.append("%s\t%.3f\t%.3f\t%.3f\t%d / %d" % (
            CUE_ROW[cue], s["fixed"], s["learned"], s["model"],
            s["above"], s["below"]))
    tables["Table 1  cue, fixed (3 DSP), learned (8 codec), models (%d main-map), "
           "above / below (of %d main-map pairings)"
           % (len(main_models), len(MAIN_PAIRED))] = t1a
    # Table 1 caption: what the three MusicGen pairings would add (same interval
    # rule): first three cues (synchronised tokens); temporal, synchronised and native
    mg_first3 = [cell_of[(k, c)]["resolves"] for k in CASE for c in ("freq", "harm", "onset")]
    mg_sync = [cell_of[(k, "temp")]["resolves"] for k in CASE]
    nat0 = [nat["native"][k]["0"] for k in MG]
    t1c = (mg_first3.count("above"), mg_first3.count("below"),
           mg_sync.count("above"), mg_sync.count("below"),
           sum(c["dE_auc_lo"] > 0 for c in nat0), sum(c["dE_auc_hi"] < 0 for c in nat0))
    tables["Table 1 caption  counts at read-out offset 0; MusicGen-S/M/L would add: first three "
           "cues above, below; temporal synchronised above, below; temporal native above, below"] = [
        "\t".join("%d" % v for v in t1c)]
    tables["Table 2 caption  ticked pairings outside MusicGen, unpaired encoders, "
           "main-map checkpoints in the rate check"] = ["%d\t%d\t%d" % t2]

    def mgrow(label, vals):
        return label + "\t" + "\t".join("%+.3f" % v for v in vals)
    # MusicGen case study (Sec. 3.3); exact AUC; synchronised tokens unless native
    cs = [mgrow("Delta_E %s, sync. tokens" % CUE_KEY[c], [cell_of[(k, c)]["delta"] for k in MG])
          for c in CUES]
    cs += [mgrow("temporal, sync. tokens, offset 0, AUC",
                 [primary["cells"]["%s|temp_proximity" % k]["point"]["auc"] for k in MG]),
           mgrow("temporal, read-out +1 frame, AUC", rows[("auc", 1)]["musicgen_temporal"]),
           mgrow("Delta_E temporal, offset 0", [paired_de[(k, 0)] for k in MG]),
           mgrow("Delta_E temporal, +1 frame (front-end shifted likewise)",
                 [paired_de[(k, 1)] for k in MG]),
           mgrow("temporal, native schedule, AUC", [nat["native"][k]["0"]["auc"] for k in MG]),
           mgrow("Delta_E temporal, native schedule", [nat["native"][k]["0"]["dE_auc"] for k in MG])]
    tables["MusicGen case study (Sec. 3.3)  MusicGen-S, -M, -L; exact AUC; Delta_E vs EnCodec RVQ"] = cs

    flagged_anchor = {e["cell"].split("|")[0] for e in cr["flags"]["cluster_audit_failures"]}
    flagged_method = {e.split("|")[0] for e in cr["flags"]["non_bca_intervals"]}
    arm_prefix = {"cd_noise": "noise__", "cd_shuffle": "shuffle__", "cd_silence": "clicks__"}
    def cr_row(key, label):
        out = []
        for arm, _ in CR_ARMS:
            c = cr["cells"][key][arm]
            txt = "%+.3f" % c["rho"]
            if c["lo"] > 0 or c["hi"] < 0:
                txt = "*" + txt
            tag = arm_prefix.get(arm, "") + key
            if c["n_clusters"] != 24:
                txt += "a"
            if c["method"] != "bca":
                txt += "b"
            if (tag in flagged_anchor) != (c["n_clusters"] != 24) or \
               (tag in flagged_method) != (c["method"] != "bca"):
                fail("Table 3 flags disagree with the stored flag list at %s" % tag)
            out.append(txt)
        s = sel["cells"][key]
        dtxt = "%+.3f" % s["delta"]
        if s["resolves_above"] or s["resolves_below"]:
            dtxt = "*" + dtxt
        out[1:1] = ["%+.3f" % s["trajectory"], dtxt]
        if coarse[(key, "temp")]:
            label += " (dagger)"
        return label + "\t" + "\t".join(out)
    t3 = [cr_row(k, lab) for k, lab in CR_MAIN_ROWS]
    t3.append("D, median (%d main-map checkpoints)\t---\t---\t---\t" % len(main_ckpts) + "\t".join(
        "%+.3f" % main_d[arm]["median"] for arm, _ in CR_ARMS[1:]))
    t3 += [cr_row(k, lab) for k, lab in CR_CASE_ROWS]
    if set(sel["cells"]) != set(cr["checkpoints"]) | set(cr["front_ends"]):
        fail("index_selectivity.json and content_replacement.json read-outs differ")
    tables["Table 3 caption  checkpoints and front-ends in the artifact"] = [
        "%d\t%d" % (len(cr["checkpoints"]), len(cr["front_ends"]))]
    tables["Table 3  read-out, orig., shared, Delta_I, noise, shuffle, clicks  (* interval excludes 0; "
           "a: 21 of 24 anchors; b: percentile fallback; dagger: coarse frame grid, temporal cue; "
           "rows after the median: MusicGen case study)"] = t3

    fig2 = ["%s\t%s\t%+.3f\t[%+.3f, %+.3f]\t%s%s" % (
        r["checkpoint"], CUE_KEY[r["cue"]], r["delta"], r["lo"], r["hi"], r["resolves"],
        "\tdagger (coarse frame grid)" if r["dagger"] else "") for r in cells
        if r["key"] not in CASE]
    if len(fig2) != 4 * len(MAIN_PAIRED):
        fail("Fig. 2 main-map cells: %d" % len(fig2))
    tables["Fig. 2  checkpoint, cue, Delta_E (exact AUC), 95% interval, resolves, "
           "dagger (coarse frame grid; counted like every cell); main map, " + str(len(fig2)) + " cells"] = fig2
    return b, tables


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--macros", action="store_true", help="print only the macros")
    a = ap.parse_args()
    try:
        b, tables = build()
    except (KeyError, IndexError, TypeError) as exc:
        fail("a result file lacks an expected entry (%s: %s)" % (type(exc).__name__, exc))
    for name, text, where in b.rows:
        print("\\%s\t%s\t%s" % (name, text, where))
    if not a.macros:
        for title, lines in tables.items():
            print("\n# " + title)
            for line in lines:
                print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
