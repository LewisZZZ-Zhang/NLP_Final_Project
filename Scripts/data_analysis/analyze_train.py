import os, json, csv
import re
from collections import Counter, defaultdict
from pathlib import Path

def read_arnor_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)  # list of samples
        except json.JSONDecodeError:
            f.seek(0)
            data = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return data

def read_opennre_txt(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 4:
                sent, rel, h, t = parts[:4]
                data.append({'sentence': sent, 'relation': rel, 'h': h, 't': t})
    return data

def simple_tokenize(text):
    return [t for t in re.split(r"\W+", text.lower()) if t]

def ensure_reports_dir():
    # Create base and subfolders for better readability
    Path("reports").mkdir(exist_ok=True)
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    Path("reports/stats").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    Path("reports/debug").mkdir(parents=True, exist_ok=True)

def count_word_ngrams(samples, fmt="arnor", topk=200):
    counter1 = Counter()
    counter2 = Counter()
    for s in samples:
        if fmt == "arnor":
            toks = [t.lower() for t in s.get("tokens", [])]
        else:
            toks = simple_tokenize(s.get("sentence", ""))
        counter1.update(toks)
        counter2.update(zip(toks, toks[1:]))
    ensure_reports_dir()
    with open("reports/tables/train_word_freq.csv","w",encoding="utf-8") as f:
        f.write("token,count\n")
        for w,c in counter1.most_common(topk):
            f.write(f"{w},{c}\n")
    with open("reports/tables/train_bigram_freq.csv","w",encoding="utf-8") as f:
        f.write("bigram,count\n")
        for (a,b),c in counter2.most_common(topk):
            f.write(f"{a} {b},{c}\n")
    return counter1, counter2

# ---- Additional analysis helpers for ARNOR ----

def normalize_relation(rel):
    if rel is None:
        return "NA"
    if isinstance(rel, str):
        r = rel.strip()
        if r.lower() in {"", "na", "none", "no_relation", "n/a"}:
            return "NA"
        return r
    return str(rel)

def arnor_to_canonical(raw_samples, max_samples=None):
    """
    Convert ARNOR sentence-level JSON (with relationMentions/entityMentions) into
    canonical mention-level samples: {tokens, h:{name}, t:{name}, relation}.
    """
    out = []
    count = 0
    for s in raw_samples:
        sent = s.get("sentText", "")
        if not isinstance(sent, str):
            continue
        toks = simple_tokenize(sent)
        rms = s.get("relationMentions") or []
        if isinstance(rms, list):
            for rm in rms:
                rel = rm.get("label", "NA")
                h_text = rm.get("em1Text", "")
                t_text = rm.get("em2Text", "")
                out.append({
                    "tokens": toks,
                    "h": {"name": h_text},
                    "t": {"name": t_text},
                    "relation": rel,
                })
                count += 1
                if max_samples and count >= max_samples:
                    return out
        # If no relation mention, we skip NA construction from all entity pairs to avoid explosion
    return out

def get_stopwords():
    # A minimal English stopword list; extend if needed
    return set('''a an the and or of in on to for with at by from as is are was were be been being this that these those it its if then else than so such not no nor but into over under up down out about above below between within without do does did doing have has had having can could should would may might must will shall'''.split())

def write_word_freq_no_stop(counter1, topk=200):
    sw = get_stopwords()
    filtered = [(w,c) for w,c in counter1.items() if w not in sw]
    filtered.sort(key=lambda x: x[1], reverse=True)
    ensure_reports_dir()
    with open("reports/tables/train_word_freq_no_stop.csv","w",encoding="utf-8") as f:
        f.write("token,count\n")
        for w,c in filtered[:topk]:
            f.write(f"{w},{c}\n")

def relation_and_entity_stats(samples, fmt="arnor"):
    rel_cnt = Counter()
    h_chars, t_chars = [], []
    for s in samples:
        rel_raw = s.get("relation") or s.get("rel")
        rel = normalize_relation(rel_raw)
        rel_cnt[rel] += 1
        if fmt == "arnor":
            h = (s.get("h") or {}).get("name", "")
            t = (s.get("t") or {}).get("name", "")
        else:
            h = s.get("h", "")
            t = s.get("t", "")
        h_chars.append(len(h))
        t_chars.append(len(t))
    ensure_reports_dir()
    with open("reports/tables/relation_freq.csv","w",encoding="utf-8") as f:
        f.write("relation,count\n")
        for r,c in rel_cnt.most_common():
            f.write(f"{r},{c}\n")
    with open("reports/stats/entity_length_stats.txt","w",encoding="utf-8") as f:
        if h_chars:
            f.write(f"h_chars_count={len(h_chars)}\n")
            f.write(f"h_chars_mean={sum(h_chars)/len(h_chars):.3f}\n")
        if t_chars:
            f.write(f"t_chars_count={len(t_chars)}\n")
            f.write(f"t_chars_mean={sum(t_chars)/len(t_chars):.3f}\n")
    return rel_cnt, {"h_chars": h_chars, "t_chars": t_chars}

def find_entity_pos(tokens, entity_name):
    name_toks = simple_tokenize(entity_name)
    if not name_toks:
        return None
    toks = [t.lower() for t in tokens]
    nt = len(name_toks)
    for i in range(0, len(toks)-nt+1):
        if toks[i:i+nt] == name_toks:
            return (i, i+nt)
    return None

def _percentiles(arr, ps):
    if not arr:
        return {p: None for p in ps}
    arr_sorted = sorted(arr)
    n = len(arr_sorted)
    out = {}
    for p in ps:
        if n == 1:
            out[p] = arr_sorted[0]
            continue
        rank = p * (n - 1)
        lo = int(rank)
        hi = min(lo + 1, n - 1)
        frac = rank - lo
        out[p] = arr_sorted[lo] * (1 - frac) + arr_sorted[hi] * frac
    return out

def distance_and_length_stats(samples, fmt="arnor"):
    dists, sent_lens = [], []
    for s in samples:
        if fmt == "arnor":
            toks = s.get("tokens", [])
            sent_lens.append(len(toks))
            hpos = (s.get("h") or {}).get("pos")
            tpos = (s.get("t") or {}).get("pos")
            if not hpos or not tpos:
                hpos = find_entity_pos(toks, (s.get("h") or {}).get("name",""))
                tpos = find_entity_pos(toks, (s.get("t") or {}).get("name",""))
        else:
            sent = s.get("sentence","")
            toks = simple_tokenize(sent)
            sent_lens.append(len(toks))
            hpos = find_entity_pos(toks, s.get("h",""))
            tpos = find_entity_pos(toks, s.get("t",""))
        if hpos and tpos:
            d = abs(hpos[0] - tpos[0])
            dists.append(d)
    ensure_reports_dir()
    with open("reports/stats/distance_stats.txt","w",encoding="utf-8") as f:
        f.write(f"dist_count={len(dists)}\n")
        if dists:
            f.write(f"dist_mean={sum(dists)/len(dists):.3f}\n")
            for p,v in _percentiles(dists, [0.5,0.7,0.8,0.9,0.95]).items():
                f.write(f"dist_p{int(p*100)}={v:.3f}\n")
        f.write(f"sent_count={len(sent_lens)}\n")
        if sent_lens:
            f.write(f"sent_mean={sum(sent_lens)/len(sent_lens):.3f}\n")
            for p,v in _percentiles(sent_lens, [0.5,0.7,0.8,0.9,0.95]).items():
                f.write(f"sent_p{int(p*100)}={v:.3f}\n")
    return dists, sent_lens

def bag_stats(samples, fmt="arnor"):
    bags = defaultdict(list)
    for s in samples:
        if fmt == "arnor":
            h = (s.get("h") or {}).get("name","")
            t = (s.get("t") or {}).get("name","")
            rel = s.get("relation","NA")
        else:
            h = s.get("h","")
            t = s.get("t","")
            rel = s.get("relation","NA") or s.get("rel","NA")
        key = (h, t)
        bags[key].append(rel)
    sizes = [len(v) for v in bags.values()]
    inconsistency = sum(1 for v in bags.values() if len(set(v))>1)
    ensure_reports_dir()
    with open("reports/stats/bag_size_stats.txt","w",encoding="utf-8") as f:
        f.write(f"num_bags={len(bags)}\n")
        f.write(f"mean_size={(sum(sizes)/len(sizes)) if sizes else 0:.3f}\n")
        f.write(f"inconsistency_count={inconsistency}\n")
    return sizes, inconsistency

def na_ratio(samples):
    total = len(samples)
    na = 0
    for s in samples:
        rel = normalize_relation(s.get("relation") or s.get("rel"))
        if rel == "NA":
            na += 1
    ratio = na/total if total else 0
    ensure_reports_dir()
    with open("reports/stats/na_ratio.txt","w",encoding="utf-8") as f:
        f.write(f"total={total}\nna={na}\nratio={ratio:.4f}\n")
    return ratio

def try_plot_histograms(dists, sent_lens, rel_cnt):
    try:
        import matplotlib.pyplot as plt
        ensure_reports_dir()
        if dists:
            # 延迟导入 numpy，不存在则使用简单 range
            try:
                import numpy as np
            except ImportError:
                np = None
            max_dist = max(dists)
            p95 = _percentiles(dists, [0.95])[0.95]
            # 原始完整直方图
            plt.figure(figsize=(6,4))
            bins = min(60, max_dist + 1)
            plt.hist(dists, bins=bins, color="#4C78A8")
            plt.title("Entity Distance Distribution (Full)")
            plt.xlabel("Distance (tokens)")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig("reports/figures/distance_hist.png", dpi=150)
            plt.close()
            # 裁剪到 p95 以内的直方图，突出主体分布
            trimmed = [x for x in dists if x <= p95]
            plt.figure(figsize=(6,4))
            if np:
                bins_trim = np.arange(0, int(p95)+2)
            else:
                bins_trim = list(range(0, int(p95)+2))
            plt.hist(trimmed, bins=bins_trim, color="#4C78A8")
            plt.title(f"Entity Distance Distribution (<= p95={int(p95)})")
            plt.xlabel("Distance (tokens)")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig("reports/figures/distance_hist_trimmed.png", dpi=150)
            plt.close()
        if sent_lens:
            try:
                import numpy as np
            except ImportError:
                np = None
            max_len = max(sent_lens)
            p95l = _percentiles(sent_lens, [0.95])[0.95]
            plt.figure(figsize=(6,4))
            bins = min(60, max_len + 1)
            plt.hist(sent_lens, bins=bins, color="#72B7B2")
            plt.title("Sentence Length Distribution (Full)")
            plt.xlabel("Length (tokens)")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig("reports/figures/sent_length_hist.png", dpi=150)
            plt.close()
            trimmed_l = [x for x in sent_lens if x <= p95l]
            plt.figure(figsize=(6,4))
            if np:
                bins_trim_l = np.arange(0, int(p95l)+2)
            else:
                bins_trim_l = list(range(0, int(p95l)+2))
            plt.hist(trimmed_l, bins=bins_trim_l, color="#72B7B2")
            plt.title(f"Sentence Length Distribution (<= p95={int(p95l)})")
            plt.xlabel("Length (tokens)")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig("reports/figures/sent_length_hist_trimmed.png", dpi=150)
            plt.close()
        if rel_cnt:
            top = rel_cnt.most_common(20)
            labels = [r for r,_ in top]
            values = [c for _,c in top]
            plt.figure(figsize=(8,4.5))
            plt.bar(range(len(values)), values, color="#E45756")
            plt.title("Top-20 Relations")
            plt.xlabel("Relation")
            plt.ylabel("Count")
            plt.xticks(range(len(labels)), labels, rotation=60, ha='right', fontsize=8)
            plt.tight_layout()
            plt.savefig("reports/figures/relation_top20.png", dpi=150)
            plt.close()
    except Exception:
        # Ignore if matplotlib is unavailable or plotting fails
        pass

def _format_table(headers, rows):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"]*len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(map(str, r)) + " |")
    return "\n".join(out)

def write_quick_report(
    num_samples,
    rel_cnt,
    top_tokens,
    top_bigrams,
    dist_percentiles,
    sent_percentiles,
    na_ratio_value,
    bag_summary,
    extra_figures=False,
):
    ensure_reports_dir()
    lines = []
    lines.append("# Train Data Profile (ARNOR)")
    lines.append("")
    lines.append(f"- Total samples: `{num_samples}`")
    lines.append(f"- Generated: automatic")
    lines.append("")

    # 概览链接
    lines += [
        "- Word frequencies: `reports/tables/train_word_freq.csv`, stopwords-removed: `reports/tables/train_word_freq_no_stop.csv`",
        "- Bigrams: `reports/tables/train_bigram_freq.csv`",
        "- Relation distribution: `reports/tables/relation_freq.csv`",
        "- Entity length stats: `reports/stats/entity_length_stats.txt`",
        "- Distances and sentence length: `reports/stats/distance_stats.txt`",
        "- NA ratio: `reports/stats/na_ratio.txt`",
        "- Bag stats: `reports/stats/bag_size_stats.txt`",
        "",
    ]

    # 关系分布 Top-20
    lines.append("## Top-20 Relations")
    rel_rows = [(r, c) for r,c in rel_cnt.most_common(20)]
    if rel_rows:
        lines.append(_format_table(["Relation", "Count"], rel_rows))
    if os.path.exists("reports/figures/relation_top20.png"):
        lines.append("")
        lines.append("![Top Relations](figures/relation_top20.png)")
    lines.append("")

    # 词频与二元短语 Top-20（去停用词）
    lines.append("## Trigger Tokens and Bigrams (Top-20)")
    tok_rows = [(w,c) for w,c in top_tokens[:20]]
    if tok_rows:
        lines.append("**Top Tokens (stopwords removed)**")
        lines.append(_format_table(["Token", "Count"], tok_rows))
        lines.append("")
    bi_rows = [(f"{a} {b}", c) for (a,b),c in top_bigrams[:20]]
    if bi_rows:
        lines.append("**Top Bigrams**")
        lines.append(_format_table(["Bigram", "Count"], bi_rows))
        lines.append("")

    # 距离与句长分位数
    lines.append("## Distance and Sentence Length Percentiles")
    dist_rows = []
    for p in [50,70,80,90,95]:
        key = p/100
        v = dist_percentiles.get(key)
        if v is not None:
            dist_rows.append((f"p{p}", f"{v:.2f}"))
    if dist_rows:
        lines.append("**Entity distance (tokens)**")
        lines.append(_format_table(["Percentile", "Value"], dist_rows))
        lines.append("")

    sent_rows = []
    for p in [50,70,80,90,95]:
        key = p/100
        v = sent_percentiles.get(key)
        if v is not None:
            sent_rows.append((f"p{p}", f"{v:.2f}"))
    if sent_rows:
        lines.append("**Sentence length (tokens)**")
        lines.append(_format_table(["Percentile", "Value"], sent_rows))
        lines.append("")
    if os.path.exists("reports/figures/distance_hist.png"):
        lines.append("![Distance Histogram](figures/distance_hist.png)")
    if os.path.exists("reports/figures/sent_length_hist.png"):
        lines.append("![Sentence Length Histogram](figures/sent_length_hist.png)")
    if extra_figures and os.path.exists("reports/figures/distance_hist_trimmed.png"):
        lines.append("![Distance Histogram Trimmed](figures/distance_hist_trimmed.png)")
    if extra_figures and os.path.exists("reports/figures/sent_length_hist_trimmed.png"):
        lines.append("![Sentence Length Histogram Trimmed](figures/sent_length_hist_trimmed.png)")
    lines.append("")

    # NA 与 bag
    lines.append("## NA and Bag Overview")
    lines.append(f"- NA ratio: `{na_ratio_value:.4f}`")
    if bag_summary:
        lines.append(f"- Number of bags: `{bag_summary.get('num_bags', 'N/A')}`")
        lines.append(f"- Mean bag size: `{bag_summary.get('mean_size', 'N/A')}`")
        lines.append(f"- Bags with relation inconsistency: `{bag_summary.get('inconsistency_count', 'N/A')}`")
    lines.append("")

    # 阈值建议（基于分位数）
    lines.append("## Data-Driven Threshold Suggestions")
    D = dist_percentiles.get(0.80) or dist_percentiles.get(0.90)
    L = sent_percentiles.get(0.90) or sent_percentiles.get(0.80)
    if D is not None and L is not None:
        lines.append(f"- Suggested entity distance threshold D: use p80 ≈ `{D:.0f}` (or p90 depending on task)")
        lines.append(f"- Suggested sentence length threshold L: use p90 ≈ `{L:.0f}` (adjust per task)")
    else:
        lines.append("- Percentiles unavailable; please check data or statistics steps.")
    lines.append("")

    # 下一步规则草案
    lines.append("## Next: Turn Statistics into Rules (Draft)")
    lines.append("- If strong trigger bigram present and distance ≤ D: mark high-confidence/keep.")
    lines.append("- If sentence length > L and distance > D and no trigger word: low-confidence/filter candidate.")
    lines.append("- Long-tail relations: keep only when exclusive trigger words present and distance ≤ D.")
    lines.append("- Bag conflicts: keep only sentences with trigger words and within thresholds; downweight others.")
    lines.append("")

    with open("reports/train_data_profile.md","w",encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def build_global_triggers(samples, min_total=500, min_ratio=0.6, top_k=200):
    """简单触发词挖掘：统计 token 在非NA与全部中的比例，筛选高相关词。"""
    non_na_counter = Counter()
    total_counter = Counter()
    for s in samples:
        rel = s.get("relation")
        toks = s.get("tokens", [])
        if not toks:
            continue
        uniq = set(toks)  # 防止重复位置放大统计，可改成频次统计
        for t in uniq:
            total_counter[t] += 1
            if rel != "NA":
                non_na_counter[t] += 1
    triggers = []
    for tok, tot in total_counter.items():
        if tot < min_total:
            continue
        ratio = non_na_counter[tok] / tot if tot else 0
        if ratio >= min_ratio:
            triggers.append((tok, tot, ratio))
    triggers.sort(key=lambda x: (-x[2], -x[1], x[0]))
    ensure_reports_dir()
    with open("reports/tables/global_triggers.csv", "w", encoding="utf-8") as f:
        f.write("token,total_doc_freq,non_na_ratio\n")
        for tok, tot, ratio in triggers[:top_k]:
            f.write(f"{tok},{tot},{ratio:.4f}\n")
    return {t for t,_,_ in triggers[:top_k]}

def apply_rules(samples, triggers, dist_map, sent_len_map, D=18, L=54, extended_D=24):
    """对样本应用规则并输出概览统计。dist_map/sent_len_map: sample index -> distance/length"""
    summary = Counter()
    for idx, s in enumerate(samples):
        rel = s.get("relation")
        toks = s.get("tokens", [])
        dist = dist_map.get(idx)
        slen = sent_len_map.get(idx, len(toks))
        has_trigger = any(t in triggers for t in toks)
        if rel != "NA" and has_trigger and dist is not None and dist <= D:
            summary['high_confidence'] += 1
        elif has_trigger and dist is not None and dist <= extended_D:
            summary['medium_confidence'] += 1
        elif rel == "NA" and has_trigger and dist is not None and dist <= D:
            summary['na_possible_missed'] += 1
        elif dist is not None and dist > extended_D and slen > L and not has_trigger:
            if rel == "NA":
                summary['na_noise'] += 1
            else:
                summary['low_confidence_candidate'] += 1
        else:
            summary['other'] += 1
    ensure_reports_dir()
    with open("reports/stats/rule_effect_overview.txt", "w", encoding="utf-8") as f:
        total = len(samples)
        for k in ['high_confidence','medium_confidence','na_possible_missed','na_noise','low_confidence_candidate','other']:
            v = summary.get(k,0)
            f.write(f"{k}={v}\n")
        f.write(f"total={total}\n")
    return summary

if __name__ == "__main__":
    # 固定使用 ARNOR 训练集
    arnor_path = "data/arnor_dataset-2.0.0/train.json"
    if not os.path.exists(arnor_path):
        print("ARNOR training set not found: data/arnor_dataset-2.0.0/train.json")
        raise SystemExit(1)

    raw_samples = read_arnor_json(arnor_path)
    print(f"Loaded {len(raw_samples)} ARNOR sentence items")

    # Convert to mention-level canonical samples
    samples = arnor_to_canonical(raw_samples)
    print(f"Flattened to {len(samples)} relation mentions")

    # Quick debug snapshot to validate fields
    try:
        ensure_reports_dir()
        snap = []
        uniq_rels = set()
        nonempty_tokens = 0
        for s in raw_samples[:2000]:
            if isinstance(s, dict):
                uniq_rels.add(s.get("relation") or s.get("rel") or "NA")
                if s.get("tokens"):
                    nonempty_tokens += 1
                if len(snap) < 5:
                    snap.append({
                        "keys": list(s.keys()),
                        "relation": s.get("relation") or s.get("rel"),
                        "tokens_len": len(s.get("tokens", [])),
                        "h_keys": list((s.get("h") or {}).keys()) if isinstance(s.get("h"), dict) else type(s.get("h")).__name__,
                        "t_keys": list((s.get("t") or {}).keys()) if isinstance(s.get("t"), dict) else type(s.get("t")).__name__,
                    })
        # find first item with relationMentions and entityMentions details
        rm_preview = None
        em_preview = None
        for s in raw_samples[:2000]:
            if isinstance(s, dict):
                rms = s.get("relationMentions")
                ems = s.get("entityMentions")
                if isinstance(rms, list) and rms and rm_preview is None:
                    rm = rms[0]
                    rm_preview = {"keys": list(rm.keys())}
                if isinstance(ems, list) and ems and em_preview is None:
                    em = ems[0]
                    em_preview = {"keys": list(em.keys())}
                if rm_preview and em_preview:
                    break
        with open("reports/debug/debug_snapshot.json", "w", encoding="utf-8") as f:
            json.dump({
                "raw_sentence_items": len(raw_samples),
                "canonical_relation_mentions": len(samples),
                "unique_relations_sample": sorted(list(uniq_rels))[:50],
                "nonempty_tokens_in_first_2k": nonempty_tokens,
                "samples_preview": snap,
                "relationMention_first_keys": rm_preview,
                "entityMention_first_keys": em_preview,
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # 1) 词频 / n-gram
    c1, c2 = count_word_ngrams(samples, fmt="arnor", topk=1000)
    # 去停用词后的 top tokens（内存中，用于报告表格）
    sw = get_stopwords()
    top_tokens = [(w,c) for w,c in c1.items() if w not in sw]
    top_tokens.sort(key=lambda x: x[1], reverse=True)
    write_word_freq_no_stop(c1, topk=300)

    # 2) 关系与实体
    rel_cnt, ent_stats = relation_and_entity_stats(samples, fmt="arnor")

    # 3) 距离与句长
    dists, sent_lens = distance_and_length_stats(samples, fmt="arnor")
    dist_ps = _percentiles(dists, [0.5,0.7,0.8,0.9,0.95]) if dists else {}
    sent_ps = _percentiles(sent_lens, [0.5,0.7,0.8,0.9,0.95]) if sent_lens else {}

    # 可选：绘图
    try_plot_histograms(dists, sent_lens, rel_cnt)

    # 构建全局触发词（简易版本）
    triggers = build_global_triggers(samples)

    # 为规则评估准备距离和句长映射
    dist_map = {}
    sent_len_map = {}
    # 重新计算距离（与前面逻辑保持一致）
    for idx, s in enumerate(samples):
        toks = s.get("tokens", [])
        sent_len_map[idx] = len(toks)
        h = (s.get("h") or {}).get("name", "")
        t = (s.get("t") or {}).get("name", "")
        hpos = find_entity_pos(toks, h)
        tpos = find_entity_pos(toks, t)
        if hpos and tpos:
            dist_map[idx] = abs(hpos[0] - tpos[0])

    rule_summary = apply_rules(samples, triggers, dist_map, sent_len_map, D=18, L=54, extended_D=24)

    # 4) bag 粗统计
    sizes, inconsistency = bag_stats(samples, fmt="arnor")
    bag_summary = {
        "num_bags": len(sizes) if sizes else 0,
        "mean_size": (sum(sizes)/len(sizes)) if sizes else 0,
        "inconsistency_count": inconsistency,
    }

    # 5) NA 比例
    ratio = na_ratio(samples)

    # 6) 快速汇总报告
    # Bigram top list for report
    top_bigrams = list(c2.most_common(200))
    write_quick_report(
        num_samples=len(samples),
        rel_cnt=rel_cnt,
        top_tokens=top_tokens,
        top_bigrams=top_bigrams,
        dist_percentiles=dist_ps,
        sent_percentiles=sent_ps,
        na_ratio_value=ratio,
        bag_summary=bag_summary,
        extra_figures=True,
    )
    print("Analysis complete: please check reports/ outputs and train_data_profile.md")
