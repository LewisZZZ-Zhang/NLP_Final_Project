"""Shared strict loader: legacy local IDs require explicit reconversion."""
import json,sys,logging
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_data import parse_noise, normalize_name

def load_dataset(path,limit_bags=None):
    from model import SentenceInstance
    bags={}; order=[]; quarantined=0
    with Path(path).open() as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            if r.get('validation_status') not in ('valid','ambiguous_span','missing_span'):
                raise ValueError('Input must be reconverted with scripts/benchmark_data.py (V2).')
            if r['validation_status']!='valid':
                quarantined+=1
                continue
            text=r['text']; h,t=r['h'],r['t']
            for e in (h,t):
                start,end=e['pos']
                if not (0<=start<end<=len(text)) or normalize_name(text[start:end])!=normalize_name(e['name']): raise ValueError('Invalid span')
            key=json.dumps([r['relation'],h['id'],t['id']],ensure_ascii=False)
            if key not in bags:
                if limit_bags and len(order)>=limit_bags: continue
                bags[key]=[]; order.append(key)
            r['is_noise']=parse_noise(r.get('is_noise'))
            bags[key].append(SentenceInstance(r['relation'],text,text.lower(),h['name'],t['name'],tuple(h['pos']),tuple(t['pos']),r))
    if quarantined: logging.getLogger('nyt10m_filter').warning('%d invalid/ambiguous records quarantined; inspect input validation_status and candidate_spans',quarantined)
    return bags,order
