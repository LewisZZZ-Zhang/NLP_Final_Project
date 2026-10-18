"""Strict, traceable ARNOR conversion. Character offsets are half-open."""
import json, re, hashlib, unicodedata
from pathlib import Path

def digest(x):
    return hashlib.sha256(x.encode('utf-8')).hexdigest()

def normalize_name(x):
    return ' '.join(unicodedata.normalize('NFC', x).split())

def normalize_label(x):
    if x is None or str(x).strip().lower() in ('', 'none', 'null', 'na'): return 'NA'
    return '/'+str(x).strip().lstrip('/').replace('deceasedperson/', 'deceased_person/')

def parse_noise(x):
    if isinstance(x,bool): return x
    if isinstance(x,int) and x in (0,1): return bool(x)
    if isinstance(x,str) and x.strip().lower() in ('true','false','0','1'): return x.strip().lower() in ('true','1')
    return None

def spans(text,name):
    if not name.strip(): return []
    pattern=r'\s+'.join(re.escape(t) for t in name.split())
    return [[m.start(),m.end()] for m in re.finditer(r'(?<!\w)'+pattern+r'(?!\w)',text)]

def read_objects(path):
    text=Path(path).read_text(encoding='utf-8')
    return json.loads(text) if text.lstrip().startswith('[') else [json.loads(l) for l in text.splitlines() if l.strip()]

def convert_split(path, common_norm_set=None):
    path=Path(path); split=path.stem; out=[]
    for line,obj in enumerate(read_objects(path),1):
        text=obj.get('sentText',obj.get('text',''))
        sid=digest(json.dumps([obj.get('articleId'),obj.get('sentId'),text],ensure_ascii=False))
        rms=obj.get('relationMentions')
        if rms is None: rms=[obj] if ('relation' in obj or 'label' in obj) else []
        seen={}
        for i,rm in enumerate(rms):
            h=rm.get('em1Text',rm.get('em1','')); t=rm.get('em2Text',rm.get('em2',''))
            hs,ts=spans(text,h),spans(text,t)
            status='valid' if len(hs)==len(ts)==1 else 'ambiguous_span' if hs and ts else 'missing_span'
            key=json.dumps(rm,sort_keys=True,ensure_ascii=False)
            iid=f'{split}:{line}:{i}'
            r={'instance_id':iid,'source_sentence_id':sid,'sentence_text_id':digest(normalize_name(text)),
               'mention_id':f'{sid}:{i}','source_split':split,'source_line':line,'source_mention_index':i,
               'articleId':obj.get('articleId'),'sentId':obj.get('sentId'),'text':text,
               'relation':normalize_label(rm.get('label',rm.get('relation'))),'source_relation':rm.get('label',rm.get('relation')),
               'is_noise':parse_noise(rm.get('is_noise',obj.get('is_noise'))),'source_is_noise':rm.get('is_noise',obj.get('is_noise')),
               'validation_status':status,'duplicate_of':seen.get(key),
               'h':{'name':h,'id':'surface:'+digest(normalize_name(h)),'pos':hs[0] if len(hs)==1 else None,'candidate_spans':hs},
               't':{'name':t,'id':'surface:'+digest(normalize_name(t)),'pos':ts[0] if len(ts)==1 else None,'candidate_spans':ts}}
            seen.setdefault(key,iid); out.append(r)
    return out

def write_jsonl(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(f'Refusing to overwrite {path}')
    with path.open('w') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')

def convert_file(input_path,out_path,rel2id_path):
    rows=convert_split(input_path); write_jsonl(out_path,rows)
    rels=['NA']+sorted({r['relation'] for r in rows}-{'NA'})
    with open(rel2id_path,'x') as f: json.dump(dict(zip(rels,range(len(rels)))),f,indent=2)

def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--arnor-dir',default='data/arnor_dataset-2.0.0'); p.add_argument('--out-dir','--outdir',default='data/arnor_benchmark_v2'); p.add_argument('--inputs',nargs='+'); a=p.parse_args()
    for src in (list(map(Path,a.inputs)) if a.inputs else sorted(Path(a.arnor_dir).glob('*.json'))):
        convert_file(src,Path(a.out_dir)/f'arnor_{src.stem}.txt',Path(a.out_dir)/f'arnor_{src.stem}_rel2id.json')
if __name__=='__main__': main()
