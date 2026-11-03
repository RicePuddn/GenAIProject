#!/usr/bin/env python3
"""Sequential DeepEval scenario runner for the seeded evaluation tickets."""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from pathlib import Path
from datetime import datetime
import requests

ROOT=Path(__file__).resolve().parents[1]
# Direct script execution puts deepeval/, not the repository root, on sys.path.
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
CASES=Path(__file__).with_name('cases.csv')
SETUP_IDS=[f's{s}-{v}' for s in (1,2,5,6) for v in 'ab']+[f's{s}-{v}' for s in (3,4,7,8) for v in 'abcd']

def read_cases():
    with CASES.open(newline='') as f: return list(csv.DictReader(f))

def events_as_steps(events):
    """Preserve API event order and turn trace tool events into a stable trajectory view."""
    steps=[]
    for e in events:
        item={'kind':e.get('kind'),'agent':e.get('agent'),'node':e.get('node')}
        if e.get('kind') in ('tool_call_requested','tool_start','tool_end'):
            item.update(tool=e.get('tool'),arguments=e.get('input'),output=e.get('output'),call_id=e.get('call_id'))
        elif e.get('kind') in ('plan_created','plan','llm_request','handoff','review'):
            item.update({k:v for k,v in e.items() if k not in ('t','kind')})
        steps.append(item)
    return steps

def preflight(base):
    # Match the API's config loader: shell variables win, then values come from repo .env.
    from SupportOpsBot.runtime.shared.config import load_env
    load_env()
    r=requests.get(base.rstrip('/')+'/customers',timeout=15); r.raise_for_status()
    if not any(x.get('id')=='CUS-EVAL-001' for x in r.json()): raise RuntimeError('CUS-EVAL-001 missing from API database')
    # Read-only direct checks ensure all ten seeded tickets/orders exist and belong to the evaluation customer.
    import psycopg
    dsn=os.environ.get('SUPPORT_OPS_DATABASE_URL')
    if not dsn: raise RuntimeError('SUPPORT_OPS_DATABASE_URL is required for seeded-record preflight')
    with psycopg.connect(dsn,connect_timeout=5,options='-c default_transaction_read_only=on') as c, c.cursor() as cur:
        cur.execute("SELECT count(*) FROM support_tickets WHERE id LIKE %s AND customer_id=%s",('TKT-EVAL-%','CUS-EVAL-001'))
        if cur.fetchone()[0] != 10: raise RuntimeError('Expected all ten evaluation tickets for CUS-EVAL-001')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--api',default=os.getenv('DEEPEVAL_API','http://127.0.0.1:8000'))
    p.add_argument('--setup',action='append',choices=SETUP_IDS)
    p.add_argument('--case',action='append',help='Case ID (for example TKT-EVAL-01)')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--judge-model',default=os.getenv('DEEPEVAL_JUDGE_MODEL','ollama/ornith:9b'))
    p.add_argument('--output',type=Path,default=Path('deepeval/results.jsonl'))
    a=p.parse_args(); cases=read_cases(); setups=a.setup or SETUP_IDS
    if a.case: cases=[c for c in cases if c['ticket_id'] in a.case or c['case_id'] in a.case]
    if not cases: p.error('no cases selected')
    if a.dry_run:
        print(json.dumps({'case_count':len(cases),'setup_count':len(setups),'runs':len(cases)*len(setups),'setups':setups,'cases':[c['ticket_id'] for c in cases]},indent=2)); return 0
    preflight(a.api)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('a') as out:
      for c in cases:
       expected=json.loads(c['expected_tool_calls_by_setup'])[None] if False else json.loads(c['expected_tool_calls_by_setup'])
       for setup in setups:
        payload={'customer_message':c['Query'],'customer_id':c['customer_id'],'ticket_id':c['ticket_id'],'setup':setup,'clock_start':c['clock_start'],'history':[]}
        response=requests.post(a.api.rstrip('/')+'/chat',json=payload,timeout=900); response.raise_for_status(); data=response.json()
        steps=events_as_steps(data.get('trace',{}).get('events',[]))
        # Stable ticket IDs intentionally exercise idempotent replay across setup variants.
        row={'case_id':c['case_id'],'setup':setup,'ticket_id':c['ticket_id'],'replayed':bool(data.get('writes') and any(w.get('replayed') for w in data['writes'])),'input':c['Query'],'ground_truth':c['Ground Truth'],'actual_output':data.get('reply',''),'ticket_status':data.get('ticket_status'),'expected_ticket_status':c['expected_ticket_status'],'expected_write':json.loads(c['expected_write']),'writes':data.get('writes',[]),'expected_tools':expected[setup],'tools_called':[{'name':x.get('tool'),'arguments':x.get('input',{})} for x in data.get('tool_calls',[])],'trajectory':steps,'judge_model':a.judge_model,'trace':data.get('trace',{})}
        out.write(json.dumps(row,default=str)+'\n'); out.flush(); print(f"{c['ticket_id']} {setup}: status={data.get('ticket_status')} tools={len(row['tools_called'])}")
        time.sleep(.05)
    print(f'Raw scenario results written to {a.output}. Run deepeval test/evaluation commands after configuring the judge.')
    return 0
if __name__=='__main__': raise SystemExit(main())
