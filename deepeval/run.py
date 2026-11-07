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

def read_cases(path=CASES):
    with Path(path).open(newline='') as f:
        cases = list(csv.DictReader(f))
    for row_number, case in enumerate(cases, 2):
        customer_id = case.get('customer_id') or ''
        if not customer_id.startswith('CUS-') or any(c.isspace() for c in customer_id):
            raise ValueError(
                f"{path}: row {row_number} ({case.get('case_id', 'unknown')}): "
                f"invalid customer_id {customer_id!r}; expected a seeded customer ID starting with 'CUS-'"
            )
    return cases

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
    if not r.ok: r.raise_for_status()
    return {x.get('id') for x in r.json()}

def reset_case_state(ticket_id):
    """Remove only this test ticket's writes, then restore canonical seed rows without truncating DB."""
    from SupportOpsBot.runtime.shared.config import load_env
    import psycopg
    load_env()
    dsn=os.environ.get('SUPPORT_OPS_DATABASE_URL')
    if not dsn: raise RuntimeError('SUPPORT_OPS_DATABASE_URL is required to reset case fixtures')
    with psycopg.connect(dsn,connect_timeout=5) as conn, conn.cursor() as cur:
        cur.execute("SELECT operation, result FROM backend_writes WHERE ticket_id=%s",(ticket_id,))
        for operation, result in cur.fetchall():
            write_id=(result or {}).get('write_id')
            if not write_id: continue
            if operation == 'refund':
                cur.execute("DELETE FROM refunds WHERE id=%s",(write_id,))
            elif operation == 'return':
                cur.execute("DELETE FROM return_items WHERE return_id=%s",(write_id,))
                cur.execute("DELETE FROM returns WHERE id=%s",(write_id,))
        cur.execute("DELETE FROM backend_writes WHERE ticket_id=%s",(ticket_id,))
        cur.execute("DELETE FROM support_tickets WHERE id=%s",(ticket_id,))
    # Upsert the committed fixtures (including orders/items/shipments) without deleting unrelated data.
    from scripts.seed_db import seed
    seed(dsn)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--api',default=os.getenv('DEEPEVAL_API','http://127.0.0.1:8000'))
    p.add_argument('--setup',action='append',choices=SETUP_IDS)
    p.add_argument('--case',action='append',help='Case ID (for example TKT-EVAL-01)')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--judge-model',default=os.getenv('DEEPEVAL_JUDGE_MODEL','ollama/ornith:9b'))
    p.add_argument('--output',type=Path,default=Path('deepeval/results.jsonl'))
    p.add_argument('--cases',type=Path,default=CASES,help='Case CSV (default deepeval/cases.csv); e.g. a robustness variant file')
    a=p.parse_args(); cases=read_cases(a.cases); setups=a.setup or ['s1-a']
    if a.case: cases=[c for c in cases if c['ticket_id'] in a.case or c['case_id'] in a.case]
    if not cases: p.error('no cases selected')
    if a.dry_run:
        print(json.dumps({'case_count':len(cases),'setup_count':len(setups),'runs':len(cases)*len(setups),'setups':setups,'cases':[c['ticket_id'] for c in cases]},indent=2)); return 0
    customer_ids = preflight(a.api)
    missing = sorted({c['customer_id'] for c in cases} - customer_ids)
    if missing: raise RuntimeError(f'customers missing from API database: {missing}')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('a') as out:
      for c in cases:
       expected_by_setup=json.loads(c.get('expected_tool_calls_by_setup') or '{}')
       for setup in setups:
        reset_case_state(c['ticket_id'])
        expected=expected_by_setup.get(setup, json.loads(c.get('expected_tools') or '[]'))
        turns=json.loads(c.get('Turns') or c.get('turns') or '[]') or [c['Query']]
        history=[]; all_tools=[]; all_events=[]; data={}; writes=[]
        for message in turns:
         payload={'customer_message':message,'customer_id':c['customer_id'],'ticket_id':c['ticket_id'],'setup':setup,'clock_start':c['clock_start'],'history':history}
         response=requests.post(a.api.rstrip('/')+'/chat',json=payload,timeout=900); response.raise_for_status(); data=response.json()
         all_tools.extend(data.get('tool_calls',[])); all_events.extend(data.get('trace',{}).get('events',[])); writes.extend(data.get('writes',[]))
         history=data.get('history',history)
        steps=events_as_steps(all_events)
        # Stable ticket IDs intentionally exercise idempotent replay across setup variants.
        row={'case_id':c['case_id'],'setup':setup,'ticket_id':c['ticket_id'],'replayed':bool(any(w.get('replayed') for w in writes)),'input':'\n'.join(turns),'ground_truth':c['Ground Truth'],'actual_output':data.get('reply',''),'ticket_status':data.get('ticket_status'),'expected_ticket_status':c['expected_ticket_status'],'expected_write':json.loads(c['expected_write']),'writes':writes,'expected_tools':expected,'tools_called':[{'name':x.get('tool'),'arguments':x.get('input',{})} for x in all_tools],'trajectory':steps,'judge_model':a.judge_model,'trace':{'events':all_events}}
        out.write(json.dumps(row,default=str)+'\n'); out.flush(); print(f"{c['ticket_id']} {setup}: status={data.get('ticket_status')} tools={len(row['tools_called'])}")
        time.sleep(.05)
    print(f'Raw scenario results written to {a.output}. Run deepeval test/evaluation commands after configuring the judge.')
    return 0
if __name__=='__main__': raise SystemExit(main())
