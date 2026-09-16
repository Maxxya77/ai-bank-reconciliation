import json,re,hashlib,sqlite3
from itertools import combinations
import pandas as pd
import requests
import streamlit as st
from gl import GL_FIELDS, amount, dates, validate_mapping, normalize_gl, deposit_accounts, extract_ledger, gl_warnings
from card import CARD_FIELDS, CARD_LABELS, normalize_card, card_payable_accounts, extract_card_ledger, reconcile_card, card_summary

OLLAMA="http://localhost:11434"
MODEL_OPTIONS=["qwen2.5:7b","gemma2:9b","llama3.1:8b","gpt-oss:20b"]
BANK=["transaction_date","description","withdrawal","deposit","balance"]
CARD=CARD_FIELDS
LEDGER=GL_FIELDS
DB_PATH="reviewed.db"

# ============ 파일 읽기 / AI 연동 ============

def read_file(f):
    if not f.name.lower().endswith(".csv"):
        return pd.read_excel(f)
    for enc in ["utf-8-sig","cp949","euc-kr"]:
        try:
            f.seek(0)
            return pd.read_csv(f,encoding=enc,dtype=object)
        except (UnicodeDecodeError,UnicodeError):
            continue
    f.seek(0)
    return pd.read_csv(f)

def ollama_ok():
    try: return requests.get(OLLAMA+"/api/tags",timeout=2).ok
    except: return False

def heuristic(df,kind):
    m={}
    for c in df.columns:
        s=str(c).lower().replace(" ","")
        if kind=="bank":
            if any(x in s for x in ["거래일","거래일자","일자","date"]): m["transaction_date"]=c
            elif any(x in s for x in ["적요","거래내용","내용","description","memo"]): m["description"]=c
            elif any(x in s for x in ["출금","지급","찾으신","withdrawal","debit"]): m["withdrawal"]=c
            elif any(x in s for x in ["입금","수입","맡기신","deposit","credit"]): m["deposit"]=c
            elif any(x in s for x in ["잔액","잔고","balance"]): m["balance"]=c
        elif kind=="card":
            if any(x in s for x in ["이용일자","승인일자","거래일자","transaction_date","date"]): m["transaction_date"]=c
            elif any(x in s for x in ["가맹점명","가맹점","merchant","vendor"]): m["merchant"]=c
            elif any(x in s for x in ["이용금액","승인금액","결제금액","amount"]): m["amount"]=c
            elif any(x in s for x in ["거래상태","매입상태","상태","status"]): m["status"]=c
            elif any(x in s for x in ["승인번호","approval_number","approval"]): m["approval_number"]=c
            elif any(x in s for x in ["카드번호","card_number"]): m["card_number"]=c
        else:
            if any(x in s for x in ["전표번호","전표id","journal"]): m["journal_id"]=c
            elif any(x in s for x in ["전기일","전표일","일자","posting","date"]): m["posting_date"]=c
            elif any(x in s for x in ["적요","내용","description","memo"]): m["description"]=c
            elif any(x in s for x in ["거래처","counterparty","vendor"]): m["counterparty"]=c
            elif any(x in s for x in ["차변","debit"]): m["debit"]=c
            elif any(x in s for x in ["대변","credit"]): m["credit"]=c
            elif any(x in s for x in ["계정코드","계정번호","account_code","accountcode"]): m["account_code"]=c
            elif any(x in s for x in ["계정과목","계정명","account_name","accountname","계정"]): m["account_name"]=c
    return {"mapping":m,"confidence":0.5}

def ai_map(df,kind,model):
    targets=BANK if kind=="bank" else CARD if kind=="card" else LEDGER
    prompt=f"""You are an accounting data import assistant.
File type: {kind}
Target fields: {targets}
For GL: debit means 차변, credit means 대변; account_code means 계정코드,
account_name means 계정과목. For bank: withdrawal means 출금, deposit means 입금.
For card: merchant means 가맹점, amount means 이용금액, status means 거래상태.
Map column labels only. Never infer directions from transaction values.
Columns: {list(df.columns)}
Sample: {df.head(6).fillna("").astype(str).to_dict(orient="records")}
Return ONLY JSON: {{"mapping":{{"target_field":"source_column"}},"confidence":0.0}}
Use null when unavailable. Do not invent columns."""
    r=requests.post(OLLAMA+"/api/chat",json={"model":model,"messages":[{"role":"user","content":prompt}],"stream":False,"format":"json","options":{"temperature":0}},timeout=120)
    r.raise_for_status()
    result=json.loads(r.json()["message"]["content"])
    proposed=result.get("mapping",{})
    if not isinstance(proposed,dict):
        raise ValueError("AI 매핑 응답 형식 오류")
    mapping={f:v for f,v in proposed.items() if f in targets and isinstance(v,str) and v in df.columns}
    # Explicit known labels are authoritative; an LLM must never swap 차변/대변.
    known=heuristic(df,kind)["mapping"]
    corrected=[f for f,v in known.items() if mapping.get(f)!=v]
    mapping.update(known)
    return {"mapping":mapping,"confidence":result.get("confidence"),
            "method":"Ollama + known-column validation","rule_corrected_fields":corrected}

def ai_party_review(external_name, gl_name, model):
    """Explain one rule-shortlisted name pair; never certify a transaction."""
    prompt=("두 거래처 이름이 같은 업체 또는 같은 기관의 표기일 가능성이 있는지 평가하세요. "
            "영문 약칭, 한영 음역, 지역 지점명이나 업종 접미사가 붙은 경우를 고려하세요. "
            "예를 들어 KEPCO는 한국전력공사의 영문 약칭이고 FNB는 food and beverage의 약칭입니다. "
            "이는 이름 후보 설명일 뿐이며 방향·금액·날짜는 별도 규칙으로 확인했습니다. "
            "확정 회계판단은 하지 마세요. JSON만 반환하세요: same_entity(boolean), "
            "confidence(0..1), reason(짧은 한국어 문장). "
            f"External: {external_name!r}; GL: {gl_name!r}")
    response=requests.post(OLLAMA+"/api/chat",json={"model":model,"messages":[{"role":"user","content":prompt}],
                          "stream":False,"format":"json","options":{"temperature":0}},timeout=35)
    response.raise_for_status()
    answer=json.loads(response.json()["message"]["content"])
    if not isinstance(answer,dict) or type(answer.get('same_entity')) is not bool:
        raise ValueError('AI 거래처 응답 형식 오류')
    return {'same_entity':answer['same_entity'],'reason':str(answer.get('reason',''))[:240]}

def explain_party_reviews(rows, external, ledger, kind, model, available):
    """Optional AI explanation for single unresolved amount/date candidates."""
    if not available:
        return rows
    source_key='bank_idx' if kind=='bank' else 'card_idx'
    source_name='description' if kind=='bank' else 'merchant'
    for row in rows:
        if row['status']!='거래처 확인 필요' or len(row['ledger_idx'])!=1 or len(row[source_key])!=1:
            continue
        source=str(external.loc[row[source_key][0],source_name])
        target=str(ledger.loc[row['ledger_idx'][0],'counterparty'])
        try:
            answer=ai_party_review(source,target,model)
        except (requests.RequestException,ValueError,KeyError,TypeError,json.JSONDecodeError):
            continue
        row['ai_reason']=answer['reason']
        if answer['same_entity']:
            row['status']='AI 후보 매칭'
            row['hint']='AI 거래처 후보 — 담당자 최종 확인 필요'
    return rows

# ============ 정규화 / 매칭 로직 (기존 검증된 버전) ============

def num(x): return amount(x)

def txt(x):return re.sub(r"[^0-9a-zA-Z가-힣]","",str(x)).lower()

def normalize_bank(df,m):
    validate_mapping(df,m,"bank")
    o=pd.DataFrame(index=df.index)
    for f in BANK:o[f]=df[m.get(f)] if m.get(f) in df.columns else ""
    o.transaction_date=dates(o.transaction_date)
    for f in ["withdrawal","deposit","balance"]:o[f]=o[f].map(num)
    o.description=o.description.fillna("").astype(str)
    if o.transaction_date.isna().any():
        raise ValueError("은행 거래일을 확인하세요.")
    if ((o.withdrawal < 0) | (o.deposit < 0) | ((o.withdrawal > 0) & (o.deposit > 0))).any():
        raise ValueError("은행 금액은 음수 없이 입금 또는 출금 한쪽에 입력하세요.")
    o["signed"]=(o.deposit-o.withdrawal).round(2)
    return o

def normalize_ledger(df,m,account_name=None):
    gl=normalize_gl(df,m)
    accounts=deposit_accounts(gl)
    if account_name is None:
        if len(accounts)!=1:
            raise ValueError("대사할 보통예금 계정을 선택하세요.")
        account_name=next(iter(accounts))
    return extract_ledger(gl,account_name)

DATE_WINDOW=7  # 이 날짜 범위를 벗어나면 이름/금액이 맞아도 후보로 보지 않음

def guess_error_type(a,b):
    """a,b: 두 금액(절대값). 전형적인 오타 패턴이면 힌트를 반환."""
    diff=abs(round(a)-round(b))
    if diff==0:return ""
    hints=[]
    if diff%9==0:
        hints.append("자리바꿈(전위) 오류 의심")  # 예: 1051↔1015, 차액이 항상 9의 배수
    hi,lo=max(a,b),min(a,b)
    if lo>0:
        ratio=hi/lo
        for p in (10,100,1000):
            if abs(ratio-p)<0.5:
                hints.append(f"자릿수(0) 오류 의심 — 약 {p}배 차이")
    return " / ".join(hints)

MAX_GROUP_SIZE=4      # 그룹 매칭 시 최대 몇 건까지 묶어볼지
GROUP_POOL_LIMIT=10    # 후보 풀 크기 제한 (조합 폭발/오탐 방지)

def group_sum_match(target,pool,tol=.01):
    """pool: [(idx, signed값), ...]. 부분집합 합이 target과 tol 이내로 맞는 조합의 idx 리스트를 반환."""
    items=[(i,v) for i,v in pool if v*target>0][:GROUP_POOL_LIMIT]
    for size in range(2,min(MAX_GROUP_SIZE,len(items))+1):
        for combo in combinations(items,size):
            if abs(sum(v for _,v in combo)-target)<tol:
                return [i for i,_ in combo]
    return None

def name_match(x,y):
    from party import party_relation
    return any(party_relation(x.description, party) != 'none'
               for party in str(y.counterparty).split(', '))

def party_method(external, counterparties):
    from party import party_relation
    relations=[party_relation(external, party) for party in str(counterparties).split(', ')]
    return next((method for method in ('exact','normalized','alias') if method in relations),'none')

def party_hint(method):
    return {'normalized':'거래처 표기 정규화','alias':'거래처 별칭'}.get(method,'')

def date_days(x,y):
    return abs((x.transaction_date-y.posting_date).days) if pd.notna(x.transaction_date) and pd.notna(y.posting_date) else None

def assign_global(candidates,used_b,used_l):
    """candidates: [(score,bi,li,payload),...]. 점수 높은 순으로 전역 정렬 후 배정.
    (은행 행을 순서대로 처리하며 그 자리에서 바로 채가면, 먼저 처리된 행이 더 적합한 후보를
     선점해버려 뒤에 처리되는 행이 진짜 짝을 놓칠 수 있음 — 그래서 전체를 다 모아 점수순 정렬 후 배정)"""
    assigned=[]
    for score,bi,li,payload in sorted(candidates,key=lambda c:-c[0]):
        if bi in used_b or li in used_l:continue
        used_b.add(bi);used_l.add(li)
        assigned.append((bi,li,payload))
    return assigned

def reconcile(b,l):
    used_b=set(); used_l=set(); rows=[]

    # ---- 1단계: 적요(거래처)도 맞고 금액도 정확히 같은 것 — 가장 확실한 매칭 ----
    # 1:1 매칭은 반드시 적요가 맞아야만 인정함. 금액이 우연히 같다는 것만으로는(예: AWS 10만원과
    # 전혀 무관한 Apple 10만원) 절대 같은 거래로 단정하지 않음 — 이건 3단계에서도 마찬가지.
    cands=[]
    for bi,x in b.iterrows():
        for li,y in l.iterrows():
            if not name_match(x,y):continue
            if abs(x.signed-y.signed)>=.01:continue
            days=date_days(x,y)
            if days is not None and days>DATE_WINDOW:continue
            method=party_method(x.description,y.counterparty)
            score=.7+(.3 if (days is None or days<=1) else .15)+{'exact':.03,'normalized':.02,'alias':.01}.get(method,0)
            cands.append((score,bi,li,days))
    for bi,li,days in assign_global(cands,used_b,used_l):
        status="MATCHED" if (days is None or days<=1) else "MATCHED (날짜 차이)"
        rows.append({"status":status,"bank_idx":[bi],"ledger_idx":[li],"days":days,"diff":0.0,
                     "hint":party_hint(party_method(b.loc[bi,'description'],l.loc[li,'counterparty']))})

    # Direction, amount and date first; an unknown party is a review candidate,
    # never an automatically confirmed match. Preserve ambiguous alternatives.
    for bi in [i for i in b.index if i not in used_b]:
        x=b.loc[bi]
        options=[]
        for li in [i for i in l.index if i not in used_l]:
            y=l.loc[li]
            days=date_days(x,y)
            if x.signed*y.signed<=0 or abs(x.signed-y.signed)>=.01 or (days is not None and days>DATE_WINDOW):continue
            options.append((li,days))
        if len(options)==1:
            li,days=options[0]
            used_b.add(bi);used_l.add(li)
            rows.append({"status":"거래처 확인 필요","bank_idx":[bi],"ledger_idx":[li],"days":days,"diff":0.0,
                         "hint":"금액·날짜 일치, 거래처명 확인 필요"})
        elif len(options)>1:
            used_b.add(bi)
            labels=', '.join(str(l.loc[li,'counterparty']) for li,_ in options[:5])
            rows.append({"status":"거래처 확인 필요","bank_idx":[bi],"ledger_idx":[],"days":None,"diff":None,
                         "hint":"동일 금액·날짜 GL 후보: "+labels})

    # ---- 2단계: 그룹(분할·합산) 매칭 — 금액 합계만으로 판단 ----
    for bi in [i for i in b.index if i not in used_b]:
        x=b.loc[bi]
        pool=[(li,l.loc[li].signed) for li in l.index if li not in used_l
              and (pd.isna(x.transaction_date) or pd.isna(l.loc[li].posting_date)
                   or abs((x.transaction_date-l.loc[li].posting_date).days)<=DATE_WINDOW)]
        combo=group_sum_match(x.signed,pool)
        if combo:
            used_b.add(bi)
            for li in combo:used_l.add(li)
            parts=[f"{(l.loc[li].description or l.loc[li].counterparty)}:{abs(l.loc[li].signed):,.0f}" for li in combo]
            rows.append({"status":"1:N 매칭 (분할 지급)","bank_idx":[bi],"ledger_idx":combo,
                         "days":None,"diff":0.0,"hint":"원장 "+str(len(combo))+"건 합산 = "+" + ".join(parts)})

    for li in [i for i in l.index if i not in used_l]:
        y=l.loc[li]
        pool=[(bi,b.loc[bi].signed) for bi in b.index if bi not in used_b
              and (pd.isna(y.posting_date) or pd.isna(b.loc[bi].transaction_date)
                   or abs((b.loc[bi].transaction_date-y.posting_date).days)<=DATE_WINDOW)]
        combo=group_sum_match(y.signed,pool)
        if combo:
            used_l.add(li)
            for bi in combo:used_b.add(bi)
            parts=[f"{b.loc[bi].description}:{abs(b.loc[bi].signed):,.0f}" for bi in combo]
            rows.append({"status":"N:1 매칭 (합산 지급)","bank_idx":combo,"ledger_idx":[li],
                         "days":None,"diff":0.0,"hint":"은행 "+str(len(combo))+"건 합산 = "+" + ".join(parts)})

    # ---- 3단계: 적요(거래처)는 맞는데 금액이 다른 경우 — 오타 또는 입금/출금 방향 오류 ----
    cands=[]
    for bi in [i for i in b.index if i not in used_b]:
        x=b.loc[bi]
        for li in [i for i in l.index if i not in used_l]:
            y=l.loc[li]
            if not name_match(x,y):continue
            days=date_days(x,y)
            if days is not None and days>DATE_WINDOW:continue
            score=.3+(.1 if (days is None or days<=1) else .05)
            cands.append((score,bi,li,days))
    for bi,li,days in assign_global(cands,used_b,used_l):
        x=b.loc[bi];y=l.loc[li]
        diff=abs(x.signed)-abs(y.signed)
        if abs(diff)<.01:
            status="입금/출금 방향 오류 의심"
            hint="금액은 일치하나 입금·출금 방향이 반대로 기표된 것으로 보임"
        else:
            status="AMOUNT MISMATCH"
            hint=guess_error_type(abs(x.signed),abs(y.signed))
        rows.append({"status":status,"bank_idx":[bi],"ledger_idx":[li],"days":days,"diff":diff,"hint":hint})

    # ---- 4단계: 그 무엇으로도 안 엮인 건 억지로 짝짓지 않고 각자 예외로 남김 ----
    for bi in b.index:
        if bi not in used_b:rows.append({"status":"BANK ONLY","bank_idx":[bi],"ledger_idx":[],"days":None,"diff":None,"hint":""})
    for li in l.index:
        if li not in used_l:rows.append({"status":"LEDGER ONLY","bank_idx":[],"ledger_idx":[li],"days":None,"diff":None,"hint":""})
    return rows

def reconciliation_amounts(row,bank,ledger):
    """Absolute amounts per matched side; missing side stays blank, not zero."""
    return {"bank_amount": round(sum(abs(bank.loc[i,"signed"]) for i in row["bank_idx"]),2) if row["bank_idx"] else None,
            "ledger_amount": round(sum(abs(ledger.loc[i,"signed"]) for i in row["ledger_idx"]),2) if row["ledger_idx"] else None}

MATCHED_STATUS={"MATCHED","MATCHED (날짜 차이)","1:N 매칭 (분할 지급)","N:1 매칭 (합산 지급)"}

# ============ SQLite "확인완료" 저장 ============
# 왜 필요한가: 롤링 30일 창을 매일 다시 계산하기 때문에, 어제 사람이 이미 "정상이다"라고
# 확인한 예외(예: 환율차이로 인한 금액차이)도 그 거래가 30일 창 안에 있는 동안은 매일 다시
# 예외로 뜬다. 그래서 "이 건은 확인했다"는 사람의 판단을 별도로 기억해둬야 한다.

def init_db():
    conn=sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS reviewed(
        key TEXT PRIMARY KEY, account TEXT, status TEXT, note TEXT, reviewed_at TEXT)""")
    conn.commit()
    return conn

def exception_key(account,status,bank_parts,ledger_parts):
    parts=sorted(bank_parts+ledger_parts)
    raw=account+"|"+status+"|"+"|".join(parts)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]

def row_key(account,row,b,l):
    bparts=[f"B|{b.loc[i,'transaction_date']}|{b.loc[i,'signed']}|{b.loc[i,'description']}" for i in row["bank_idx"]]
    lparts=[f"L|{l.loc[i,'posting_date']}|{l.loc[i,'signed']}|{l.loc[i,'description']}" for i in row["ledger_idx"]]
    return exception_key(account,row["status"],bparts,lparts)

def get_reviewed_map(conn):
    return {r[0]:r[1] for r in conn.execute("SELECT key,note FROM reviewed")}

def mark_reviewed(conn,key,account,status,note):
    conn.execute("INSERT OR REPLACE INTO reviewed(key,account,status,note,reviewed_at) VALUES (?,?,?,?,datetime('now'))",
                 (key,account,status,note))
    conn.commit()

def unmark_reviewed(conn,key):
    conn.execute("DELETE FROM reviewed WHERE key=?",(key,))
    conn.commit()

# ================= UI =================

def main():
    st.set_page_config(page_title="은행·법인카드 회계 대사",layout="wide")
    st.markdown("<style>h1{font-size:2.5rem!important;line-height:1.25!important}</style>",unsafe_allow_html=True)
    st.title("로컬 AI 기반 은행·법인카드 회계 대사 시스템")
    st.caption("은행·카드와 전체 GL을 각각 대사합니다. 금액·날짜·방향은 규칙으로 검증하고, AI는 컬럼 의미와 미해결 거래처 후보만 설명합니다. 최종 확인은 담당자가 합니다.")
    
    conn=init_db()
    
    with st.sidebar:
        st.subheader("Local AI")
        st.write("Ollama:", "🟢 연결됨" if ollama_ok() else "🔴 미연결")
        MODEL=st.selectbox("Model",MODEL_OPTIONS,index=0)
        st.caption("노트북 VRAM이 8GB 정도면 qwen2.5:7b / gemma2:9b / llama3.1:8b가 안전합니다. gpt-oss:20b는 16GB+ 권장.")
        st.info("Ollama가 없으면 규칙 기반 컬럼 매핑·거래처 정규화·별칭으로 동작하고, 남은 이름 차이는 확인 필요로 표시합니다.")
    st.subheader("① 신한은행 주거래계좌 거래내역")
    bf=st.file_uploader("신한은행 거래내역",type=["xlsx","csv"],key="bank_main")
    st.subheader("② 신한 법인카드 이용내역")
    cf=st.file_uploader("신한카드 이용내역",type=["xlsx","csv"],key="card_main")
    st.subheader("③ 전체 GL(총계정원장) 업로드")
    st.caption("전표번호 / 전기일 / 계정코드 / 계정과목 / 차변 / 대변 / 거래처 / 적요 형식의 전체 GL을 업로드하세요.")
    lf=st.file_uploader("원장 파일",type=["xlsx","csv"],key="ledger")

    ready=all(f is not None for f in (bf,cf,lf))
    signature=hashlib.sha256(repr([(key,f.getvalue() if f else None) for key,f in
                                   (("bank",bf),("card",cf),("GL",lf))]).encode()).hexdigest()
    if st.session_state.get("input_signature") != signature:
        for key in ["bm","cm","lm","per_account","card_results","reconcile_config"]:
            st.session_state.pop(key,None)
        st.session_state.input_signature=signature

    if ready and st.button("④ AI 컬럼 매핑",type="primary"):
        try:
            lraw=read_file(lf)
            braw=read_file(bf)
            craw=read_file(cf)
        except Exception as e:
            st.error(f"파일을 읽을 수 없습니다: {e}")
            st.stop()
        if ollama_ok():
            try:
                bm=ai_map(braw,"bank",MODEL)
                cm=ai_map(craw,"card",MODEL)
                lm=ai_map(lraw,"ledger",MODEL)
            except Exception:
                st.warning("AI 분석 실패 → 규칙 기반 fallback")
                bm=heuristic(braw,"bank")
                cm=heuristic(craw,"card")
                lm=heuristic(lraw,"ledger")
        else:
            bm=heuristic(braw,"bank")
            cm=heuristic(craw,"card")
            lm=heuristic(lraw,"ledger")
        st.session_state.bm=bm
        st.session_state.cm=cm
        st.session_state.lm=lm
        st.session_state.braw=braw
        st.session_state.craw=craw
        st.session_state.lraw=lraw
        st.success("파일 분석 완료")
    
    if "bm" in st.session_state:
        st.subheader("④ 컬럼 매핑 확인·수정")
        with st.expander("AI / 규칙 분석 결과"):
            st.json({"GL":st.session_state.lm,"은행":st.session_state.bm,"법인카드":st.session_state.cm})
        def edit_mapping(raw, initial, kind, key):
            fields=BANK if kind=="bank" else CARD if kind=="card" else LEDGER
            options=[None]+list(raw.columns)
            result={}
            with st.expander(key+" 컬럼 매핑",expanded=False):
                for f in fields:
                    current=initial.get("mapping",{}).get(f)
                    result[f]=st.selectbox(f,options,index=options.index(current) if current in options else 0,
                        format_func=lambda v: v if v is not None else "(없음)",key=f"map_{signature}_{key}_{f}")
            return result
        lm=edit_mapping(st.session_state.lraw,st.session_state.lm,"ledger","GL")
        bm=edit_mapping(st.session_state.braw,st.session_state.bm,"bank","신한은행")
        cm=edit_mapping(st.session_state.craw,st.session_state.cm,"card","신한카드")
        try:
            gl=normalize_gl(st.session_state.lraw,lm)
            accounts=deposit_accounts(gl)
            card_accounts=card_payable_accounts(gl)
            for warning in gl_warnings(gl): st.warning(warning)
            st.subheader("⑤ GL 대사 계정")
            st.write({"은행 보통예금":list(accounts.values()),"법인카드 미지급금":list(card_accounts.values())})
            if len(accounts)!=1 or len(card_accounts)!=1:
                st.warning("이 데모에는 보통예금과 카드 미지급금 계정이 각각 하나여야 합니다. GL 매핑과 계정과목을 확인하세요.")
                st.stop()
            bank_key=next(iter(accounts)); card_key=next(iter(card_accounts))
            config=repr((lm,bm,cm,bank_key,card_key))
            if st.session_state.get("reconcile_config") != config:
                st.session_state.pop("per_account",None)
                st.session_state.pop("card_results",None)
            if st.button("⑥ 은행·법인카드 대사 실행"):
                b=normalize_bank(st.session_state.braw,bm)
                l=extract_ledger(gl,bank_key)
                cards=normalize_card(st.session_state.craw,cm)
                card_ledger=extract_card_ledger(gl,card_key)
                ai_available=ollama_ok()
                bank_rows=explain_party_reviews(reconcile(b,l),b,l,'bank',MODEL,ai_available)
                card_rows=explain_party_reviews(reconcile_card(cards,card_ledger),cards,card_ledger,'card',MODEL,ai_available)
                st.session_state.per_account={accounts[bank_key]:(b,l,bank_rows)}
                st.session_state.card_results=(cards,card_ledger,card_rows)
                st.session_state.reconcile_config=config
                st.success("대사 완료")
            with st.expander("GL 전표 연결 데이터"):
                st.dataframe(extract_ledger(gl,bank_key),width="stretch")
                st.dataframe(extract_card_ledger(gl,card_key),width="stretch")
        except ValueError as e:
            st.session_state.pop("per_account",None)
            st.error(str(e))
    
    if "per_account" in st.session_state:
        per_account=st.session_state.per_account
        reviewed_map=get_reviewed_map(conn)  # 매번 최신 상태로 재조회
    
        tab1,tab2=st.tabs(["🔍 은행 대사","💳 법인카드 대사"])
    
        with tab1:
            for name,(b,l,rows) in per_account.items():
                st.markdown(f"### {name}")
                out=[];keys=[]
                for row in rows:
                    bdesc=", ".join(b.loc[i,"description"] for i in row["bank_idx"]) if row["bank_idx"] else ""
                    ldesc=", ".join(l.loc[i,"description"] for i in row["ledger_idx"]) if row["ledger_idx"] else ""
                    jvno=", ".join(dict.fromkeys(str(l.loc[i,"journal_id"]) for i in row["ledger_idx"] if l.loc[i,"journal_id"])) if row["ledger_idx"] else ""
                    key=row_key(name,row,b,l)
                    amounts=reconciliation_amounts(row,b,l)
                    out.append({"status":row["status"],"bank":bdesc,"ledger":ldesc,"전표번호":jvno,"상대계정":", ".join(dict.fromkeys(l.loc[i,"account"] for i in row["ledger_idx"])),"거래처":", ".join(dict.fromkeys(l.loc[i,"counterparty"] for i in row["ledger_idx"])),"date_diff_days":row["days"],
                                "은행금액":amounts["bank_amount"],"원장금액":amounts["ledger_amount"],
                                "amount_diff":row["diff"],"의심 유형":row["hint"],"AI 설명":row.get("ai_reason",""),
                                "확인상태":"✅ 확인완료" if key in reviewed_map else ("-" if row["status"] in MATCHED_STATUS else "미확인")})
                    keys.append(key)
                res=pd.DataFrame(out,columns=["status","bank","ledger","전표번호","상대계정","거래처","은행금액","원장금액","date_diff_days","amount_diff","의심 유형","AI 설명","확인상태"])
                st.caption("핵심 7개 컬럼을 먼저 표시합니다. 날짜 차이·원장금액·확인사항은 아래 상세 내역에서 확인할 수 있습니다.")
                a,c,d=st.columns(3)
                a.metric("전체",len(res));c.metric("매칭",int(res.status.isin(MATCHED_STATUS).sum()));d.metric("예외",int((~res.status.isin(MATCHED_STATUS)).sum()))
                primary=res[["status","bank","ledger","전표번호","상대계정","거래처","은행금액"]].rename(
                    columns={"status":"상태","bank":"은행 적요","ledger":"원장 적요"})
                st.dataframe(primary,width="stretch",height=600,hide_index=True)
                with st.expander("날짜 차이·원장금액·확인사항 상세 보기"):
                    st.dataframe(res,width="stretch",hide_index=True)
    
                exc_rows=[(i,row) for i,row in enumerate(rows) if row["status"] not in MATCHED_STATUS]
                unresolved=[(i,row) for i,row in exc_rows if keys[i] not in reviewed_map]
                if unresolved:
                    with st.expander(f"미확인 예외 {len(unresolved)}건 처리"):
                        for i,row in unresolved:
                            bdesc=", ".join(b.loc[j,"description"] for j in row["bank_idx"]) if row["bank_idx"] else "-"
                            ldesc=", ".join(l.loc[j,"description"] for j in row["ledger_idx"]) if row["ledger_idx"] else "-"
                            jvno=", ".join(dict.fromkeys(str(l.loc[j,"journal_id"]) for j in row["ledger_idx"] if l.loc[j,"journal_id"])) if row["ledger_idx"] else ""
                            st.write(f"**{row['status']}** — 은행: {bdesc} / 원장: {ldesc}" + (f" (전표번호: {jvno})" if jvno else ""))
                            note=st.text_input("확인 사유",key=f"note_{name}_{i}")
                            if st.button("확인완료로 표시",key=f"btn_{name}_{i}"):
                                mark_reviewed(conn,keys[i],name,row["status"],note or "사유 미기재")
                                st.rerun()
    
                st.divider()
    
        with tab2:
            cards,card_ledger,card_rows=st.session_state.card_results
            summary=card_summary(cards,card_rows)
            labels=[("전체 카드거래","total"),("정상 매칭","matched"),("미확정","pending"),
                    ("취소","cancelled"),("확인 필요","review")]
            for col,(label,key) in zip(st.columns(5),labels):col.metric(label,summary[key])
            st.caption("미확정·취소는 GL 매칭 및 예외 건수에서 제외합니다. 이 분류는 데모 업무 가정입니다.")
            card_detail=[]
            for row in card_rows:
                c=cards.loc[row['card_idx'][0]] if row['card_idx'] else None
                l=card_ledger.loc[row['ledger_idx'][0]] if row['ledger_idx'] else None
                status={'PENDING':'매입대기 — 전표처리 대기','CANCELLED':'승인취소'}.get(row['status'],row['status'])
                card_detail.append({'상태':status,'카드 이용일':c.transaction_date.date() if c is not None else None,
                    '카드 상태':CARD_LABELS[c.status] if c is not None else '',
                    '가맹점':c.merchant if c is not None else '',
                    '카드 이용금액':c.amount if c is not None else None,
                    'GL 전표번호':l.journal_id if l is not None else '',
                    '비용계정':l.expense_accounts if l is not None else '',
                    '부가세대급금':l.vat_amount if l is not None else None,
                    '거래처':l.counterparty if l is not None else '',
                    'GL 카드미지급금':l.payable_amount if l is not None else None,
                    '날짜차이':row['days'],'확인사항':row['hint'] or '-',
                    'AI 설명':row.get('ai_reason','')})
            detail_df=pd.DataFrame(card_detail)
            primary=detail_df[["상태","카드 이용일","가맹점","카드 이용금액","GL 전표번호","비용계정","거래처","GL 카드미지급금"]]
            st.caption("핵심 8개 컬럼을 먼저 표시합니다. 부가세대급금·날짜 차이·확인사항은 아래 상세 내역에서 확인할 수 있습니다.")
            st.dataframe(primary,width="stretch",height=520,hide_index=True)
            with st.expander("부가세대급금·날짜 차이·확인사항 상세 보기"):
                st.dataframe(detail_df,width="stretch",hide_index=True)

if __name__ == "__main__":
    main()
