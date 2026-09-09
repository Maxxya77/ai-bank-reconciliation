import json,re,hashlib,sqlite3,os
from itertools import combinations
from datetime import datetime
import pandas as pd
import requests
import streamlit as st

OLLAMA="http://localhost:11434"
MODEL_OPTIONS=["qwen2.5:7b","gemma2:9b","llama3.1:8b","gpt-oss:20b"]
BANK=["transaction_date","description","withdrawal","deposit","balance"]
LEDGER=["journal_id","posting_date","description","counterparty","cash_in","cash_out","account","account_name"]
DB_PATH="reviewed.db"

# ============ 파일 읽기 / AI 연동 ============

def read_file(f):
    if not f.name.lower().endswith(".csv"):
        return pd.read_excel(f)
    for enc in ["utf-8-sig","cp949","euc-kr"]:
        try:
            f.seek(0)
            return pd.read_csv(f,encoding=enc)
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
        else:
            if any(x in s for x in ["전표번호","전표id","journal"]): m["journal_id"]=c
            elif any(x in s for x in ["전기일","전표일","일자","posting","date"]): m["posting_date"]=c
            elif any(x in s for x in ["적요","내용","description","memo"]): m["description"]=c
            elif any(x in s for x in ["거래처","counterparty","vendor"]): m["counterparty"]=c
            elif any(x in s for x in ["입금","수입","차변","cash_in"]): m["cash_in"]=c
            elif any(x in s for x in ["출금","지급","대변","cash_out"]): m["cash_out"]=c
            elif any(x in s for x in ["계좌","은행계좌","예금계좌"]): m["account_name"]=c
            elif any(x in s for x in ["계정","account"]): m["account"]=c
    return {"mapping":m,"confidence":0.5}

def ai_map(df,kind,model):
    targets=BANK if kind=="bank" else LEDGER
    prompt=f"""You are an accounting data import assistant.
File type: {kind}
Target fields: {targets}
Columns: {list(df.columns)}
Sample: {df.head(6).fillna("").astype(str).to_dict(orient="records")}
Return ONLY JSON: {{"mapping":{{"target_field":"source_column"}},"confidence":0.0}}
Use null when unavailable. Do not invent columns."""
    r=requests.post(OLLAMA+"/api/chat",json={"model":model,"messages":[{"role":"user","content":prompt}],"stream":False,"format":"json"},timeout=120)
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])

def ai_explain(rows,model):
    """미매칭/불일치 건에 대해 한 번의 호출로 원인 후보를 물어봄"""
    items=[{"idx":i,"status":r["status"],"bank":r["bank"],"ledger":r["ledger"],
            "amount_diff":r["amount_diff"],"date_diff_days":r["date_diff_days"]} for i,r in enumerate(rows)]
    prompt=f"""You are an accounting reviewer. For each item, give ONE short Korean sentence
guessing the most likely cause (e.g. 환불, 수수료 처리 기준 차이, 이연, 입력 누락, 이중 등록).
Items: {json.dumps(items,ensure_ascii=False)}
Return ONLY JSON: {{"reasons":{{"idx":"이유"}}}}"""
    r=requests.post(OLLAMA+"/api/chat",json={"model":model,"messages":[{"role":"user","content":prompt}],"stream":False,"format":"json"},timeout=180)
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"]).get("reasons",{})

# ============ 정규화 / 매칭 로직 (기존 검증된 버전) ============

def num(x):
    try:return float(str(x).replace(",","").replace("원","").strip())
    except:return 0.0
def txt(x):return re.sub(r"[^0-9a-zA-Z가-힣]","",str(x)).lower()

def normalize_bank(df,m):
    o=pd.DataFrame()
    for f in BANK:o[f]=df[m.get(f)] if m.get(f) in df.columns else ""
    o.transaction_date=pd.to_datetime(o.transaction_date,errors="coerce")
    for f in ["withdrawal","deposit","balance"]:o[f]=o[f].map(num)
    o.description=o.description.fillna("").astype(str)
    o["signed"]=o.deposit-o.withdrawal
    return o

def normalize_ledger(df,m,account_name=None):
    """account_name이 주어지면 원장에서 해당 계좌 행만 슬라이스한 뒤 표준화"""
    if account_name is not None and m.get("account_name") in df.columns:
        df=df[df[m["account_name"]]==account_name]
    o=pd.DataFrame()
    for f in LEDGER:o[f]=df[m.get(f)] if m.get(f) in df.columns else ""
    o.posting_date=pd.to_datetime(o.posting_date,errors="coerce")
    for f in ["cash_in","cash_out"]:o[f]=o[f].map(num)
    for f in ["description","counterparty"]:o[f]=o[f].fillna("").astype(str)
    o["signed"]=o.cash_in-o.cash_out
    return o

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
    items=pool[:GROUP_POOL_LIMIT]
    for size in range(2,min(MAX_GROUP_SIZE,len(items))+1):
        for combo in combinations(items,size):
            if abs(sum(v for _,v in combo)-target)<tol:
                return [i for i,_ in combo]
    return None

def name_match(x,y):
    return (txt(x.description) in txt(y.description) or txt(y.description) in txt(x.description)
            or (y.counterparty and (txt(x.description) in txt(y.counterparty) or txt(y.counterparty) in txt(x.description))))

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
            score=.7+(.3 if (days is None or days<=1) else .15)
            cands.append((score,bi,li,days))
    for bi,li,days in assign_global(cands,used_b,used_l):
        status="MATCHED" if (days is None or days<=1) else "MATCHED (날짜 차이)"
        rows.append({"status":status,"bank_idx":[bi],"ledger_idx":[li],"days":days,"diff":0.0,"hint":""})

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

MATCHED_STATUS={"MATCHED","MATCHED (날짜 차이)","1:N 매칭 (분할 지급)","N:1 매칭 (합산 지급)"}

# ============ 일일 자금 스냅샷 ============

def daily_snapshot(bank_df,account_name):
    if bank_df.empty or bank_df.transaction_date.isna().all():
        return {"계좌":account_name,"기준일":None,"전일잔액":None,"금일입금":0,"금일출금":0,"금일잔액":None}
    report_date=bank_df.transaction_date.max()
    today_rows=bank_df[bank_df.transaction_date==report_date]
    dep=today_rows.deposit.sum();wd=today_rows.withdrawal.sum()
    bal=today_rows.iloc[-1]["balance"] if len(today_rows) else None
    prev=(bal-dep+wd) if bal is not None else None
    return {"계좌":account_name,"기준일":report_date.date(),"전일잔액":prev,"금일입금":dep,"금일출금":wd,"금일잔액":bal}

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

# ============ 엑셀 리포트 (요약 + 상세 2시트) ============

def build_excel_report(report_date,snapshots,detail_df,reviewed_map,exception_keys_by_row,out_path):
    from openpyxl import Workbook
    from openpyxl.styles import Font,PatternFill,Alignment,Border,Side
    from openpyxl.utils import get_column_letter

    FONT="Arial"
    HEADER_FILL=PatternFill("solid",fgColor="1F4B43")
    HEADER_FONT=Font(name=FONT,bold=True,color="FFFFFF",size=11)
    TITLE_FONT=Font(name=FONT,bold=True,size=14)
    BODY_FONT=Font(name=FONT,size=10.5)
    BOLD_FONT=Font(name=FONT,bold=True,size=10.5)
    NUM_FMT="#,##0;(#,##0);-"
    THIN=Side(style="thin",color="D0D0D0")
    BORDER=Border(left=THIN,right=THIN,top=THIN,bottom=THIN)

    def style_header(ws,row,ncols,start_col=1):
        for c in range(start_col,start_col+ncols):
            cell=ws.cell(row=row,column=c)
            cell.font=HEADER_FONT;cell.fill=HEADER_FILL
            cell.alignment=Alignment(horizontal="center",vertical="center")
            cell.border=BORDER

    def autosize(ws,widths):
        for i,w in enumerate(widths,start=1):
            ws.column_dimensions[get_column_letter(i)].width=w

    wb=Workbook()
    ws=wb.active;ws.title="일일자금보고"
    ws["B2"]=f"일일 자금 보고 — 기준일 {report_date}"
    ws["B2"].font=TITLE_FONT
    ws.merge_cells("B2:F2")

    headers=["계좌","전일잔액","금일입금","금일출금","금일잔액"]
    hr=4
    for j,h in enumerate(headers):ws.cell(row=hr,column=2+j,value=h)
    style_header(ws,hr,len(headers),start_col=2)

    r=hr+1;first_data_row=r
    for s in snapshots:
        ws.cell(row=r,column=2,value=s["계좌"]).font=BODY_FONT
        ws.cell(row=r,column=3,value=s["전일잔액"]).number_format=NUM_FMT
        ws.cell(row=r,column=4,value=s["금일입금"]).number_format=NUM_FMT
        ws.cell(row=r,column=5,value=s["금일출금"]).number_format=NUM_FMT
        ws.cell(row=r,column=6,value=s["금일잔액"]).number_format=NUM_FMT
        for c in range(2,7):
            ws.cell(row=r,column=c).border=BORDER
            ws.cell(row=r,column=c).font=BODY_FONT
        r+=1
    last_data_row=r-1

    ws.cell(row=r,column=2,value="합계").font=BOLD_FONT
    ws.cell(row=r,column=2).border=BORDER
    for c in range(3,7):
        cl=get_column_letter(c)
        cell=ws.cell(row=r,column=c,value=f"=SUM({cl}{first_data_row}:{cl}{last_data_row})")
        cell.number_format=NUM_FMT;cell.font=BOLD_FONT;cell.border=BORDER
    r+=2

    records=detail_df.to_dict("records")
    matched_cnt=sum(1 for v in records if v["status"] in MATCHED_STATUS)
    total_exceptions=sum(1 for v,k in zip(records,exception_keys_by_row) if v["status"] not in MATCHED_STATUS and k not in reviewed_map)
    total_exception_amt=sum(abs(v["amount_diff"]) for v,k in zip(records,exception_keys_by_row)
                             if v["status"] not in MATCHED_STATUS and k not in reviewed_map and v.get("amount_diff") is not None)

    ws.cell(row=r,column=2,value="대사 현황").font=BOLD_FONT;r+=1
    ws.cell(row=r,column=2,value="확정(정상 매칭)").font=BODY_FONT
    ws.cell(row=r,column=3,value=f"{matched_cnt}건").font=BODY_FONT;r+=1
    ws.cell(row=r,column=2,value="미결(확인 필요)").font=BODY_FONT
    ws.cell(row=r,column=3,value=f"{total_exceptions}건").font=BODY_FONT
    ws.cell(row=r,column=4,value=total_exception_amt).number_format=NUM_FMT
    ws.cell(row=r,column=4).font=BODY_FONT;r+=2
    ws.cell(row=r,column=2,value="※ 확정/미결 건수는 대사 로직(파이썬) 결과 기준이며, '확인완료' 처리된 건은 미결에서 제외됩니다.").font=Font(name=FONT,italic=True,size=9,color="666666")

    autosize(ws,[2,16,14,14,14,14])
    ws.freeze_panes=f"B{hr+1}"

    ws2=wb.create_sheet("상세대사내역")
    cols=["계좌","상태","은행 적요","원장 적요","전표번호","날짜차이(일)","금액차이","의심 유형","확인상태","확인 메모"]
    for j,h in enumerate(cols,start=1):ws2.cell(row=1,column=j,value=h)
    style_header(ws2,1,len(cols))
    for i,(row,key) in enumerate(zip(records,exception_keys_by_row),start=2):
        reviewed=key in reviewed_map
        vals=[row["계좌"],row["status"],row["bank"],row["ledger"],row.get("전표번호",""),row["date_diff_days"],
              row["amount_diff"],row["의심유형"],"확인완료" if reviewed else ("-" if row["status"] in MATCHED_STATUS else "미확인"),
              reviewed_map.get(key,"")]
        for j,v in enumerate(vals,start=1):
            cell=ws2.cell(row=i,column=j,value=v);cell.font=BODY_FONT;cell.border=BORDER
            if j==7 and isinstance(v,(int,float)):cell.number_format=NUM_FMT
    autosize(ws2,[16,20,16,16,14,12,12,30,10,24])
    ws2.freeze_panes="A2"
    wb.save(out_path)

# ================= UI =================

st.set_page_config(page_title="AI Bank Reconciliation",layout="wide")
st.title("AI Bank ↔ Accounting Reconciliation")
st.caption("Local AI (Ollama) + deterministic Python reconciliation — OpenAI API not required.")

conn=init_db()

with st.sidebar:
    st.subheader("Local AI")
    st.write("Ollama:", "🟢 연결됨" if ollama_ok() else "🔴 미연결")
    MODEL=st.selectbox("Model",MODEL_OPTIONS,index=0)
    st.caption("노트북 VRAM이 8GB 정도면 qwen2.5:7b / gemma2:9b / llama3.1:8b가 안전합니다. gpt-oss:20b는 16GB+ 권장.")
    st.info("Ollama가 없으면 규칙 기반 컬럼 매핑으로 fallback합니다.")
    st.divider()
    n_accounts=st.number_input("계좌 개수",min_value=1,max_value=3,value=2)

st.subheader("① 계좌별 은행 거래내역 업로드")
st.caption("각 계좌마다 최근 30일치 거래내역을 업로드하세요 (은행 인터넷뱅킹에서 '최근 30일' 조회 후 다운로드).")
bank_files=[]
cols=st.columns(n_accounts)
for i in range(n_accounts):
    with cols[i]:
        name=st.text_input(f"계좌 {i+1} 이름",value=["주거래 보통예금","법인카드 결제계좌","세금/급여용 계좌"][i],key=f"acc_name_{i}")
        f=st.file_uploader(f"{name} 거래내역",type=["xlsx","xls","csv"],key=f"bank_{i}")
        bank_files.append((name,f))

st.subheader("② 회계/ERP 현금·예금 원장 업로드")
st.caption("계좌 구분 컬럼(예: '계좌')이 포함된 통합 원장 파일 하나를 업로드하세요.")
lf=st.file_uploader("원장 파일",type=["xlsx","xls","csv"],key="ledger")

ready=lf is not None and all(f is not None for _,f in bank_files)

if ready and st.button("AI로 파일 분석 후 대사 시작",type="primary"):
    lraw=read_file(lf)
    braws={name:read_file(f) for name,f in bank_files}
    if ollama_ok():
        try:
            bms={name:ai_map(braw,"bank",MODEL) for name,braw in braws.items()}
            lm=ai_map(lraw,"ledger",MODEL)
        except Exception:
            st.warning("AI 분석 실패 → 규칙 기반 fallback")
            bms={name:heuristic(braw,"bank") for name,braw in braws.items()}
            lm=heuristic(lraw,"ledger")
    else:
        bms={name:heuristic(braw,"bank") for name,braw in braws.items()}
        lm=heuristic(lraw,"ledger")
    st.session_state.bms=bms
    st.session_state.lm=lm
    st.session_state.braws=braws
    st.session_state.lraw=lraw
    st.session_state.account_names=[n for n,_ in bank_files]
    st.success("파일 분석 완료")

if "bms" in st.session_state:
    st.subheader("컬럼 매핑")
    st.json({"원장":st.session_state.lm,"계좌별 은행":st.session_state.bms})

    if st.button("대사 실행"):
        per_account={}
        for name in st.session_state.account_names:
            b=normalize_bank(st.session_state.braws[name],st.session_state.bms[name]["mapping"])
            l=normalize_ledger(st.session_state.lraw,st.session_state.lm["mapping"],account_name=name)
            per_account[name]=(b,l,reconcile(b,l))
        st.session_state.per_account=per_account
        st.session_state.reviewed_map=get_reviewed_map(conn)
        st.success("대사 완료")

if "per_account" in st.session_state:
    per_account=st.session_state.per_account
    reviewed_map=get_reviewed_map(conn)  # 매번 최신 상태로 재조회

    tab1,tab2=st.tabs(["🔍 대사 결과","📊 일일 자금 보고"])

    with tab1:
        all_detail=[];exception_keys=[]
        for name,(b,l,rows) in per_account.items():
            st.markdown(f"### {name}")
            out=[];keys=[]
            for row in rows:
                bdesc=", ".join(b.loc[i,"description"] for i in row["bank_idx"]) if row["bank_idx"] else ""
                ldesc=", ".join(l.loc[i,"description"] for i in row["ledger_idx"]) if row["ledger_idx"] else ""
                jvno=", ".join(str(l.loc[i,"journal_id"]) for i in row["ledger_idx"] if l.loc[i,"journal_id"]) if row["ledger_idx"] else ""
                key=row_key(name,row,b,l)
                out.append({"status":row["status"],"bank":bdesc,"ledger":ldesc,"전표번호":jvno,"date_diff_days":row["days"],
                            "amount_diff":row["diff"],"의심 유형":row["hint"],
                            "확인상태":"✅ 확인완료" if key in reviewed_map else ("-" if row["status"] in MATCHED_STATUS else "미확인")})
                keys.append(key)
                all_detail.append({"계좌":name,"status":row["status"],"bank":bdesc,"ledger":ldesc,"전표번호":jvno,
                                    "date_diff_days":row["days"],"amount_diff":row["diff"],"의심유형":row["hint"]})
                exception_keys.append(key)
            res=pd.DataFrame(out)
            a,c,d=st.columns(3)
            a.metric("전체",len(res));c.metric("매칭",int(res.status.isin(MATCHED_STATUS).sum()));d.metric("예외",int((~res.status.isin(MATCHED_STATUS)).sum()))
            st.dataframe(res,use_container_width=True)

            exc_rows=[(i,row) for i,row in enumerate(rows) if row["status"] not in MATCHED_STATUS]
            unresolved=[(i,row) for i,row in exc_rows if keys[i] not in reviewed_map]
            if unresolved:
                with st.expander(f"미확인 예외 {len(unresolved)}건 처리"):
                    for i,row in unresolved:
                        bdesc=", ".join(b.loc[j,"description"] for j in row["bank_idx"]) if row["bank_idx"] else "-"
                        ldesc=", ".join(l.loc[j,"description"] for j in row["ledger_idx"]) if row["ledger_idx"] else "-"
                        jvno=", ".join(str(l.loc[j,"journal_id"]) for j in row["ledger_idx"] if l.loc[j,"journal_id"]) if row["ledger_idx"] else ""
                        st.write(f"**{row['status']}** — 은행: {bdesc} / 원장: {ldesc}" + (f" (전표번호: {jvno})" if jvno else ""))
                        note=st.text_input("확인 사유",key=f"note_{name}_{i}")
                        if st.button("확인완료로 표시",key=f"btn_{name}_{i}"):
                            mark_reviewed(conn,keys[i],name,row["status"],note or "사유 미기재")
                            st.rerun()

            if len(unresolved) and st.button(f"🔍 {name} — AI로 예외 원인 분석",key=f"ai_{name}"):
                if ollama_ok():
                    try:
                        ex_records=[out[i] for i,_ in unresolved]
                        reasons=ai_explain(ex_records,MODEL)
                        ex_out=pd.DataFrame(ex_records)
                        ex_out["추정 원인"]=[reasons.get(str(i),"") for i in range(len(ex_records))]
                        st.dataframe(ex_out,use_container_width=True)
                    except Exception as e:
                        st.warning(f"AI 원인 분석 실패: {e}")
                else:
                    st.warning("Ollama가 연결되어 있지 않아 원인 분석을 할 수 없습니다.")
            st.divider()

        detail_df=pd.DataFrame(all_detail)
        st.session_state._detail_df=detail_df
        st.session_state._exception_keys=exception_keys

    with tab2:
        snapshots=[daily_snapshot(b,name) for name,(b,l,rows) in per_account.items()]
        report_dates=[s["기준일"] for s in snapshots if s["기준일"]]
        report_date=max(report_dates) if report_dates else datetime.now().date()
        st.markdown(f"#### 기준일: {report_date}")

        snap_df=pd.DataFrame(snapshots)
        st.dataframe(snap_df,use_container_width=True)
        totals=snap_df[["전일잔액","금일입금","금일출금","금일잔액"]].sum()
        st.markdown(f"**합계** — 전일잔액 {totals['전일잔액']:,.0f} / 금일입금 {totals['금일입금']:,.0f} / 금일출금 {totals['금일출금']:,.0f} / 금일잔액 {totals['금일잔액']:,.0f}")

        detail_df=st.session_state.get("_detail_df")
        exception_keys=st.session_state.get("_exception_keys")
        if detail_df is not None:
            reviewed_map=get_reviewed_map(conn)
            matched_cnt=int(detail_df.status.isin(MATCHED_STATUS).sum())
            unresolved_cnt=sum(1 for v,k in zip(detail_df.to_dict("records"),exception_keys) if v["status"] not in MATCHED_STATUS and k not in reviewed_map)
            c1,c2=st.columns(2)
            c1.metric("확정(정상 매칭)",f"{matched_cnt}건")
            c2.metric("미결(확인 필요)",f"{unresolved_cnt}건")

            if st.button("📥 일일 자금 보고 엑셀 다운로드",type="primary"):
                out_path="daily_report.xlsx"
                build_excel_report(report_date,snapshots,detail_df,reviewed_map,exception_keys,out_path)
                with open(out_path,"rb") as f:
                    st.download_button("다운로드 준비 완료 — 클릭해서 저장",f.read(),
                                        f"일일자금보고_{report_date}.xlsx",
                                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
