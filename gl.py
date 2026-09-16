"""Full GL import boundary; monetary calculations never use an LLM."""
import re
from decimal import Decimal, InvalidOperation
from datetime import date,datetime
from numbers import Real
import pandas as pd

GL_FIELDS = ['journal_id', 'posting_date', 'account_code', 'account_name',
             'debit', 'credit', 'counterparty', 'description']


def amount(value):
    if pd.isna(value) or str(value).strip() in ('', '-'):
        return 0.0
    s = str(value).replace(',', '').replace('원', '').strip()
    if s.startswith('(') and s.endswith(')'):
        s = '-' + s[1:-1]
    try:
        if not re.fullmatch(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)',s):
            raise ValueError('금액 형식 오류')
        n = Decimal(s)
        if not n.is_finite() or n != n.quantize(Decimal('.01')) or abs(n) > Decimal('9000000000000'):
            raise ValueError('금액 범위 또는 소수 자릿수 오류')
        return float(n)
    except (InvalidOperation, ValueError):
        raise ValueError(f'유효하지 않은 금액: {value}') from None


def dates(values):
    def parse(v):
        if pd.isna(v) or str(v).strip() == '':
            return pd.NaT
        if isinstance(v, Real):
            return pd.Timestamp('1899-12-30') + pd.to_timedelta(v, unit='D')
        if isinstance(v,(date,datetime,pd.Timestamp)):
            return pd.Timestamp(v).tz_localize(None)
        match=re.match(r'^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:$|[ T])',str(v))
        if not match:return pd.NaT
        try:return pd.Timestamp(*map(int,match.groups()))
        except ValueError:return pd.NaT
    return values.map(parse).dt.normalize()


def validate_mapping(df, mapping, kind):
    required = {'ledger': ['journal_id', 'posting_date', 'account_code', 'account_name', 'debit', 'credit'],
                'bank': ['transaction_date', 'withdrawal', 'deposit'],
                'card': ['transaction_date', 'merchant', 'amount', 'status']}[kind]
    missing = [f for f in required if mapping.get(f) not in df.columns]
    if missing:
        raise ValueError('필수 컬럼 매핑 누락: ' + ', '.join(missing))
    mapped = [v for v in mapping.values() if v in df.columns]
    if len(mapped) != len(set(mapped)):
        raise ValueError('같은 원본 컬럼을 여러 표준 필드에 매핑할 수 없습니다.')
    if df.empty:
        raise ValueError('파일에 데이터 행이 없습니다.')


def normalize_gl(df, mapping):
    validate_mapping(df, mapping, 'ledger')
    out = pd.DataFrame(index=range(len(df)))
    for f in GL_FIELDS:
        out[f] = df[mapping[f]].reset_index(drop=True) if mapping.get(f) in df.columns else ''
    for f in ['journal_id', 'account_code', 'account_name', 'counterparty', 'description']:
        out[f] = out[f].map(lambda v: '' if pd.isna(v) else str(int(v)) if isinstance(v,Real) and float(v).is_integer() else str(v).strip())
    out['posting_date'] = dates(out.posting_date)
    for f in ['debit', 'credit']:
        out[f] = out[f].map(amount)
    if (out.journal_id == '').any() or (out.account_code == '').any() or (out.account_name == '').any() or out.posting_date.isna().any():
        raise ValueError('GL의 전표번호·전기일·계정코드·계정과목을 확인하세요. 빈 값 또는 잘못된 날짜가 있습니다.')
    if ((out.debit < 0) | (out.credit < 0) | ((out.debit > 0) & (out.credit > 0))).any():
        raise ValueError('GL 금액은 음수 없이 차변 또는 대변 한쪽에 입력하세요.')
    out['source_row'] = out.index + 2
    # Account codes are the identity. Names may vary between rows.
    out['account_key'] = out.account_code
    return out


def deposit_accounts(gl):
    codes = set(gl.loc[gl.account_name.str.replace(r'\s+', '', regex=True).str.contains('보통예금'), 'account_code'])
    return {code: code + ' ' + ', '.join(dict.fromkeys(gl.loc[gl.account_code == code, 'account_name']))
            for code in gl.account_code.drop_duplicates() if code in codes}


def gl_warnings(gl):
    # Sum integer cents, matching gl.js and avoiding float cancellation.
    totals={}
    for row in gl.itertuples():
        totals[row.journal_id]=totals.get(row.journal_id,0)+round(row.debit*100)-round(row.credit*100)
    bad=[journal for journal,value in totals.items() if value!=0]
    return ['차대변 불일치 전표: ' + ', '.join(bad)] if bad else []


def extract_ledger(gl, account_key):
    if account_key not in deposit_accounts(gl):
        raise ValueError('발견된 보통예금 계정을 선택하세요.')
    out = gl[(gl.account_key == account_key) & ((gl.debit != 0) | (gl.credit != 0))].copy()
    groups = {j: rows for j, rows in gl.groupby('journal_id', sort=False)}
    for idx, row in out.iterrows():
        peers = groups[row.journal_id]
        peers = peers[peers.account_key != account_key]
        for field in ['counterparty', 'description']:
            values = [row[field]] + peers[field].tolist()
            out.at[idx, field] = ', '.join(dict.fromkeys(v for v in values if v))
        out.at[idx, 'account'] = ', '.join(dict.fromkeys(peers.account_name))
    out['cash_in'] = out.debit
    out['cash_out'] = out.credit
    out['signed'] = (out.debit - out.credit).round(2)
    if 'account' not in out:
        out['account'] = ''
    return out.reset_index(drop=True)
