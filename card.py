"""Deterministic corporate-card import and GL reconciliation.

Card use credits the card payable. Its later debit settlement is a bank item,
not a second card use. Tax treatment is supplied by the GL, never inferred here.
"""
import re
import pandas as pd
from gl import amount, dates, validate_mapping

CARD_FIELDS = ['transaction_date', 'merchant', 'amount', 'status',
               'approval_number', 'card_number']
CARD_LABELS = {'PENDING': '매입대기', 'CONFIRMED': '매입완료',
               'CANCELLED': '승인취소'}
STATUS_ALIASES = {
    'PENDING': {'매입대기', '미매입', '승인대기', 'pending', '미확정'},
    'CONFIRMED': {'매입완료', '확정', 'posted', 'confirmed'},
    'CANCELLED': {'승인취소', '취소', 'cancelled', 'canceled', 'reversed'},
}
CARD_DATE_WINDOW = 7


def card_status(value):
    token = re.sub(r'\s+', '', str(value or '')).lower()
    for standard, aliases in STATUS_ALIASES.items():
        if token in aliases:
            return standard
    raise ValueError(f'알 수 없는 카드 거래상태: {value}')


def normalize_card(df, mapping):
    validate_mapping(df, mapping, 'card')
    out = pd.DataFrame(index=range(len(df)))
    for field in CARD_FIELDS:
        source = mapping.get(field)
        out[field] = df[source].reset_index(drop=True) if source in df.columns else ''
    out.transaction_date = dates(out.transaction_date)
    out.amount = out.amount.map(amount)
    for field in ('merchant', 'approval_number', 'card_number'):
        out[field] = out[field].map(lambda value: '' if pd.isna(value) else str(value).strip())
    out.status = out.status.map(card_status)
    if out.transaction_date.isna().any() or (out.merchant == '').any() or (out.amount <= 0).any():
        raise ValueError('카드 이용일·가맹점·양수 이용금액을 확인하세요.')
    out['source_row'] = out.index + 2
    return out


def card_payable_accounts(gl):
    mask = gl.account_name.str.replace(r'\s+', '', regex=True).str.contains('미지급금') & \
           gl.account_name.str.replace(r'\s+', '', regex=True).str.contains('카드')
    codes = set(gl.loc[mask, 'account_code'])
    return {code: code + ' ' + ', '.join(dict.fromkeys(gl.loc[gl.account_code == code, 'account_name']))
            for code in gl.account_code.drop_duplicates() if code in codes}


def extract_card_ledger(gl, account_key):
    if account_key not in card_payable_accounts(gl):
        raise ValueError('카드 미지급금 계정을 확인하세요.')
    groups = {j: rows for j, rows in gl.groupby('journal_id', sort=False)}
    results = []
    for row in gl[(gl.account_code == account_key) & (gl.credit > 0)].itertuples():
        peers = groups[row.journal_id]
        peers = peers[peers.account_code != account_key]
        accounts = list(dict.fromkeys(peers.loc[~peers.account_name.str.contains('부가세대급금'), 'account_name']))
        party = ', '.join(dict.fromkeys(v for v in peers.counterparty if v))
        description = ', '.join(dict.fromkeys(v for v in [row.description, *peers.description] if v))
        vat = round(sum(round(float(v) * 100) for v in peers.loc[peers.account_name.str.contains('부가세대급금'), 'debit']) / 100, 2)
        results.append({'journal_id': row.journal_id, 'posting_date': row.posting_date,
                        'expense_accounts': ', '.join(accounts), 'vat_amount': vat,
                        'counterparty': party, 'description': description,
                        'payable_amount': round(row.credit, 2), 'source_row': row.source_row})
    return pd.DataFrame(results, columns=['journal_id', 'posting_date', 'expense_accounts',
                                         'vat_amount', 'counterparty', 'description',
                                         'payable_amount', 'source_row'])


def _name(value):
    return re.sub(r'[^0-9a-zA-Z가-힣]', '', str(value)).lower()


def _merchant_matches(card, ledger):
    from party import party_relation
    return any(party_relation(card.merchant, party) != 'none'
               for party in str(ledger.counterparty).split(', '))


def _merchant_method(card, ledger):
    from party import party_relation
    relations=[party_relation(card.merchant, party) for party in str(ledger.counterparty).split(', ')]
    return next((method for method in ('exact','normalized','alias') if method in relations),'none')


def reconcile_card(cards, ledger):
    """Match confirmed uses only. Prefer exact amount, then closest date/name.

    The approval number is retained in the card row for future GL references.
    Without such a reference, amount/date/merchant are the deterministic keys.
    """
    used_card, used_ledger = set(), set()
    rows = []
    for index, card in cards.iterrows():
        if card.status in ('PENDING', 'CANCELLED'):
            rows.append({'status': card.status, 'card_idx': [index], 'ledger_idx': [],
                         'days': None, 'diff': None,
                         'hint': ('카드사에서 아직 매입 확정되지 않아 GL 매칭 대상에서 제외'
                                  if card.status == 'PENDING' else '카드사 취소 거래 - 정상 대사 대상 제외')})
            used_card.add(index)

    def candidates(exact):
        choices = []
        for ci, card in cards.iterrows():
            if ci in used_card or card.status != 'CONFIRMED':
                continue
            for li, item in ledger.iterrows():
                if li in used_ledger or not _merchant_matches(card, item):
                    continue
                days = abs((card.transaction_date - item.posting_date).days)
                if days > CARD_DATE_WINDOW:
                    continue
                if (abs(card.amount - item.payable_amount) < .01) != exact:
                    continue
                rank={'exact':0,'normalized':1,'alias':2}[_merchant_method(card,item)]
                choices.append((days, abs(card.amount - item.payable_amount), rank, ci, li))
        # Exact amount wins as a phase; nearest date wins within each phase.
        return sorted(choices)

    for exact in (True, False):
        for days, difference, rank, ci, li in candidates(exact):
            if ci in used_card or li in used_ledger:
                continue
            used_card.add(ci); used_ledger.add(li)
            rows.append({'status': ('MATCHED' if days <= 1 else 'MATCHED (날짜 차이)') if exact else 'AMOUNT MISMATCH',
                         'card_idx': [ci], 'ledger_idx': [li], 'days': days,
                         'diff': round(cards.loc[ci, 'amount'] - ledger.loc[li, 'payable_amount'], 2),
                         'hint': ({'normalized':'가맹점 표기 정규화','alias':'가맹점 별칭'}.get(_merchant_method(cards.loc[ci],ledger.loc[li]),'')
                                  if exact else '확정 카드 이용금액과 GL 카드미지급금 대변금액 확인')})
    for ci, card in cards.iterrows():
        if ci in used_card or card.status != 'CONFIRMED':
            continue
        options=[]
        for li, item in ledger.iterrows():
            if li in used_ledger or abs(card.amount-item.payable_amount)>=.01:
                continue
            days=abs((card.transaction_date-item.posting_date).days)
            if days<=CARD_DATE_WINDOW:
                options.append((li,days))
        if len(options)==1:
            li,days=options[0]
            used_card.add(ci);used_ledger.add(li)
            rows.append({'status':'거래처 확인 필요','card_idx':[ci],'ledger_idx':[li],
                         'days':days,'diff':0.0,'hint':'금액·날짜 일치, 가맹점명 확인 필요'})
        elif len(options)>1:
            used_card.add(ci)
            labels=', '.join(str(ledger.loc[li,'counterparty']) for li,_ in options[:5])
            rows.append({'status':'거래처 확인 필요','card_idx':[ci],'ledger_idx':[],
                         'days':None,'diff':None,'hint':'동일 금액·날짜 GL 후보: '+labels})
    for ci, card in cards.iterrows():
        if ci not in used_card:
            rows.append({'status': 'CARD ONLY', 'card_idx': [ci], 'ledger_idx': [],
                         'days': None, 'diff': None, 'hint': '확정 거래이나 GL 카드 사용 전표 없음'})
    for li in ledger.index:
        if li not in used_ledger:
            rows.append({'status': 'LEDGER ONLY', 'card_idx': [], 'ledger_idx': [li],
                         'days': None, 'diff': None, 'hint': '카드 GL 전표는 있으나 카드사 이용내역 없음'})
    return rows


def card_summary(cards, rows):
    return {'total': len(cards),
            'matched': sum(r['status'] in ('MATCHED', 'MATCHED (날짜 차이)') for r in rows),
            'pending': sum(r['status'] == 'PENDING' for r in rows),
            'cancelled': sum(r['status'] == 'CANCELLED' for r in rows),
            'review': sum(r['status'] in ('CARD ONLY', 'LEDGER ONLY', 'AMOUNT MISMATCH','거래처 확인 필요','AI 후보 매칭') for r in rows)}
