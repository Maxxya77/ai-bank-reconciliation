// Same import contract as gl.py. Account identity is the ERP account code.
const GL_FIELDS = ['journal_id','posting_date','account_code','account_name','debit','credit','counterparty','description'];
const PARTY_ALIASES={한전:'한국전력공사',kepco:'한국전력공사',한국전력서울본부:'한국전력공사',신한카드결제:'신한카드',shinhancard:'신한카드',서울아리수본부:'서울아리수본부'};
function normalizeParty(value){return String(value??'').trim().toLowerCase().replace(/\(\s*주\s*\)|㈜|주식회사/g,'').replace(/[^0-9a-z가-힣]/g,'');}
function partyRelation(external,glParty){
  const left=String(external??'').trim(),right=String(glParty??'').trim();
  if(!left||!right)return 'none';
  if(left===right)return 'exact';
  const a=normalizeParty(left),b=normalizeParty(right);
  if(!a||!b)return 'none';
  if(a===b)return 'normalized';
  return (PARTY_ALIASES[a]||a)===(PARTY_ALIASES[b]||b)?'alias':'none';
}
function validateMapping(rows,m,kind){
  if(!rows.length) throw Error('파일에 데이터 행이 없습니다.');
  const cols=Object.keys(rows[0]);
  const required={ledger:GL_FIELDS.slice(0,6),bank:['transaction_date','withdrawal','deposit'],
    card:['transaction_date','merchant','amount','status']}[kind];
  const missing=required.filter(f=>!cols.includes(m[f]));
  if(missing.length) throw Error('필수 컬럼 매핑 누락: '+missing.join(', '));
  const values=Object.values(m).filter(v=>cols.includes(v));
  if(new Set(values).size!==values.length) throw Error('같은 원본 컬럼을 여러 표준 필드에 매핑할 수 없습니다.');
}
function normalizeGL(rows,m){
  validateMapping(rows,m,'ledger');
  return rows.map((r,i)=>{
    const o={};
    GL_FIELDS.forEach(f=>o[f]=String(r[m[f]]??'').trim());
    o.date=parseDate(r[m.posting_date]);
    o.debit=num(r[m.debit]); o.credit=num(r[m.credit]);
    if(!o.journal_id||!o.account_code||!o.account_name||!o.date) throw Error('GL의 전표번호·전기일·계정코드·계정과목을 확인하세요.');
    if(o.debit<0||o.credit<0||(o.debit>0&&o.credit>0)) throw Error('GL 금액은 음수 없이 차변 또는 대변 한쪽에 입력하세요.');
    o.source_row=i+2; o.account_key=o.account_code;
    return o;
  });
}
function depositAccounts(gl){
  const codes=new Set(gl.filter(r=>r.account_name.replace(/\s/g,'').includes('보통예금')).map(r=>r.account_code));
  return [...new Set(gl.map(r=>r.account_code))].filter(c=>codes.has(c)).map(key=>({key,label:key+' '+[...new Set(gl.filter(r=>r.account_code===key).map(r=>r.account_name))].join(', ')}));
}
function glWarnings(gl){
  const sums=new Map();
  gl.forEach(r=>sums.set(r.journal_id,(sums.get(r.journal_id)||0)+Math.round(r.debit*100)-Math.round(r.credit*100)));
  const bad=[...sums].filter(([j,v])=>v!==0).map(([j])=>j);
  return bad.length?['차대변 불일치 전표: '+bad.join(', ')]:[];
}
function extractLedger(gl,key){
  if(!depositAccounts(gl).some(a=>a.key===key)) throw Error('발견된 보통예금 계정을 선택하세요.');
  // Card-use journals contain no cash line; only the monthly settlement enters reconciliation.
  const groups=new Map();
  gl.forEach(r=>{if(!groups.has(r.journal_id))groups.set(r.journal_id,[]);groups.get(r.journal_id).push(r);});
  const join=values=>[...new Set(values.filter(Boolean))].join(', ');
  return gl.filter(r=>r.account_key===key&&(r.debit!==0||r.credit!==0)).map(r=>{
    const peers=groups.get(r.journal_id).filter(p=>p.account_key!==key);
    return {...r,journalId:r.journal_id,description:join([r.description,...peers.map(p=>p.description)]),
      counterparty:join([r.counterparty,...peers.map(p=>p.counterparty)]),account:join(peers.map(p=>p.account_name)),
      cashIn:r.debit,cashOut:r.credit,signed:(Math.round(r.debit*100)-Math.round(r.credit*100))/100};
  });
}

// Card use and its later bank settlement are distinct journal events.
const CARD_FIELDS=['transaction_date','merchant','amount','status','approval_number','card_number'];
const CARD_LABELS={PENDING:'매입대기',CONFIRMED:'매입완료',CANCELLED:'승인취소'};
const CARD_ALIASES={
  PENDING:['매입대기','미매입','승인대기','pending','미확정'],
  CONFIRMED:['매입완료','확정','posted','confirmed'],
  CANCELLED:['승인취소','취소','cancelled','canceled','reversed']
};
function cardStatus(value){
  const token=String(value??'').replace(/\s+/g,'').toLowerCase();
  for(const [standard,aliases] of Object.entries(CARD_ALIASES))if(aliases.includes(token))return standard;
  throw Error('알 수 없는 카드 거래상태: '+value);
}
function normalizeCard(rows,m){
  validateMapping(rows,m,'card');
  return rows.map((r,i)=>{
    const date=parseDate(r[m.transaction_date]);
    const merchant=String(r[m.merchant]??'').trim();
    const amount=num(r[m.amount]);
    const status=cardStatus(r[m.status]);
    if(!date||!merchant||amount<=0)throw Error('카드 이용일·가맹점·양수 이용금액을 확인하세요.');
    const optional=f=>m[f]?String(r[m[f]]??'').trim():'';
    return {date,merchant,amount,status,approval_number:optional('approval_number'),
      card_number:optional('card_number'),source_row:i+2};
  });
}
function cardPayableAccounts(gl){
  const codes=new Set(gl.filter(r=>{const name=r.account_name.replace(/\s/g,'');return name.includes('미지급금')&&name.includes('카드');}).map(r=>r.account_code));
  return [...new Set(gl.map(r=>r.account_code))].filter(c=>codes.has(c)).map(key=>({key,label:key+' '+[...new Set(gl.filter(r=>r.account_code===key).map(r=>r.account_name))].join(', ')}));
}
function extractCardLedger(gl,key){
  if(!cardPayableAccounts(gl).some(a=>a.key===key))throw Error('카드 미지급금 계정을 확인하세요.');
  const groups=new Map();
  gl.forEach(r=>{if(!groups.has(r.journal_id))groups.set(r.journal_id,[]);groups.get(r.journal_id).push(r);});
  const unique=values=>[...new Set(values.filter(Boolean))].join(', ');
  return gl.filter(r=>r.account_code===key&&r.credit>0).map(r=>{
    const peers=groups.get(r.journal_id).filter(p=>p.account_code!==key);
    const vat=peers.filter(p=>p.account_name.includes('부가세대급금')).reduce((sum,p)=>sum+Math.round(p.debit*100),0)/100;
    return {journal_id:r.journal_id,date:r.date,expense_accounts:unique(peers.filter(p=>!p.account_name.includes('부가세대급금')).map(p=>p.account_name)),
      vat_amount:vat,counterparty:unique(peers.map(p=>p.counterparty)),
      description:unique([r.description,...peers.map(p=>p.description)]),
      payable_amount:r.credit,source_row:r.source_row};
  });
}
function cardMerchantMatches(card,ledger){
  return String(ledger.counterparty||'').split(', ').some(p=>partyRelation(card.merchant,p)!=='none');
}
function cardMerchantMethod(card,ledger){
  const found=String(ledger.counterparty||'').split(', ').map(p=>partyRelation(card.merchant,p));
  return ['exact','normalized','alias'].find(method=>found.includes(method))||'none';
}
function reconcileCard(cards,ledger){
  const usedC=new Set(),usedL=new Set(),rows=[];
  cards.forEach((c,i)=>{
    if(c.status==='PENDING'||c.status==='CANCELLED'){
      usedC.add(i);
      rows.push({status:c.status,cardIdx:[i],ledgerIdx:[],days:null,diff:null,
        hint:c.status==='PENDING'?'카드사에서 아직 매입 확정되지 않아 GL 매칭 대상에서 제외':'카드사 취소 거래 - 정상 대사 대상 제외'});
    }
  });
  for(const exact of [true,false]){
    const choices=[];
    cards.forEach((c,ci)=>{
      if(usedC.has(ci)||c.status!=='CONFIRMED')return;
      ledger.forEach((l,li)=>{
        if(usedL.has(li)||!cardMerchantMatches(c,l))return;
        const days=Math.abs((c.date-l.date)/86400000);
        if(days>7||(Math.abs(c.amount-l.payable_amount)<.01)!==exact)return;
        choices.push({days,difference:Math.abs(c.amount-l.payable_amount),rank:{exact:0,normalized:1,alias:2}[cardMerchantMethod(c,l)],ci,li});
      });
    });
    choices.sort((a,b)=>a.days-b.days||a.difference-b.difference||a.rank-b.rank||a.ci-b.ci||a.li-b.li);
    choices.forEach(({days,ci,li})=>{
      if(usedC.has(ci)||usedL.has(li))return;
      usedC.add(ci);usedL.add(li);
      rows.push({status:exact?(days<=1?'MATCHED':'MATCHED (날짜 차이)'):'AMOUNT MISMATCH',
        cardIdx:[ci],ledgerIdx:[li],days,diff:Math.round((cards[ci].amount-ledger[li].payable_amount)*100)/100,
        hint:exact?({normalized:'가맹점 표기 정규화',alias:'가맹점 별칭'}[cardMerchantMethod(cards[ci],ledger[li])]||''):'확정 카드 이용금액과 GL 카드미지급금 대변금액 확인'});
    });
  }
  cards.forEach((c,ci)=>{
    if(usedC.has(ci)||c.status!=='CONFIRMED')return;
    const options=[];
    ledger.forEach((l,li)=>{
      if(usedL.has(li)||Math.abs(c.amount-l.payable_amount)>=.01)return;
      const days=Math.abs((c.date-l.date)/86400000);
      if(days<=7)options.push({li,days});
    });
    if(options.length===1){
      const {li,days}=options[0];usedC.add(ci);usedL.add(li);
      rows.push({status:'거래처 확인 필요',cardIdx:[ci],ledgerIdx:[li],days,diff:0,hint:'금액·날짜 일치, 가맹점명 확인 필요'});
    }else if(options.length>1){
      usedC.add(ci);
      rows.push({status:'거래처 확인 필요',cardIdx:[ci],ledgerIdx:[],days:null,diff:null,
        hint:'동일 금액·날짜 GL 후보: '+options.slice(0,5).map(o=>ledger[o.li].counterparty).join(', ')});
    }
  });
  cards.forEach((c,ci)=>{if(!usedC.has(ci))rows.push({status:'CARD ONLY',cardIdx:[ci],ledgerIdx:[],days:null,diff:null,hint:'확정 거래이나 GL 카드 사용 전표 없음'});});
  ledger.forEach((l,li)=>{if(!usedL.has(li))rows.push({status:'LEDGER ONLY',cardIdx:[],ledgerIdx:[li],days:null,diff:null,hint:'카드 GL 전표는 있으나 카드사 이용내역 없음'});});
  return rows;
}
function cardSummary(cards,rows){return {total:cards.length,matched:rows.filter(r=>r.status==='MATCHED'||r.status==='MATCHED (날짜 차이)').length,
  pending:rows.filter(r=>r.status==='PENDING').length,cancelled:rows.filter(r=>r.status==='CANCELLED').length,
  review:rows.filter(r=>['CARD ONLY','LEDGER ONLY','AMOUNT MISMATCH','거래처 확인 필요','AI 후보 매칭'].includes(r.status)).length};}
