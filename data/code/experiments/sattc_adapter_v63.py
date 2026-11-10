"""Pinned official SATTC scoring functions and label-free structural block.

AST extraction executes only the original numerical functions. The structural
block is taken unchanged from evaluate_model and audited to contain no target
or label references. This adapts the postprocessor to our frozen within-subject
features; it is not a reproduction of SATTC's LOSO encoder-training protocol.
"""
import ast,copy,contextlib,io,math
from typing import Any,Dict,Optional,Tuple
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from native_temporal_screen import ROOT,sha

SOURCE=ROOT/'code/upstream/SATTC/Retrieval/run_sattc_loso.py'
COMMIT='617cba60db52fa2e51e52f05b8f75bc13259f8ef'

def author_functions():
    tree=ast.parse(SOURCE.read_text(encoding='utf8'));names={'_eigh_inv_sqrt','estimate_mu_cov','subject_adaptive_whiten','csls_adaptive','build_structural_expert','fuse_poe_scores'}
    ns={'torch':torch,'F':F,'np':np,'math':math,'Optional':Optional,'Any':Any,'Dict':Dict,'Tuple':Tuple}
    chosen=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(chosen)==len(names)
    exec(compile(ast.Module(body=chosen,type_ignores=[]),str(SOURCE),'exec'),ns)
    eval_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_model')
    start=next(i for i,n in enumerate(eval_node.body) if isinstance(n,ast.Assign) and n.lineno==1436)
    nodes=copy.deepcopy(eval_node.body[start:start+6]);assert isinstance(nodes[-1],ast.Try) and nodes[-1].end_lineno==1765
    identifiers={n.id for statement in nodes for n in ast.walk(statement) if isinstance(n,ast.Name)}
    assert not any('label' in n or 'target' in n or 'truth' in n for n in identifiers)
    fn=ast.parse('def evidence(base_logits,config):\n    pass\n').body[0];fn.body=nodes+[ast.Return(value=ast.Name(id='pre_csls_evidence',ctx=ast.Load()))]
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));exec(compile(module,str(SOURCE),'exec'),ns)
    return ns

def score(query,gallery,ns=None):
    """All queries/gallery in one unlabeled evaluation pool; no labels accepted."""
    if ns is None:ns=author_functions()
    q=torch.tensor(query,dtype=torch.float32);v=torch.tensor(gallery,dtype=torch.float32)
    q=ns['subject_adaptive_whiten'](q,shrink=.6,diag=False);v=ns['subject_adaptive_whiten'](v,shrink=.5,diag=False)
    s=q@v.T;s=(s-s.mean())/s.std(unbiased=False).clamp_min(1e-6)
    config=SimpleNamespace(pre_csls_row_safe_k=35,pre_csls_col_safe_k=35,pre_csls_row_topL=5,pre_csls_case2_tau_ratio=.5,pre_csls_case2_col_penalty=.5,pre_csls_case5_hub_high_quantile=.85,pre_csls_case5_hub_mid_quantile=.70,pre_csls_bg_penalty=.3,pre_csls_case6_penalty_high=.5,pre_csls_case6_penalty_mid=.25)
    with contextlib.redirect_stdout(io.StringIO()):
        evidence=ns['evidence'](s,config);geometry=ns['csls_adaptive'](s,k0=12,kmin=5,kmax=20,alpha=1.,m=10,k_side=12,col_alpha=1.2,col_m=20,col_kmin=8,col_kmax=24)
        struct=ns['build_structural_expert'](evidence,geometry)
        result=ns['fuse_poe_scores'](geometry,struct,beta=1.9)
    assert torch.isfinite(result).all();return result.numpy()
