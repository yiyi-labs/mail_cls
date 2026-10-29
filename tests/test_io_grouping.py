from io import BytesIO
from zipfile import ZipFile
import pandas as pd
import pytest
from spam_analysis.io import read_index
from spam_analysis.grouping import grouping_view,shingles,jaccard,UnionFind
from spam_analysis.splitting import assert_disjoint

def archive(index):
    b=BytesIO()
    with ZipFile(b,"w") as z:
        z.writestr("trec/full/index",index); z.writestr("trec/data/000/000",b"body")
        z.writestr("trec/delay/index","Spam ../data/000/000\nSPAM ../data/000/000")
        z.writestr("__MACOSX/trec/full/index","junk")
    return ZipFile(b)

def test_only_full_valid_unique():
    rows=read_index(archive("spam ../data/000/000")); assert len(rows)==1 and rows[0]["label"]==1
    for bad in ["maybe ../data/000/000","spam ../data/000/000\nham ../data/000/000","spam ../../../private"]:
        with pytest.raises(ValueError): read_index(archive(bad))

def test_content_grouping_primitives():
    assert grouping_view("联系 a@b.com 12345678")==grouping_view("联系 c@d.com 98765432")
    assert jaccard(shingles("abcdef"),shingles("abcdef"))==1
    uf=UnionFind(3); uf.union(0,1); uf.union(1,2); assert uf.find(2)==uf.find(0)

def test_split_overlap_rejected():
    m=pd.DataFrame({"doc_id":["a","b"],"template_group_id":["g","g"],"split":["train","test"]})
    with pytest.raises(AssertionError): assert_disjoint(m)
