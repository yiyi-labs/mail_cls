import pytest
from spam_analysis.io import sha256
from spam_analysis.model_assets import verify_identity


def test_hf_style_snapshot_link_is_verified_by_content(tmp_path):
    blob=tmp_path/'blobs'/'fixed'; blob.parent.mkdir(); blob.write_bytes(b'fixed official asset')
    snapshot=tmp_path/'snapshots'/'revision'; snapshot.mkdir(parents=True)
    link=snapshot/'weights.bin'
    try:link.symlink_to(blob)
    except OSError as exc:pytest.skip('This host cannot create symbolic links: '+str(exc))
    identity={'folder':str(snapshot),'files':{'weights.bin':sha256(blob)}}
    assert verify_identity(identity)==identity
    blob.write_bytes(b'changed content')
    with pytest.raises(ValueError,match='identity mismatch'):verify_identity(identity)


@pytest.mark.parametrize('name',['../outside.bin','C:/outside.bin','/outside.bin','nested/../../outside.bin'])
def test_manifest_cannot_escape_by_name(tmp_path,name):
    with pytest.raises(ValueError,match='Unsafe model manifest path'):
        verify_identity({'folder':str(tmp_path),'files':{name:'untrusted'}})


def test_missing_or_changed_regular_model_file_is_rejected(tmp_path):
    file=tmp_path/'config.json';file.write_bytes(b'{}')
    identity={'folder':str(tmp_path),'files':{'config.json':sha256(file)}}
    verify_identity(identity)
    file.write_bytes(b'{"changed":true}')
    with pytest.raises(ValueError,match='identity mismatch'):verify_identity(identity)
    file.unlink()
    with pytest.raises(ValueError,match='identity mismatch'):verify_identity(identity)
