from pathlib import Path

import numpy as np
import pytest

from cectcls.manifest import write_json
from cectcls.synthesis import export_requests, register_outputs


def test_synthesis_roundtrip(tmp_path):
    matched = tmp_path / "matched.json"
    requests = tmp_path / "requests.json"
    syn_index = tmp_path / "syn_index.json"
    write_json(matched, [{"patient_id": "P001", "selected_positions": [
        {"reference_slice_id": 3, "matched_ncct_slice_id": 5}]}])
    rows = export_requests(matched, tmp_path / "syn", requests)
    assert len(rows) == 2
    with pytest.raises(FileNotFoundError):
        register_outputs(requests, syn_index)
    for row in rows:
        np.save(row["output_path"], np.ones((32, 32), dtype=np.float32))
    mapping = register_outputs(requests, syn_index)
    assert set(mapping) == {"P001|3|ap", "P001|3|pvp"}
