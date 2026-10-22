import pytest
import torch

from cectcls.models import DualPhaseMIL


@pytest.mark.parametrize("phase", ["dual", "ncct"])
def test_resnet_mil_shapes_and_padding(phase):
    model = DualPhaseMIL("resnet18_mil", num_classes=4, phase_mode=phase,
                         pretrained=False, encode_chunk=3).eval()
    ap, pvp = torch.rand(2, 3, 1, 64, 64), torch.rand(2, 3, 1, 64, 64)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    with torch.no_grad():
        output = model(ap, pvp, mask)
    assert output["logits"].shape == (2, 4)
    assert output["attention_scores"].shape == (2, 3)
    assert output["attention_scores"][0, 2].item() < 1e-6
    assert torch.allclose(output["attention_scores"].sum(dim=1), torch.ones(2), atol=1e-5)
