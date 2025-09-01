import pytest
from src.serving.inference import score_record
from src.serving.schemas import Txn


def test_api_vs_offline(client):
    # Example fake transaction
    sample = {
        "step": 5,
        "type": "TRANSFER",
        "amount": 5000.0,
        "nameOrig": "C123456",
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 5000.0,
        "nameDest": "M987654",
        "oldbalanceDest": 2000.0,
        "newbalanceDest": 7000.0,
    }

    txn = Txn(**sample)

    # Offline inference
    offline_result = score_record(txn)

    # API inference
    response = client.post("/score", json=sample)
    assert response.status_code == 200
    api_result = response.json()

    # Compare outputs
    assert "fraud_probability" in api_result
    assert pytest.approx(api_result["fraud_probability"], rel=1e-3) == offline_result["fraud_probability"]
    assert api_result["is_fraud"] == offline_result["is_fraud"]
