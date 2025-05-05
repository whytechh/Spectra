from os.path import realpath

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_read_api_is_runninng():
    response = client.get("/")

    assert response.status_code == 200


def test_api_send_422_on_invalid_request():
    response = client.post("/calculate-parameters")

    assert response.status_code == 422


def test_api_accept_invalid_form_data_request():
    files = {"file": ("file", [0, 1, 2], "multipart/form-data")}

    response = client.post("/calculate-parameters", files=files)

    assert response.status_code == 400


def test_api_predict_parameters():
    with open(realpath(f'{realpath(__file__)}/../test_files/S_20250211_153211_ko_T223_0.png'), 'rb') as f:
        binary = f.read()
    files = {"file": ("file", binary, "multipart/form-data")}

    response = client.post("/calculate-parameters", files=files)
    result = response.json()
    assert response.status_code == 200
    assert 'result' in result
