from locust import HttpUser, task


class MyTestUser(HttpUser):
    @task
    def predict_params(self):
        with open('../app/test_files/S_20250211_153211_ko_T223_0.png', 'rb') as f:
            binary = f.read()
        files = {"file": ("file", binary, "multipart/form-data")}
        response = self.client.post("/calculate-parameters", files=files)
        print(response)
