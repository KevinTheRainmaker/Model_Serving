import numpy as np
import json
import requests

# test용 더미 데이터 생성
xs = np.array([[9.0], [10.0]])
data = json.dumps({
    'signature_name':'serving_default', 
    'instances':xs.tolist()
    })

# request 보내기
headers = {'content-type':'application/json'}
json_response = requests.post(
    'http://localhost:8501/v1/models/helloworld:predict', 
    data=data, headers=headers
    )

# JSON 문자열 응답 확인
print(json_response.text)