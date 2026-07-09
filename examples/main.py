import requests

resp = requests.post("http://localhost:3000/insert", json={
	"hello": 123
})

print(resp.text)