import uvicorn

if __name__ == "__main__":
	# 서버 구동 이후 디버깅을 위해 reload=True 옵션을 추가한다
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)