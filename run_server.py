import ssl
import uvicorn

if __name__ == "__main__":
    # Chạy với SSL
    uvicorn.run("backend.main:app", 
                host="0.0.0.0", 
                port=8000, 
                ssl_keyfile='./ssl/privkey.pem',
                ssl_certfile='./ssl/fullchain.pem')