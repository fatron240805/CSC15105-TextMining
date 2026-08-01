import torch, socket, sys
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_name(0), flush=True)
try:
    s = socket.create_connection(("huggingface.co", 443), timeout=10); s.close()
    print("INTERNET_OK", flush=True)
except Exception as e:
    print("INTERNET_FAIL", str(e)[:80], flush=True)
