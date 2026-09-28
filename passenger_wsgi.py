import sys
import os

# Limit OpenBLAS / NumPy / OpenMP threads to 1 to stay safely within cPanel nproc limit (40)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# Set root directory for cPanel Phusion Passenger
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

# Force cloud mode on cPanel hosting
os.environ["AI_MODE"] = os.getenv("AI_MODE", "cloud")

try:
    from src.api.routes import app
    from a2wsgi import ASGIMiddleware
    fastapi_asgi = ASGIMiddleware(app)

    def application(environ, start_response):
        # Fast path for direct health check and root check to avoid gateway timeout
        path_info = environ.get('PATH_INFO', '')
        if path_info in ['/', '', '/health']:
            status = '200 OK'
            output = b'{"status": "ok", "message": "Visual-RAG Cloud Backend is Running"}'
            response_headers = [
                ('Content-type', 'application/json'),
                ('Content-Length', str(len(output)))
            ]
            start_response(status, response_headers)
            return [output]
        
        return fastapi_asgi(environ, start_response)

except Exception as e:
    import traceback
    err_trace = traceback.format_exc()
    print(f"[Passenger Boot Error] {err_trace}", file=sys.stderr)
    
    # Fallback minimal WSGI app that prints the error in browser instead of 503
    def application(environ, start_response):
        status = '500 Internal Server Error'
        output = f"<h3>Application Startup Error</h3><pre>{err_trace}</pre>".encode('utf-8')
        response_headers = [('Content-type', 'text/html'), ('Content-Length', str(len(output)))]
        start_response(status, response_headers)
        return [output]
