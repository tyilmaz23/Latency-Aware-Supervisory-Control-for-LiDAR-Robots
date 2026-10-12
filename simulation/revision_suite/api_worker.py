"""Minimal isolated REST worker. No torch, training, or filesystem model loads."""
import json,os,sys,time
import requests

def emit(data):
    print(json.dumps(data,allow_nan=False),flush=True)
def main():
    emit({"event":"worker_ready"})
    session=requests.Session()
    for line in sys.stdin:
        request=json.loads(line);t0=time.perf_counter();raw="";failed=False;called=False;error="";usage={};model_version=""
        try:
            if request.get("stub"):
                raw=json.dumps(request["stub_decision"])
            elif request.get("forced_failure"):
                failed=True;error="injected_failure"
            else:
                key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                if not key:raise RuntimeError("missing_api_key")
                called=True
                model=request["model"]
                body={"contents":[{"role":"user","parts":[{"text":request["prompt"]}]}],
                      "generationConfig":{"temperature":0,"responseMimeType":"application/json","maxOutputTokens":1024,
                                          "thinkingConfig":{"thinkingBudget":0}}}
                response=session.post("https://generativelanguage.googleapis.com/v1beta/models/"+model+":generateContent",
                                      headers={"x-goog-api-key":key},json=body,timeout=(5,float(request.get("http_timeout_s",15))))
                if response.status_code!=200:raise RuntimeError("http_"+str(response.status_code))
                payload=response.json();usage=payload.get("usageMetadata",{});model_version=payload.get("modelVersion","")
                raw="".join(part.get("text","") for c in payload.get("candidates",[]) for part in c.get("content",{}).get("parts",[]) if not part.get("thought",False))
                if not raw:raise RuntimeError("empty_model_response")
        except Exception as exc:
            failed=True
            # No exception URL, headers, API key, or request body is persisted.
            error=exc.args[0] if isinstance(exc,RuntimeError) else type(exc).__name__
        time.sleep(max(0,float(request.get("extra_delay_s",0))))
        emit({"event":"reply","id":request["id"],"raw":raw,"request_failed":failed,"error":str(error),
              "provider_called":called,"duration_s":time.perf_counter()-t0,"usage":usage,"model_version":model_version})
if __name__=="__main__":main()
