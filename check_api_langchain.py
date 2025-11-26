import requests
import json

def test_ask():
    url = "http://localhost:8000/ask"
    session_id = "test_session_v1"
    
    # Question 1: Initial question
    q1 = "Phở bò bao nhiêu calo?"
    print(f"\n❓ Asking: {q1}")
    payload1 = {"question": q1, "session_id": session_id}
    try:
        resp1 = requests.post(url, json=payload1)
        resp1.raise_for_status()
        data1 = resp1.json()
        print(f"✅ Answer 1: {data1['answer']}")
        print(f"📄 Sources 1: {data1['sourceDocuments']}")
    except Exception as e:
        print(f"❌ Error 1: {e}")
        if resp1:
            print(resp1.text)
        return

    # Question 2: Follow-up question (Ambiguous)
    q2 = "Ăn món đó có mập không?"
    print(f"\n❓ Asking: {q2}")
    payload2 = {"question": q2, "session_id": session_id}
    try:
        resp2 = requests.post(url, json=payload2)
        resp2.raise_for_status()
        data2 = resp2.json()
        print(f"✅ Answer 2: {data2['answer']}")
        print(f"📄 Sources 2: {data2['sourceDocuments']}")
        
        # Verification logic
        if "Phở bò" in str(data2['sourceDocuments']) or "Phở" in str(data2['sourceDocuments']):
             print("🎉 SUCCESS: Context was correctly understood!")
        else:
             print("⚠️ WARNING: Check if context was understood. Sources might be irrelevant.")
             
    except Exception as e:
        print(f"❌ Error 2: {e}")
        if resp2:
            print(resp2.text)

if __name__ == "__main__":
    test_ask()
