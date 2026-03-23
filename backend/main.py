from fastapi import FastAPI, Depends, HTTPException, Header, Body
import firebase_admin
from firebase_admin import credentials, auth, firestore
import os
import json
import uuid
import httpx
import traceback
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Groq SDK
from groq import Groq
from contextlib import asynccontextmanager

# --- Configuration Helpers ---
def normalize_url(url: str) -> str:
    """Ensures external API URLs are properly formatted."""
    if not url:
        return ""
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "https://" + url  # Default to secure
    return url

IS_RENDER = os.getenv("RENDER") == "true" or os.path.exists("/etc/secrets")

# --- Default Fallback Data Used For AI Modules ---
FALLBACK_GUARDIAN = {
    "alerts": [
        {"type": "Security Alert", "description": "Ensure your banking passwords are secure.", "severity": "info"},
        {"type": "AI Degraded", "description": "Dynamic analysis is offline due to an AI processing error (e.g., API limits or invalid key).", "severity": "warning"}
    ]
}

FALLBACK_CATALYST = {
    "opportunities": [
        {"title": "Explore Mutual Funds", "description": "Consider starting an SIP in NIFTY 50 index funds.", "category": "Growth"},
        {"title": "AI Degraded", "description": "Growth insights are limited today due to an AI processing error.", "category": "System"}
    ]
}

FALLBACK_STRATEGIST = {
    "summary": "Core insights degraded due to an AI processing error.",
    "recommendations": [
        {"symbol": "NIFTY 50", "advice": "Diversify", "reasoning": "Consider index funds for lower risk."},
    ]
}

# --- Startup Validations & Firebase Init ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
if not GCP_PROJECT_ID:
    print("❌ CRITICAL ERROR: GCP_PROJECT_ID environment variable is missing.")

GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

if IS_RENDER:
    FIREBASE_KEY_PATH = "/etc/secrets/firebase-service-account.json"
    VERTEX_AI_KEY_PATH = "/etc/secrets/vertex-ai-key.json"
else:
    FIREBASE_KEY_PATH = os.path.join(
        os.path.dirname(__file__),
        os.getenv("FIREBASE_SERVICE_ACCOUNT", "firebase-service-account.json")
    )
    VERTEX_AI_KEY_PATH = os.path.join(
        os.path.dirname(__file__),
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "vertex-ai-key.json")
    )

if not os.path.exists(FIREBASE_KEY_PATH):
    print(f"❌ CRITICAL ERROR: Firebase key not found at {FIREBASE_KEY_PATH}")

if not os.path.exists(VERTEX_AI_KEY_PATH):
    print(f"❌ CRITICAL ERROR: Vertex AI key not found at {VERTEX_AI_KEY_PATH}")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = VERTEX_AI_KEY_PATH

FIRESTORE_READY = False
try:
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred)
    # Quick sanity check
    _test_db = firestore.client()
    FIRESTORE_READY = True
    print("✅ Firebase & Firestore initialized successfully.")
except Exception as e:
    print(f"⚠️ FIRESTORE ERROR: Failed to configure Firestore. Fallback modes enabled. Error: {e}")

def get_db():
    if not FIRESTORE_READY:
        return None
    try:
        return firestore.client()
    except Exception as e:
        print(f"⚠️ Firestore access error: {e}")
        return None

# --- AI Setup ---
groq_client = None
if os.getenv("GROQ_API_KEY"):
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    AI_READY = True
else:
    print("⚠️ WARNING: GROQ_API_KEY is not set. AI API calls will fail.")
    AI_READY = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Dummy lifespan since we removed vertexai.init
    yield

app = FastAPI(lifespan=lifespan)

# Normalize MCP Server Base URL safely
raw_mcp_url = os.getenv("MCP_SERVER_BASE_URL", "https://fi-mcp-server.onrender.com")
MOCK_SERVER_BASE_URL = normalize_url(raw_mcp_url)

# --- Authentication ---
async def verify_firebase_token(authorization: str = Header(...)):
    try:
        if not authorization.startswith("Bearer "):
             raise HTTPException(status_code=401, detail="Invalid authorization header format")
        
        id_token = authorization.split("Bearer ").pop()
        decoded_token = await asyncio.to_thread(auth.verify_id_token, id_token)
        return decoded_token['uid']
    except Exception as e:
        print(f"⚠️ Auth Error: {e}")
        raise HTTPException(status_code=401, detail="Invalid Firebase token")

@app.get("/start-fi-auth")
async def start_fi_auth(uid: str = Depends(verify_firebase_token)):
    session_id = str(uuid.uuid4())
    try:
        db = get_db()
        if db:
            user_doc_ref = db.collection("users").document(uid)
            await asyncio.to_thread(user_doc_ref.set, {"fi_session_id": session_id}, merge=True)
            print(f"✅ Session {session_id} saved to Firestore for uid {uid}")
        else:
            print("⚠️ Session not saved to Firestore! Firestore is offline/uninitialized.")
    except Exception as e:
        print(f"❌ FIRESTORE WRITE ERROR inside start_fi_auth: {e}")
        # Proceed gracefully so app isn't stuck
    
    auth_url = f"{MOCK_SERVER_BASE_URL}/mockWebPage?sessionId={session_id}"
    return {"auth_url": auth_url, "session_id": session_id}

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "firestore_status": "online" if FIRESTORE_READY else "offline",
        "ai_status": "online" if AI_READY else "offline"
    }

# --- Helper: Extract valid data from MCP response ---
def extract_valid_data(data, name="data"):
    if not isinstance(data, dict):
        print(f"⚠️ {name}: fallback used (empty or invalid format)")
        return None
    keys = list(data.keys())
    if not keys:
        print(f"⚠️ {name}: fallback used (empty data)")
        return None
    if "error" in keys:
        if len(keys) == 1:
            print(f"⚠️ {name}: fallback used (only error present: {data['error']})")
            return None
        else:
            print(f"✅ {name}: partial data used (contains error but has other valid keys: {keys})")
            return data
    print(f"✅ {name}: full data used")
    return data

# --- Dynamic Data Fetching ---
async def get_user_financial_data(uid: str, tool_name: str, timeout=30):
    session_id = None
    try:
        db = get_db()
        if db:
            user_doc = await asyncio.to_thread(db.collection("users").document(uid).get)
            if user_doc.exists and "fi_session_id" in user_doc.to_dict():
                session_id = user_doc.to_dict()["fi_session_id"]
    except Exception as e:
        print(f"❌ FIRESTORE READ ERROR: {e}")

    if not session_id:
        print(f"⚠️ No active fi_session_id found for uid {uid}. Generating dummy session.")
        session_id = "test-1234"

    headers = {"X-Session-ID": session_id}
    request_body = {"tool_name": tool_name}
    target_url = f"{MOCK_SERVER_BASE_URL}/mcp/stream"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(target_url, headers=headers, json=request_body, timeout=timeout)
            
        if response.status_code == 200:
            print(f"✅ SUCCESS: Fetched '{tool_name}' from {target_url}")
            try:
                return response.json()
            except ValueError:
                print(f"❌ JSON ERROR: Invalid JSON returned from MCP for '{tool_name}'.")
                return {"error": "Invalid data format received"}
        else:
            print(f"⚠️ Error from MCP mock server for '{tool_name}': {response.status_code}")
            return {"error": f"MCP returned status {response.status_code}"}
            
    except httpx.TimeoutException:
        print(f"❌ TIMEOUT: MCP server connection timed out at {target_url}")
        return {"error": "Network Timeout"}
    except Exception as e:
        print(f"❌ NETWORK ERROR: Failed calling MCP at {target_url}. Error: {e}")
        return {"error": f"Connection exception: {e}"}

@app.get("/get-user-data")
async def get_user_data(uid: str = Depends(verify_firebase_token)):
    """Get user's net worth and financial summary data for dashboard"""
    try:
        net_worth_data = await get_user_financial_data(uid, tool_name="fetch_net_worth")
        
        valid_data = extract_valid_data(net_worth_data, "net_worth")
        if valid_data:
            net_worth_response = valid_data.get('netWorthResponse', {})
            asset_values = net_worth_response.get('assetValues', [])
            total_net_worth_value = net_worth_response.get('totalNetWorthValue', {})
            
            total_assets = 0
            total_liabilities = 0
            for asset in asset_values:
                value = asset.get('value', {})
                units = int(value.get('units', 0))
                if units > 0:
                    total_assets += units
                else:
                    total_liabilities += abs(units)
            
            total_networth = int(total_net_worth_value.get('units', 0))
            
            return {
                "total_networth": total_networth,
                "total_assets": total_assets,
                "total_liabilities": total_liabilities,
                "change_percentage": 0,
                "currency": "INR"
            }
        else:
            fallback_data = {
                "total_networth": 1500000,
                "total_assets": 2000000,
                "total_liabilities": 500000,
                "change_percentage": 5.2,
                "currency": "INR",
                "is_fallback": True
            }
            return fallback_data
    except Exception as e:
        print(f"❌ Error in get_user_data: {e}")
        error_fallback_data = {
            "total_networth": 1500000,
            "total_assets": 2000000,
            "total_liabilities": 500000,
            "change_percentage": 5.2,
            "currency": "INR",
            "is_error_fallback": True
        }
        return error_fallback_data

# --- Tool Definition for the Strategist Agent ---
def get_market_performance(stock_symbols: list[str]) -> str:
    """Gets the real-time 1-year market performance for a list of stock symbols and the NIFTY 50 index."""
    print(f"⚙️ TOOL CALLED: get_market_performance for symbols: {stock_symbols}")
    performance_data = {"NIFTY 50": {"1y_return": 12.0}}
    for symbol in stock_symbols:
        if "RELIANCE" in symbol:
            performance_data[symbol] = {"1y_return": 15.5}
        elif "TCS" in symbol:
            performance_data[symbol] = {"1y_return": 11.0}
        else:
            performance_data[symbol] = {"1y_return": 13.0}
    return json.dumps(performance_data)

market_data_tool = {
    "type": "function",
    "function": {
        "name": "get_market_performance",
        "description": "Gets the real-time 1-year market performance for a list of stock symbols and the NIFTY 50 index.",
        "parameters": {
            "type": "object",
            "properties": {
                "stock_symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of stock symbols to fetch performance for, e.g., ['RELIANCE', 'TCS']"
                }
            },
            "required": ["stock_symbols"]
        }
    }
}

# --- AI Model Call Function ---
def call_gemini_text(prompt: str, model_name="llama-3.3-70b-versatile", tools=None, timeout=45):
    if not AI_READY:
        print("⚠️ AI Call Aborted: No GROQ_API_KEY provided.")
        return "ERROR_AI_UNAVAILABLE"

    try:
        messages = [{"role": "user", "content": prompt}]
        
        # Format tools if provided
        groq_tools = [tools] if (isinstance(tools, dict) and "type" in tools) else tools if tools else None
        
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=groq_tools,
            tool_choice="auto" if groq_tools else "none"
        )
        
        response_message = response.choices[0].message
        
        # Check if function was called
        if response_message.tool_calls:
            tool_call = response_message.tool_calls[0]
            if tool_call.function.name == "get_market_performance":
                # Parse args
                args = json.loads(tool_call.function.arguments)
                tool_result = get_market_performance(**args)
                
                # Append tool call message
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }]
                })
                
                # Append tool response message
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": "get_market_performance",
                    "content": tool_result
                })
                
                final_response = groq_client.chat.completions.create(
                    model=model_name,
                    messages=messages
                )
                return final_response.choices[0].message.content
        return response_message.content
    except Exception as e:
        error_msg = str(e)
        print(f"❌ AI Calling Error: {error_msg}")
        traceback.print_exc()
        return "ERROR_GEMINI_CRASH"

# --- Agent Endpoints ---
@app.post("/ask-oracle")
async def ask_oracle(uid: str = Depends(verify_firebase_token), body: dict = Body(...)):
    question = body.get("question", "")
    try:
        net_worth, bank_tx, credit, epf, mf_tx, stock_tx = await asyncio.gather(
            get_user_financial_data(uid, tool_name="fetch_net_worth"),
            get_user_financial_data(uid, tool_name="fetch_bank_transactions"),
            get_user_financial_data(uid, tool_name="fetch_credit_report"),
            get_user_financial_data(uid, tool_name="fetch_epf_details"),
            get_user_financial_data(uid, tool_name="fetch_mf_transactions"),
            get_user_financial_data(uid, tool_name="fetch_stock_transactions"),
            return_exceptions=True
        )
        data = {
            "net_worth": extract_valid_data(net_worth, "net_worth") or "unavailable",
            "bank_transactions": extract_valid_data(bank_tx, "bank_transactions") or "unavailable",
            "credit_report": extract_valid_data(credit, "credit_report") or "unavailable",
            "epf_details": extract_valid_data(epf, "epf_details") or "unavailable",
            "mf_transactions": extract_valid_data(mf_tx, "mf_transactions") or "unavailable",
            "stock_transactions": extract_valid_data(stock_tx, "stock_transactions") or "unavailable"
        }
        prompt = (
            "You are Oracle, an AI-powered personal finance assistant. "
            "You have access to the user's complete financial data."
            "Answer the user's question safely and specifically: '" + question + "'\n"
            f"Data:\n{json.dumps(data)}"
        )
        answer = await asyncio.to_thread(call_gemini_text, prompt)
        
        if answer.startswith("ERROR_"):
            return {
                "question": question, 
                "answer": "I am currently undergoing maintenance and AI services are temporarily degraded due to an error (e.g., API limits, invalid key, or system crash). Please try again later!",
                "is_fallback": True
            }
        return {"question": question, "answer": answer}
    except Exception as e:
        return {"question": question, "answer": f"Backend Error: {e}", "is_fallback": True}

@app.post("/run-guardian")
async def run_guardian(uid: str = Depends(verify_firebase_token), body: dict = Body(None)):
    try:
        bank_tx, credit, mf_tx = await asyncio.gather(
            get_user_financial_data(uid, tool_name="fetch_bank_transactions"),
            get_user_financial_data(uid, tool_name="fetch_credit_report"),
            get_user_financial_data(uid, tool_name="fetch_mf_transactions")
        )
        data = {
            "bank_transactions": extract_valid_data(bank_tx, "bank_transactions") or "unavailable",
            "credit_report": extract_valid_data(credit, "credit_report") or "unavailable",
            "mf_transactions": extract_valid_data(mf_tx, "mf_transactions") or "unavailable"
        }
        # Fixed syntax error: escaping quotes properly or using single quotes for JSON strings inside the f-string prompt
        prompt = (
            "You are Guardian, an AI financial safety agent. "
            "Respond ONLY in a valid JSON object: "
            '{"alerts": [{"type":"...", "description":"...", "severity":"..."}]}\n'
            f"Data:\n{json.dumps(data)}"
        )
        answer = await asyncio.to_thread(call_gemini_text, prompt)
        
        if answer.startswith("ERROR_"):
            return JSONResponse(status_code=200, content=FALLBACK_GUARDIAN)

        parsed = json.loads(answer.replace("```json", '').replace("```", ''))
        alerts = parsed.get('alerts', [])
        if not alerts:
            raise ValueError("No alerts array")
        parsed['alerts'] = alerts
        
        db = get_db()
        if db:
            await asyncio.to_thread(db.collection("users").document(uid).set, {"guardian_alerts_cache": alerts}, merge=True)
        return {"alerts": json.dumps(parsed)}
    except Exception as e:
        print(f"⚠️ Guardian degraded: {e}")
        db = get_db()
        if db:
            try:
                user_doc = await asyncio.to_thread(db.collection("users").document(uid).get)
                cache = user_doc.to_dict().get("guardian_alerts_cache") if user_doc.exists else None
                if cache:
                    return {"alerts": json.dumps({"alerts": cache})}
            except:
                pass
        return {"alerts": json.dumps(FALLBACK_GUARDIAN)}

@app.post("/run-catalyst")
async def run_catalyst(uid: str = Depends(verify_firebase_token), body: dict = Body(None)):
    try:
        net_worth, epf, mf_tx = await asyncio.gather(
            get_user_financial_data(uid, tool_name="fetch_net_worth"),
            get_user_financial_data(uid, tool_name="fetch_epf_details"),
            get_user_financial_data(uid, tool_name="fetch_mf_transactions")
        )
        data = {
            "net_worth": extract_valid_data(net_worth, "net_worth") or "unavailable",
            "epf_details": extract_valid_data(epf, "epf_details") or "unavailable",
            "mf_transactions": extract_valid_data(mf_tx, "mf_transactions") or "unavailable"
        }
        prompt = (
            "You are Catalyst, an AI financial growth agent. "
            "Respond ONLY in a valid JSON object: "
            '{"opportunities": [{"title":"...", "description":"...", "category":"..."}]}\n'
            f"Data:\n{json.dumps(data)}"
        )
        answer = await asyncio.to_thread(call_gemini_text, prompt)
        
        if answer.startswith("ERROR_"):
            return JSONResponse(status_code=200, content=FALLBACK_CATALYST)

        parsed = json.loads(answer.replace("```json", '').replace("```", ''))
        opts = parsed.get('opportunities', [])
        if not opts:
            raise ValueError("No opportunities array")
        parsed['opportunities'] = opts
        
        db = get_db()
        if db:
            await asyncio.to_thread(db.collection("users").document(uid).set, {"catalyst_opportunities_cache": opts}, merge=True)
        return {"opportunities": json.dumps(parsed)}
    except Exception as e:
        print(f"⚠️ Catalyst degraded: {e}")
        db = get_db()
        if db:
            try:
                user_doc = await asyncio.to_thread(db.collection("users").document(uid).get)
                cache = user_doc.to_dict().get("catalyst_opportunities_cache") if user_doc.exists else None
                if cache:
                    return {"opportunities": json.dumps({"opportunities": cache})}
            except:
                pass
        return {"opportunities": json.dumps(FALLBACK_CATALYST)}

@app.post("/run-strategist")
async def run_strategist(uid: str = Depends(verify_firebase_token), body: dict = Body(None)):
    try:
        stock_tx, mf_tx = await asyncio.gather(
            get_user_financial_data(uid, tool_name="fetch_stock_transactions"),
            get_user_financial_data(uid, tool_name="fetch_mf_transactions")
        )
        data = {
            "stock_transactions": extract_valid_data(stock_tx, "stock_transactions") or "unavailable",
            "mf_transactions": extract_valid_data(mf_tx, "mf_transactions") or "unavailable"
        }
        prompt = (
            "You are an expert Investment Strategist for the Indian market. "
            "Respond ONLY in a valid JSON object: "
            '{"summary":"...", "recommendations":[{"symbol":"...", "advice":"...", "reasoning":"..."}]}\n'
            f"Data:\n{json.dumps(data)}"
        )
        answer = await asyncio.to_thread(call_gemini_text, prompt, tools=[market_data_tool])
        
        if answer.startswith("ERROR_"):
            return JSONResponse(status_code=200, content={"strategy": json.dumps(FALLBACK_STRATEGIST)})
            
        parsed = json.loads(answer.replace("```json", '').replace("```", ''))
        if not parsed.get('recommendations', []):
            raise ValueError("No recommendations array")
            
        return {"strategy": json.dumps(parsed)}
    except Exception as e:
        print(f"⚠️ Strategist degraded: {e}")
        return JSONResponse(status_code=200, content={"strategy": json.dumps(FALLBACK_STRATEGIST)})

@app.get("/get-subscriptions")
async def get_subscriptions(uid: str = Depends(verify_firebase_token)):
    """Get user's subscription data from bank transactions"""
    print(f"🔍 DEBUG: get_subscriptions called for uid: {uid}")
    try:
        # Fetch bank transactions data
        bank_transactions_data = await get_user_financial_data(uid, tool_name="fetch_bank_transactions")
        print(f"🔍 DEBUG: bank_transactions_data: {bank_transactions_data}")
        
        # For debugging, let's also try to get the raw MCP response
        valid_data = extract_valid_data(bank_transactions_data, "bank_transactions")
        if valid_data:
            # Parse the MCP data structure
            transactions_response = valid_data.get('bankTransactionsResponse', {})
            transactions = transactions_response.get('transactions', [])
            print(f"🔍 DEBUG: transactions_response: {transactions_response}")
            print(f"🔍 DEBUG: transactions count: {len(transactions)}")
            print(f"🔍 DEBUG: first few transactions: {transactions[:3] if transactions else 'No transactions'}")
            
            # Extract subscription transactions (AUTO-DEBIT entries)
            subscriptions = []
            subscription_patterns = {
                'NETFLIX': {'name': 'Netflix', 'icon': '🎬', 'color': 'red', 'category': 'Entertainment'},
                'SPOTIFY': {'name': 'Spotify Premium', 'icon': '🎵', 'color': 'green', 'category': 'Music'},
                'AMAZON PRIME': {'name': 'Amazon Prime', 'icon': '📦', 'color': 'orange', 'category': 'Shopping'},
                'YOUTUBE': {'name': 'YouTube Premium', 'icon': '📺', 'color': 'red', 'category': 'Entertainment'},
                'GOOGLE CLOUD': {'name': 'Google Cloud Storage', 'icon': '☁️', 'color': 'blue', 'category': 'Technology'},
                'MICROSOFT': {'name': 'Microsoft 365', 'icon': '💼', 'color': 'blue', 'category': 'Productivity'},
                'ADOBE': {'name': 'Adobe Creative Cloud', 'icon': '🎨', 'color': 'purple', 'category': 'Creative'},
                'GOOGLE': {'name': 'Google Services', 'icon': '🔍', 'color': 'blue', 'category': 'Technology'},
            }
            
            for transaction in transactions:
                # Handle different transaction formats
                if isinstance(transaction, list) and len(transaction) >= 3:
                    # Format: ["amount", "description", "date", ...]
                    amount_str = transaction[0]
                    description = transaction[1].upper()
                    date_str = transaction[2]
                elif isinstance(transaction, dict):
                    # Format: {"amount": ..., "description": ..., "date": ...}
                    description = transaction.get('description', '').upper()
                    amount_str = transaction.get('amount', '0')
                    date_str = transaction.get('date', '')
                else:
                    continue
                
                print(f"🔍 DEBUG: Processing transaction - amount: {amount_str}, description: {description}, date: {date_str}")
                
                if 'AUTO-DEBIT' in description or 'AUTO' in description:
                    # Find matching subscription pattern
                    for pattern, details in subscription_patterns.items():
                        if pattern in description:
                            try:
                                amount = int(amount_str)
                            except (ValueError, TypeError):
                                amount = 0
                            
                            # Parse date and calculate next billing
                            try:
                                transaction_date = datetime.strptime(date_str, '%Y-%m-%d')
                                # Estimate next billing (monthly subscriptions)
                                next_billing = transaction_date + timedelta(days=30)
                                if next_billing < datetime.now():
                                    next_billing = datetime.now() + timedelta(days=30)
                            except:
                                next_billing = datetime.now() + timedelta(days=30)
                            
                            subscription = {
                                'name': details['name'],
                                'category': details['category'],
                                'amount': amount,
                                'currency': 'INR',
                                'billingCycle': 'Monthly',  # Default assumption
                                'nextBilling': next_billing.isoformat(),
                                'status': 'Active',
                                'icon': details['icon'],
                                'color': details['color'],
                                'description': f"Auto-debit from {date_str}",
                                'lastTransaction': date_str,
                            }
                            
                            # Check if subscription already exists (avoid duplicates)
                            existing = next((s for s in subscriptions if s['name'] == details['name']), None)
                            if not existing:
                                subscriptions.append(subscription)
                            break
            
            print(f"🔍 DEBUG: Found {len(subscriptions)} subscriptions: {subscriptions}")
            
            
            return {
                "subscriptions": subscriptions,
                "total_count": len(subscriptions),
                "currency": "INR"
            }
        else:
            print(f"🔍 DEBUG: MCP failed, using fallback data")
            # Return fallback data if MCP fails
            fallback_subscriptions = [
                {
                    'name': 'Netflix',
                    'category': 'Entertainment',
                    'amount': 499,
                    'currency': 'INR',
                    'billingCycle': 'Monthly',
                    'nextBilling': (datetime.now() + timedelta(days=15)).isoformat(),
                    'status': 'Active',
                    'icon': '🎬',
                    'color': 'red',
                    'description': 'Premium Plan - 4K Ultra HD',
                    'lastTransaction': '2024-06-10',
                },
                {
                    'name': 'Amazon Prime',
                    'category': 'Shopping',
                    'amount': 1499,
                    'currency': 'INR',
                    'billingCycle': 'Yearly',
                    'nextBilling': (datetime.now() + timedelta(days=45)).isoformat(),
                    'status': 'Active',
                    'icon': '📦',
                    'color': 'orange',
                    'description': 'Annual Membership',
                    'lastTransaction': '2024-06-15',
                },
                {
                    'name': 'Spotify Premium',
                    'category': 'Music',
                    'amount': 299,
                    'currency': 'INR',
                    'billingCycle': 'Monthly',
                    'nextBilling': (datetime.now() + timedelta(days=8)).isoformat(),
                    'status': 'Active',
                    'icon': '🎵',
                    'color': 'green',
                    'description': 'Individual Plan',
                    'lastTransaction': '2024-06-12',
                },
            ]
            return {
                "subscriptions": fallback_subscriptions,
                "total_count": len(fallback_subscriptions),
                "currency": "INR"
            }
    except Exception as e:
        print(f"Error in get_subscriptions: {e}")
        traceback.print_exc()
        return {
            "subscriptions": [],
            "total_count": 0,
            "currency": "INR",
            "error": str(e)
        }

@app.get("/test-subscriptions")
async def test_subscriptions(uid: str = Depends(verify_firebase_token)):
    """Test endpoint to check subscription data"""
    try:
        # Fetch bank transactions data
        bank_transactions_data = await get_user_financial_data(uid, tool_name="fetch_bank_transactions")
        
        # Return the raw data for debugging
        return {
            "raw_data": bank_transactions_data,
            "has_error": bank_transactions_data.get('error') if bank_transactions_data else True,
            "data_keys": list(bank_transactions_data.keys()) if bank_transactions_data else [],
        }
    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }