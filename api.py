from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os
import uuid
import json
from datetime import datetime
import redis.asyncio as redis
from generator_graph import compiled_graph, AgentState, export_testcases_to_excel
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(
    title="Test Case Generator API",
    description="AI-powered test case generation using LangGraph and Gemini",
    version="1.0.0"
)

redis_client: Optional[redis.Redis] = None



class JiraTicketRequest(BaseModel):
    ticket: str


class ClarificationResponse(BaseModel):
    session_id: str
    clarifications: str


class UserClarifications(BaseModel):
    session_id: str
    answers: str


class TestCaseResponse(BaseModel):
    test_cases: str
    excel_file: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    global redis_client
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    redis_client = await redis.from_url(redis_url, decode_responses=True)
    print(f"Connected to Redis at {redis_url}")


@app.on_event("shutdown")
async def shutdown_event():
    if redis_client:
        await redis_client.close()
        print("Redis connection closed")


async def save_session(session_id: str, data: dict):
    serialized_data = json.dumps(data, default=str)
    await redis_client.set(f"session:{session_id}", serialized_data)


async def get_session(session_id: str) -> Optional[dict]:
    data = await redis_client.get(f"session:{session_id}")
    return json.loads(data) if data else None


async def delete_session(session_id: str):
    await redis_client.delete(f"session:{session_id}")


def serialize_messages(messages):
    return [
        {
            "type": msg.__class__.__name__,
            "content": msg.content
        }
        for msg in messages
    ]


def deserialize_messages(messages_data):
    message_types = {
        "HumanMessage": HumanMessage,
        "AIMessage": AIMessage
    }
    return [
        message_types[msg["type"]](content=msg["content"])
        for msg in messages_data
    ]


@app.post("/api/v1/start", response_model=ClarificationResponse)
async def start_generation(request: JiraTicketRequest):
    """
    Submit a JIRA ticket and receive clarification questions.
    """
    try:
        session_id = str(uuid.uuid4())

        state: AgentState = {
            "messages": [HumanMessage(content=request.ticket)],
            "clarifications_asked": False
        }

        clarifications_output = compiled_graph.invoke(state)
        clarifications = clarifications_output["messages"][-1].content

        session_data = {
            "messages": serialize_messages(clarifications_output["messages"]),
            "clarifications_asked": clarifications_output.get("clarifications_asked", False),
            "created_at": datetime.now().isoformat()
        }
        await save_session(session_id, session_data)

        return ClarificationResponse(
            session_id=session_id,
            clarifications=clarifications
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/v1/generate", response_model=TestCaseResponse)
async def generate_test_cases(request: UserClarifications, background_tasks: BackgroundTasks):
    try:
        session_data = await get_session(request.session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        previous_messages = deserialize_messages(session_data["messages"])

        state_with_clarifications: AgentState = {
            "messages": previous_messages + [HumanMessage(content=request.answers)],
            "clarifications_asked": True
        }

        testcases_output = compiled_graph.invoke(state_with_clarifications)
        testcases = testcases_output["messages"][-1].content

        # Export to Excel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_cases_{request.session_id[:8]}_{timestamp}.xlsx"
        excel_path = export_testcases_to_excel(testcases, filename)

        # Clean up session
        background_tasks.add_task(delete_session, request.session_id)

        return TestCaseResponse(
            test_cases=testcases,
            excel_file=filename if excel_path else None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/api/v1/download/{filename}")
async def download_excel(filename: str):
    """
    Download the generated Excel file.
    """
    try:
        # Use local path when not in Docker
        exports_dir = os.getenv("EXPORTS_DIR", "./exports")
        file_path = os.path.join(exports_dir, filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/api/v1/health")
async def health_check():
    """
    Check API health status.
    """
    try:
        # Check Redis connectivity
        await redis_client.ping()
        redis_status = "connected"
    except:
        redis_status = "disconnected"

    return {
        "status": "healthy" if redis_status == "connected" else "degraded",
        "timestamp": datetime.now().isoformat(),
        "redis": redis_status
    }


@app.delete("/api/v1/session/{session_id}")
async def clear_session(session_id: str):
    try:
        await delete_session(session_id)
        return {"message": "Session cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
