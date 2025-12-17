import asyncio

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

app = FastAPI()

# Allow CORS just in case, though we are proxying
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEST_RESPONSE = {
    "results": [
        {
            "agent": "TestAgent",
            "scenario": "TestScenario",
            "passed": False,
            "transcript": [
                {
                    "role": "system",
                    "content": "You are a helpful agent. Say hello in the beginning. Then after getting the first response, say goodbye and hang up the call. Do not wait for the next response.",
                },
                {"role": "system", "content": "Call the user."},
                {"role": "system", "content": "end the call if the user says goodbye"},
                {"role": "user", "content": "Hello. How can I help you?"},
                {"role": "assistant", "content": "Hello! I'm here to assist you with any questions or tasks you have. How can I help you today?"},
                {
                    "role": "user",
                    "content": "Sure thing. What can I help you with today? Whether it's a dental question, scheduling an appointment, or anything else,\nlet me know.",
                },
            ],
            "recording_url": "https://api.twilio.com/2010-04-01/Accounts/ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX/Recordings/REXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            "error": None,
            "evaluations": {
                "Greeting and Farewell": {
                    "passed": True,
                    "reasoning": "The assistant greeted the user with 'Hello!' and said 'Goodbye!' as instructed.",
                },
                "Call Termination": {
                    "passed": True,
                    "reasoning": "The assistant ended the interaction after saying 'Goodbye!' as per the instructions.",
                },
            },
        }
    ],
    "passed": False,
}

LOG_DATA_RAW = """data: 2025-12-16 16:11:08 | INFO | Setting VAD params to: confidence=0.7 start_secs=0.2 stop_secs=0.8 min_volume=0.6
data:

data: 2025-12-16 16:11:08 | DEBUG | Loading Silero VAD model...
data:

data: 2025-12-16 16:11:08 | DEBUG | Loaded Silero VAD
data:

data: 2025-12-16 16:11:08 | DEBUG | Linking PipelineSource#0 -> FastAPIWebsocketInputTransport#0
data:

data: 2025-12-16 16:11:08 | DEBUG | Linking FastAPIWebsocketInputTransport#0 -> DeepgramSTTService#0
data:

data: 2025-12-16 16:11:08 | DEBUG | Linking DeepgramSTTService#0 -> OpenAIUserContextAggregator#0
data:

data: 2025-12-16 16:11:08 | DEBUG | Linking OpenAIUserContextAggregator#0 -> OpenAILLMService#0
data:

data: 2025-12-16 16:11:08 | DEBUG | Linking OpenAILLMService#0 -> CartesiaTTSService#0
data:

data: 2025-12-16 16:11:08 | DEBUG | Linking CartesiaTTSService#0 -> FastAPIWebsocketOutputTransport#0
data:

data: 2025-12-16 16:11:08 | DEBUG | Linking FastAPIWebsocketOutputTransport#0 -> OpenAIAssistantContextAggregator#0
data:

data: 2025-12-16 16:11:08 | DEBUG | Linking OpenAIAssistantContextAggregator#0 -> PipelineSink#0
data:

data: 2025-12-16 16:11:08 | DEBUG | Linking Source#0 -> Pipeline#0
data:

data: 2025-12-16 16:11:08 | DEBUG | Linking Pipeline#0 -> Sink#0
data:

data: 2025-12-16 16:11:08 | DEBUG | Runner PipelineRunner#0 started running PipelineTask#0
data:

data: 2025-12-16 16:11:08 | DEBUG | Connecting to Deepgram
data:

data: 2025-12-16 16:11:10 | DEBUG | Connecting to Cartesia
data:

: ping - 2025-12-16 10:41:19.689144+00:00

data: 2025-12-16 16:11:20 | DEBUG | User started speaking
data:

data: 2025-12-16 16:11:20 | DEBUG | User stopped speaking
data:

data: 2025-12-16 16:11:21 | DEBUG | Generating chat: [{"role": "system", "content": "You are a helpful agent. Say hello in the beginning. Then after getting the first response, say goodbye and hang up the call. Do not wait for the next response."}, {"role": "system", "content": "Call the user."}, {"role": "system", "content": "end the call if the user says goodbye"}, {"role": "user", "content": "Hello. How can I help you?"}]
data:

data: 2025-12-16 16:11:23 | DEBUG | Generating TTS: [Hello!]
data:

data: 2025-12-16 16:11:23 | DEBUG | Generating TTS: [ I just wanted to greet you and initiate our conversation.]
data:

data: 2025-12-16 16:11:23 | DEBUG | Generating TTS: [ If you have any questions or need assistance, I'm here to help.]
data:

data: 2025-12-16 16:11:23 | DEBUG | Bot started speaking
data:

data: 2025-12-16 16:11:26 | DEBUG | User started speaking
data:

data: 2025-12-16 16:11:26 | DEBUG | Bot stopped speaking
data:

data: 2025-12-16 16:11:28 | DEBUG | User stopped speaking
data:

data: 2025-12-16 16:11:28 | DEBUG | Generating chat: [{"role": "system", "content": "You are a helpful agent. Say hello in the beginning. Then after getting the first response, say goodbye and hang up the call. Do not wait for the next response."}, {"role": "system", "content": "Call the user."}, {"role": "system", "content": "end the call if the user says goodbye"}, {"role": "user", "content": "Hello. How can I help you?"}, {"role": "assistant", "content": "Hello! I just wanted to greet you and initiate our conversation."}, {"role": "user", "content": "Hello. How can I"}]
data:

data: 2025-12-16 16:11:30 | DEBUG | User started speaking
data:

data: 2025-12-16 16:11:33 | DEBUG | User stopped speaking
data:

data: 2025-12-16 16:11:33 | DEBUG | Generating chat: [{"role": "system", "content": "You are a helpful agent. Say hello in the beginning. Then after getting the first response, say goodbye and hang up the call. Do not wait for the next response."}, {"role": "system", "content": "Call the user."}, {"role": "system", "content": "end the call if the user says goodbye"}, {"role": "user", "content": "Hello. How can I help you?"}, {"role": "assistant", "content": "Hello! I just wanted to greet you and initiate our conversation."}, {"role": "user", "content": "Hello. How can I"}, {"role": "user", "content": "hi there? Thanks for reaching out. How can I help you today?"}]
data:

: ping - 2025-12-16 10:41:34.689684+00:00

data: 2025-12-16 16:11:35 | INFO | Calling function end_call with arguments {}
data:

data: 2025-12-16 16:11:35 | DEBUG | FunctionCallInProgressFrame: FunctionCallInProgressFrame#0
data:

data: 2025-12-16 16:11:37 | DEBUG | Disconnecting from Deepgram
data:

data: 2025-12-16 16:11:38 | DEBUG | Disconnecting from Cartesia
data:

data: 2025-12-16 16:11:38 | DEBUG | Runner PipelineRunner#0 finished running PipelineTask#0
data:

: ping - 2025-12-16 10:41:49.690833+00:00

: ping - 2025-12-16 10:42:04.692328+00:00

: ping - 2025-12-16 10:42:19.693122+00:00

: ping - 2025-12-16 10:42:34.695213+00:00
"""


@app.post("/test")
async def run_test(request: Request):
    # Simulate some delay if needed, or return immediately
    await asyncio.sleep(5)
    return TEST_RESPONSE


@app.get("/logs")
async def logs(request: Request):
    async def event_generator():
        lines = LOG_DATA_RAW.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("data: "):
                content = line[6:]
                # Emit data event - yield the content directly
                # sse_starlette will format it properly
                yield {"data": content}
                # Simulate timing - add delay to show streaming effect
                await asyncio.sleep(0.5)
            elif line.startswith(": ping"):
                # SSE Comment / Ping - skip or add small delay
                await asyncio.sleep(0.2)

            if await request.is_disconnected():
                break

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
