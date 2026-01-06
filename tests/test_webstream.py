import asyncio
import json
from typing import Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
from app.data_layer.data_classes.domain_models.user_input_source import UserInputSource
from app.services.handlers.realtime_voice_handler import BaseRealtimeVoiceHandler
from app.static_memory_cache import StaticMemoryCache
from app.telemetries.logger import logger
import time
 
class WebSocketStreamHandler:
    def __init__(
        self,
        max_pending_chunks: int = 15,
        processing_delay_threshold: float = 0.2,
        cleanup_interval: float = 2.0,
        max_concurrent_tasks: int = 25
    ):
        self.max_pending_chunks = max_pending_chunks
        self.processing_delay_threshold = processing_delay_threshold
        self.cleanup_interval = cleanup_interval
        self.processing_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.active_tasks: Set[asyncio.Task] = set()
        self.should_stop = asyncio.Event()
        self.chunk_counter = 0
        self.last_cleanup = time.time()
        self.loop = asyncio.get_event_loop()
        self.stream_sid: Optional[str] = None
        self.call_sid: Optional[str] = None  # Store the actual Twilio call SID
        self.message_counter = 0
        self.session_timeout = StaticMemoryCache.get_config("general", "call_session_timeout") # timeouts in seconds.
        self.session_start_time = None
        self.pending_first_response_message = None
 
 
    async def handle_stream(
        self,
        websocket: WebSocket,
        real_time_handler: BaseRealtimeVoiceHandler,
        user_input_source : UserInputSource
    ):
        try:
            await websocket.accept()
            logger.info("web_socket_stream_handler", message = "Incoming web socket connection is established.")
            self.session_start_time = time.time()
 
            # First response message is now handled by the voice handlers
 
            # Create timeout checker task
            timeout_checker = asyncio.create_task(self._check_session_timeout(websocket))
 
            # Send first response for web users immediately (they don't have 'start' events)
            # Twilio will send first response after the 'start' event
            await real_time_handler.lazy_initialize()
            if user_input_source == UserInputSource.WEBSITE:
                await real_time_handler.generate_first_response_from_agent(UserInputSource.WEBSITE)
 
            while not self.should_stop.is_set():
                if user_input_source == UserInputSource.WEBSITE:
                    await websocket.send_text(json.dumps({"event_type": "start_media_streaming"}))
                    # Handle web audio stream
                    while not self.should_stop.is_set():
                        data = await websocket.receive_bytes()
                        await self._process_incoming_data(data, real_time_handler)
                        await self._cleanup_tasks()
                else:
                    # Handle Twilio stream
                    async for message in websocket.iter_text():
                        if self.should_stop.is_set():
                            break
                        message_data = json.loads(message)
                        event_type = message_data.get('event')
 
                        if event_type == 'stop':
                            logger.info("web_socket_stream_handler", message="Twilio stream stopped")
                            self.should_stop.set()
                            break
 
                        await self._process_incoming_twilio_data(message, real_time_handler)
                        await self._cleanup_tasks()
        except WebSocketDisconnect:
            # TODO : call handle disconnect method from realtime voice handler to cleanup the resources.
            logger.error("web_socket_stream_handler", message="Media stream disconnected.")
        except Exception as e:
            logger.error(f"Error in stream handling: {e}")
            raise
        finally:
            if 'timeout_checker' in locals():
                timeout_checker.cancel()
                try:
                    await timeout_checker
                except asyncio.CancelledError:
                    pass
            await self._cleanup()
 
    async def _process_incoming_twilio_data(self, data: str, real_time_handler):
        self.message_counter += 1
        arrival_time = time.time()
 
        if len(self.active_tasks) > self.max_pending_chunks:
            logger.debug(f"Too many pending tasks ({len(self.active_tasks)}), skipping message {self.message_counter}")
            return
 
        task = asyncio.create_task(
            self._process_twilio_message(data, self.message_counter, arrival_time, real_time_handler)
        )
        self.active_tasks.add(task)
 
    async def _process_twilio_message(
        self,
        data: str,
        message_id: int,
        arrival_time: float,
        real_time_handler : BaseRealtimeVoiceHandler
    ):
        try:
            # Never skip critical events like 'start' or 'stop'
            message_data = json.loads(data)
            event_type = message_data.get('event')
            
            if event_type not in ['start', 'stop']:
                elapsed = time.time() - arrival_time
                if elapsed > self.processing_delay_threshold:
                    logger.debug(f"Skipping message {message_id} - too old ({elapsed:.3f}s)")
                    return
            else:
                logger.info(f"Processing critical event: {event_type} for message {message_id}")
 
            async with self.processing_semaphore:
                start_time = time.time()
                message_data = json.loads(data)
                event_type = message_data.get('event')
 
                if event_type == 'media':
                    await real_time_handler.handle_user_audio_stream(
                        message_data['media']['payload']
                    )
                elif event_type == 'start':
                    self.stream_sid = message_data['start']['streamSid']
                    # Try to get call SID from the start message if available
                    if 'callSid' in message_data['start']:
                        self.call_sid = message_data['start']['callSid']
                        logger.info(f"Started stream: {self.stream_sid} for call: {self.call_sid}")
                    else:
                        logger.info(f"Started stream: {self.stream_sid}")
 
                    # Send first response message now that Twilio stream is ready
                    # and then lazy initialize stt, so in deepgram STT, if we open up
                    # a websocket before hand, and response generation is taking time then
                    # it will timeout, and eventually call will be without STT.
                    await real_time_handler.generate_first_response_from_agent(UserInputSource.TWILIO)
                processing_time = time.time() - start_time
                if processing_time > 0.02:
                    logger.debug(f"Message {message_id} processed in {processing_time:.3f}s")
 
        except Exception as e:
            logger.error(f"Error processing Twilio message {message_id}: {e}")
        finally:
            self.active_tasks.discard(asyncio.current_task())
 
    async def _process_incoming_data(self, data, real_time_handler):
        self.chunk_counter += 1
        arrival_time = time.time()
 
        if len(self.active_tasks) > self.max_pending_chunks:
            logger.debug(f"Too many pending tasks ({len(self.active_tasks)}), skipping chunk {self.chunk_counter}")
            return
 
        task = asyncio.create_task(
            self._process_audio_chunk(data, self.chunk_counter, arrival_time, real_time_handler)
        )
        self.active_tasks.add(task)
 
    async def _process_audio_chunk(self, data: bytes, chunk_id: int, arrival_time: float, real_time_handler : BaseRealtimeVoiceHandler):
        try:
            elapsed = time.time() - arrival_time
            if elapsed > self.processing_delay_threshold:
                logger.debug(f"Skipping chunk {chunk_id} - too old ({elapsed:.3f}s)")
                return
 
            async with self.processing_semaphore:
                start_time = time.time()
 
                # Process web audio data
                await real_time_handler.handle_web_audio_stream(data)
 
                processing_time = time.time() - start_time
                if processing_time > 0.02:
                    logger.debug(f"Chunk {chunk_id} processed in {processing_time:.3f}s")
        except Exception as e:
            logger.error(f"Error processing audio chunk {chunk_id}: {e}")
        finally:
            self.active_tasks.discard(asyncio.current_task())
 
    async def _check_session_timeout(self, websocket: WebSocket):
        try:
            while not self.should_stop.is_set():
                await asyncio.sleep(1)  # Check every second
                if time.time() - self.session_start_time > self.session_timeout:
                    logger.info("web_socket_stream_handler",
                              message=f"Session timeout reached ({self.session_timeout}s). Closing connection.")
                    self.should_stop.set()
                    await self.close_websocket(websocket, code=1000, reason="Session timeout reached")
                    break
        except Exception as e:
            logger.error(f"Error in timeout checker: {e}")
            self.should_stop.set()
 
    async def close_websocket(self, websocket: WebSocket, code: int = 1000, reason: str = "Normal Closure"):
        try:
            if websocket.client_state != WebSocket.DISCONNECTED:
                # End Twilio call before closing websocket
                await websocket.close(code=code, reason=reason)
                logger.info("web_socket_stream_handler", message=f"WebSocket closed with code {code}: {reason}")
        except Exception as e:
            logger.error(f"Error closing websocket: {e}")
 
    async def _cleanup_tasks(self):
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            done_tasks = {t for t in self.active_tasks if t.done()}
            self.active_tasks -= done_tasks
            logger.debug(f"Cleaned up {len(done_tasks)} completed tasks. Active: {len(self.active_tasks)}")
            self.last_cleanup = current_time
 
    async def _cleanup(self):
        self.should_stop.set()
        for task in self.active_tasks:
            task.cancel()
        if self.active_tasks:
            try:
                await asyncio.wait(self.active_tasks, timeout=0.5)
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")