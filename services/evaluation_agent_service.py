"""
Evaluation Agent Service for Voice Testing Platform.

This module provides a service to evaluate test conversations using the Pranthora
evaluation agent via WebSocket text endpoint.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, ConnectionClosedError, InvalidURI

from static_memory_cache import StaticMemoryCache
from telemetrics.logger import logger


class EvaluationAgentService:
    """Service for evaluating test conversations using the Pranthora evaluation agent."""

    # The evaluation agent ID in Pranthora platform
    EVALUATION_AGENT_ID = "c5887ff1-b36d-4794-b380-bad598a0aac6"
    
    # Connection timeout in seconds
    CONNECTION_TIMEOUT = 30
    
    # Response timeout in seconds
    RESPONSE_TIMEOUT = 120

    def __init__(self):
        self.api_key = StaticMemoryCache.get_pranthora_api_key()
        self.base_url = StaticMemoryCache.get_pranthora_base_url()

    def _build_websocket_url(self) -> str:
        """Build the WebSocket URL for the text endpoint."""
        if not self.base_url:
            raise ValueError("Pranthora base URL not configured")
        
        # Convert HTTP to WS
        if self.base_url.startswith("https://"):
            ws_url = self.base_url.replace("https://", "wss://")
        else:
            ws_url = self.base_url.replace("http://", "ws://")
        
        # The correct path is /api/text/stream_text_ws (not /api/v1/text/stream_text_ws)
        return f"{ws_url}/api/text/stream_text_ws"

    def _build_evaluation_prompt(
        self,
        transcript: List[Dict[str, Any]],
        goals: List[Dict[str, Any]],
        evaluation_criteria: List[Dict[str, Any]],
        test_case_name: str
    ) -> str:
        """Build the evaluation prompt for the evaluation agent."""
        
        # Format transcript for evaluation
        formatted_transcript = []
        for msg in transcript:
            role = msg.get("role", "unknown")
            content = msg.get("content", msg.get("message", ""))
            timestamp = msg.get("timestamp", "")
            
            # In the transcript: "user" = TARGET agent, "assistant" = TESTER agent
            speaker = "TARGET" if role == "user" else "TESTER"
            formatted_transcript.append(f"[{speaker}]: {content}")
        
        transcript_text = "\n".join(formatted_transcript) if formatted_transcript else "No conversation recorded."
        
        # Format goals
        goals_text = ""
        if goals:
            goals_list = []
            for i, goal in enumerate(goals, 1):
                if isinstance(goal, dict):
                    goal_desc = goal.get("description", goal.get("prompt", goal.get("goal", str(goal))))
                    goal_type = goal.get("type", "general")
                    goals_list.append(f"{i}. [{goal_type.upper()}] {goal_desc}")
                else:
                    goals_list.append(f"{i}. {str(goal)}")
            goals_text = "\n".join(goals_list)
        else:
            goals_text = "No specific goals defined."
        
        # Format evaluation criteria
        criteria_text = ""
        if evaluation_criteria:
            criteria_list = []
            for i, criterion in enumerate(evaluation_criteria, 1):
                if isinstance(criterion, dict):
                    crit_type = criterion.get("type", "general")
                    crit_expected = criterion.get("expected", criterion.get("description", str(criterion)))
                    crit_weight = criterion.get("weight", 1.0)
                    criteria_list.append(f"{i}. [{crit_type.upper()}] {crit_expected} (weight: {crit_weight})")
                else:
                    criteria_list.append(f"{i}. {str(criterion)}")
            criteria_text = "\n".join(criteria_list)
        else:
            criteria_text = "No specific evaluation criteria defined."
        
        # Build the complete prompt
        prompt = f"""Please evaluate the following conversation between a TESTER agent (automated test agent) and a TARGET agent (the agent being tested).

TEST CASE: {test_case_name}

=== CONVERSATION TRANSCRIPT ===
{transcript_text}

=== GOALS TO EVALUATE ===
{goals_text}

=== EVALUATION CRITERIA ===
{criteria_text}

Please analyze this conversation and provide a comprehensive evaluation. Return your response as a valid JSON object with the following structure:

{{
    "overall_score": <float between 0.0 and 1.0>,
    "overall_status": "<passed|partial|failed>",
    "summary": "<brief summary of the evaluation>",
    "goals_analysis": [
        {{
            "goal_id": <number>,
            "goal_description": "<the goal being evaluated>",
            "status": "<passed|failed|partial>",
            "score": <float between 0.0 and 1.0>,
            "analysis": "<detailed analysis of how well this goal was achieved>",
            "evidence": "<specific quotes or examples from the conversation>"
        }}
    ],
    "criteria_evaluated": [
        {{
            "criterion_id": <number>,
            "type": "<criterion type>",
            "expected": "<what was expected>",
            "actual": "<what actually happened>",
            "passed": <true|false>,
            "score": <float between 0.0 and 1.0>,
            "details": "<explanation of the evaluation>",
            "evidence": "<specific quotes or examples>"
        }}
    ],
    "strengths": ["<list of things the TARGET agent did well>"],
    "weaknesses": ["<list of areas where the TARGET agent could improve>"],
    "recommendations": ["<actionable recommendations for improvement>"]
}}

IMPORTANT: 
- Return ONLY the JSON object, no additional text before or after.
- Ensure the JSON is valid and properly formatted.
- Be objective and thorough in your evaluation.
- Use specific examples from the conversation to support your analysis.
- Score of 0.8+ is "passed", 0.5-0.8 is "partial", below 0.5 is "failed"."""

        return prompt

    async def evaluate_conversation(
        self,
        transcript: List[Dict[str, Any]],
        goals: List[Dict[str, Any]],
        evaluation_criteria: List[Dict[str, Any]],
        test_case_name: str = "Test Case"
    ) -> Dict[str, Any]:
        """
        Evaluate a conversation using the Pranthora evaluation agent.

        Args:
            transcript: List of conversation messages with role and content
            goals: List of test goals to evaluate
            evaluation_criteria: List of evaluation criteria
            test_case_name: Name of the test case being evaluated

        Returns:
            Evaluation result as a dictionary
        """
        try:
            ws_url = self._build_websocket_url()
            
            logger.info(f"Connecting to evaluation agent at {ws_url}")
            logger.info(f"Using system prompt for evaluation agent (logged for debugging)")
            
            # Build the evaluation prompt
            prompt = self._build_evaluation_prompt(
                transcript, goals, evaluation_criteria, test_case_name
            )
            
            # Log the system prompt as per user preference
            logger.info(f"Evaluation prompt for test case '{test_case_name}': {prompt[:500]}...")
            
            # Connect to the WebSocket endpoint
            # No authentication headers needed - the endpoint accepts connections without auth
            logger.info(f"Connecting to WebSocket at {ws_url} (no auth required)")
            
            async with websockets.connect(
                ws_url,
                # No additional_headers - endpoint doesn't require authentication
                open_timeout=self.CONNECTION_TIMEOUT,
                close_timeout=10
            ) as websocket:
                
                # Send initial configuration with agent_id (required first message)
                # The endpoint expects this as the first message after connection
                init_message = {
                    "agent_id": self.EVALUATION_AGENT_ID
                }
                await websocket.send(json.dumps(init_message))
                logger.info(f"Sent init message to evaluation agent: {init_message}")
                
                # Small delay to ensure connection is established and agent is initialized
                await asyncio.sleep(0.5)
                
                # Send the evaluation prompt as input_text
                # The endpoint expects {"input_text": "..."} for subsequent messages
                user_message = {
                    "input_text": prompt
                }
                await websocket.send(json.dumps(user_message))
                logger.info(f"Sent evaluation prompt to agent ({len(prompt)} chars)")
                
                # Collect response chunks continuously
                # The response comes in streaming chunks, so we need to collect ALL chunks
                # until the connection closes or times out
                response_chunks = []
                chunk_count = 0
                start_time = asyncio.get_event_loop().time()
                
                logger.info("Starting to collect streaming response chunks from evaluation agent...")
                
                while True:
                    try:
                        # Set timeout for receiving each chunk
                        # Use a shorter timeout per chunk, but continue until connection closes
                        response = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=5.0  # Shorter timeout per chunk to detect end of stream
                        )
                        
                        try:
                            response_data = json.loads(response)
                            if "response" in response_data:
                                chunk = response_data["response"]
                                response_chunks.append(chunk)
                                chunk_count += 1
                                logger.debug(f"Received chunk #{chunk_count}: {len(chunk)} chars (total: {sum(len(c) for c in response_chunks)} chars)")
                            elif "error" in response_data:
                                raise Exception(f"Evaluation agent error: {response_data['error']}")
                            else:
                                # Unknown response format, log it
                                logger.warning(f"Unexpected response format: {response_data}")
                        except json.JSONDecodeError:
                            # If not JSON, treat as raw text chunk
                            response_chunks.append(response)
                            chunk_count += 1
                            logger.debug(f"Received raw chunk #{chunk_count}: {len(response)} chars")
                            
                    except asyncio.TimeoutError:
                        # If we haven't received any chunks yet, it's a real timeout
                        if chunk_count == 0:
                            logger.warning("Timeout waiting for first response chunk from evaluation agent")
                            break
                        # If we have chunks, wait a bit more for potential final chunks
                        # Then break if no more data comes
                        logger.info(f"Timeout after {chunk_count} chunks. Waiting for final chunks...")
                        try:
                            # Give it one more chance with a short timeout
                            response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                            response_data = json.loads(response)
                            if "response" in response_data:
                                response_chunks.append(response_data["response"])
                                chunk_count += 1
                                logger.debug(f"Received final chunk #{chunk_count}")
                            continue
                        except (asyncio.TimeoutError, ConnectionClosed, ConnectionClosedOK, ConnectionClosedError):
                            # No more chunks coming, break
                            logger.info(f"Stream ended after {chunk_count} chunks")
                            break
                    except (ConnectionClosed, ConnectionClosedOK, ConnectionClosedError) as e:
                        logger.info(f"WebSocket connection closed by evaluation agent after {chunk_count} chunks")
                        break
                
                # Combine all chunks into complete response
                full_response = "".join(response_chunks)
                logger.info(f"Collected {chunk_count} chunks, total response length: {len(full_response)} chars")
                
                if not full_response or len(full_response.strip()) == 0:
                    logger.warning("No response received from evaluation agent")
                    return self._create_error_result(
                        "No response received from evaluation agent",
                        goals,
                        evaluation_criteria
                    )
                
                # Parse the complete combined response
                evaluation_result = self._parse_evaluation_response(full_response)
                
                logger.info(f"Evaluation completed: overall_score={evaluation_result.get('overall_score', 0.0)}, status={evaluation_result.get('overall_status', 'unknown')}")
                
                return evaluation_result
                
        except InvalidURI as e:
            logger.error(f"Invalid WebSocket URI: {e}", exc_info=True)
            return self._create_error_result(f"Invalid WebSocket URI: {str(e)}", goals, evaluation_criteria)
        except (ConnectionClosed, ConnectionClosedOK, ConnectionClosedError) as e:
            logger.error(f"WebSocket connection closed unexpectedly: {e}", exc_info=True)
            return self._create_error_result(f"Connection closed: {str(e)}", goals, evaluation_criteria)
        except asyncio.TimeoutError as e:
            logger.error(f"WebSocket connection timeout: {e}", exc_info=True)
            return self._create_error_result(f"Connection timeout: {str(e)}", goals, evaluation_criteria)
        except Exception as e:
            logger.error(f"Error evaluating conversation: {e}", exc_info=True)
            # Return a fallback error result with detailed error message
            error_msg = f"{type(e).__name__}: {str(e)}"
            return self._create_error_result(error_msg, goals, evaluation_criteria)

    def _is_complete_json(self, text: str) -> bool:
        """
        Check if the text contains a complete JSON object.
        
        Note: This method is kept for backward compatibility but is no longer
        used in the main streaming logic, as we now collect all chunks until
        the connection closes.
        """
        text = text.strip()
        if not text:
            return False
        
        # Try to find and parse JSON
        try:
            # Look for JSON object boundaries
            start_idx = text.find('{')
            if start_idx == -1:
                return False
            
            # Try to parse from the first {
            json.loads(text[start_idx:])
            return True
        except json.JSONDecodeError:
            return False

    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse the evaluation agent's response into a structured result."""
        try:
            # Find the JSON object in the response
            response = response.strip()
            
            # Look for JSON object boundaries
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx + 1]
                result = json.loads(json_str)
                
                # Validate and normalize the result
                return self._normalize_evaluation_result(result)
            
            # If no valid JSON found, return error result
            logger.warning(f"No valid JSON found in evaluation response: {response[:500]}...")
            return self._create_fallback_result(response)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse evaluation response as JSON: {e}")
            return self._create_fallback_result(response)

    def _normalize_evaluation_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and validate the evaluation result structure."""
        # Ensure all required fields exist with defaults
        normalized = {
            "overall_score": float(result.get("overall_score", 0.0)),
            "overall_status": result.get("overall_status", "failed"),
            "summary": result.get("summary", "Evaluation completed."),
            "goals_analysis": result.get("goals_analysis", []),
            "criteria_evaluated": result.get("criteria_evaluated", []),
            "passed_criteria": 0,
            "total_criteria": 0,
            "strengths": result.get("strengths", []),
            "weaknesses": result.get("weaknesses", []),
            "recommendations": result.get("recommendations", []),
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Count passed criteria
        criteria = normalized.get("criteria_evaluated", [])
        normalized["total_criteria"] = len(criteria)
        normalized["passed_criteria"] = sum(1 for c in criteria if c.get("passed", False))
        
        # Validate overall_status based on score
        score = normalized["overall_score"]
        if score >= 0.8:
            normalized["overall_status"] = "passed"
        elif score >= 0.5:
            normalized["overall_status"] = "partial"
        else:
            normalized["overall_status"] = "failed"
        
        return normalized

    def _create_fallback_result(self, response: str) -> Dict[str, Any]:
        """Create a fallback result when JSON parsing fails."""
        return {
            "overall_score": 0.0,
            "overall_status": "failed",
            "summary": f"Evaluation response could not be parsed. Raw response: {response[:500]}...",
            "goals_analysis": [],
            "criteria_evaluated": [],
            "passed_criteria": 0,
            "total_criteria": 0,
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "error": "Failed to parse evaluation response",
            "raw_response": response[:1000],
            "timestamp": asyncio.get_event_loop().time()
        }

    def _create_error_result(
        self,
        error_message: str,
        goals: List[Dict[str, Any]],
        evaluation_criteria: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create an error result when evaluation fails."""
        import time
        return {
            "overall_score": 0.0,
            "overall_status": "failed",
            "summary": f"Evaluation failed: {error_message}",
            "goals_analysis": [],
            "criteria_evaluated": [],
            "passed_criteria": 0,
            "total_criteria": len(evaluation_criteria) if evaluation_criteria else 0,
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "error": error_message,
            "error_type": "evaluation_error",
            "timestamp": time.time()
        }

    async def evaluate_with_retry(
        self,
        transcript: List[Dict[str, Any]],
        goals: List[Dict[str, Any]],
        evaluation_criteria: List[Dict[str, Any]],
        test_case_name: str = "Test Case",
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Evaluate a conversation with retry logic.

        Args:
            transcript: List of conversation messages
            goals: List of test goals
            evaluation_criteria: List of evaluation criteria
            test_case_name: Name of the test case
            max_retries: Maximum number of retry attempts

        Returns:
            Evaluation result as a dictionary
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                result = await self.evaluate_conversation(
                    transcript, goals, evaluation_criteria, test_case_name
                )
                
                # If we got a valid result (not an error result), return it
                if not result.get("error"):
                    return result
                
                last_error = result.get("error", "Unknown error")
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Evaluation attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries:
                # Wait before retry with exponential backoff
                wait_time = 2 ** attempt
                logger.info(f"Retrying evaluation in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
        
        # All retries exhausted
        logger.error(f"All evaluation attempts failed after {max_retries + 1} attempts. Last error: {last_error}")
        error_result = self._create_error_result(
            f"All evaluation attempts failed: {last_error}",
            goals,
            evaluation_criteria
        )
        error_result["retry_attempts"] = max_retries + 1
        error_result["last_error"] = last_error
        return error_result

