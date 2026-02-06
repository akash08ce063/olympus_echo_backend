"""
Pranthora API Client for Voice Testing Platform.

This module provides a client to interact with the Pranthora backend API
for creating, updating, and managing agents.
"""

from typing import Dict, Any, Optional, List
import httpx
from pydantic import BaseModel, Field

from static_memory_cache import StaticMemoryCache
from telemetrics.logger import logger


class AgentCreateRequest(BaseModel):
    """Agent creation request schema matching Pranthora API."""

    name: str = Field(..., description="Agent name")
    description: Optional[str] = Field(None, description="Agent description")
    is_active: bool = Field(True, description="Whether agent is active")
    apply_noise_reduction: bool = Field(False, description="Apply noise reduction")
    recording_enabled: bool = Field(False, description="Enable recording")
    tts_filler_enabled: Optional[bool] = Field(None, description="Enable TTS filler")
    first_response_message: Optional[str] = Field(None, description="First response message")


class ModelConfigRequest(BaseModel):
    """Model configuration request schema."""

    model_provider_id: str = Field(..., description="Model provider ID")
    api_key_reference: Optional[str] = Field(None, description="API key reference")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum tokens")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    tool_prompt: Optional[str] = Field(None, description="Tool prompt")
    other_params: Optional[Dict[str, Any]] = Field(None, description="Other parameters")


class CompleteAgentRequest(BaseModel):
    """Complete agent creation request matching Pranthora API."""

    agent: AgentCreateRequest
    agent_model_config: Optional[ModelConfigRequest] = None


class PranthoraApiClient:
    """Client for interacting with Pranthora backend API."""

    def __init__(self):
        self.api_key = StaticMemoryCache.get_pranthora_api_key()
        self.base_url = StaticMemoryCache.get_pranthora_base_url()
        self.client = httpx.AsyncClient(
            headers={"x-api-key": self.api_key, "Content-Type": "application/json"}, timeout=30.0
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def create_agent(
        self, agent_data: Dict[str, Any], request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new agent in Pranthora backend.

        Args:
            agent_data: Agent configuration data
            request_id: Optional request ID to send in headers

        Returns:
            Created agent response
        """
        try:
            # Create SimpleAgentCreateRequest for simple agent creation
            request_data = {
                "name": agent_data.get("name", ""),
                "system_prompt": agent_data.get("system_prompt"),
                "temperature": agent_data.get("temperature", 0.7),
            }

            url = f"{self.base_url}/api/v1/agents/simple"
            logger.info(f"Creating agent in Pranthora: {agent_data.get('name')}")

            # Prepare headers
            headers = {}
            if request_id:
                headers["x-pranthora-callid"] = request_id

            response = await self.client.post(url, json=request_data, headers=headers)

            if response.status_code == 201:
                result = response.json()
                logger.info(
                    f"Successfully created agent in Pranthora: {result.get('agent', {}).get('id')}"
                )
                return result
            else:
                error_detail = response.text
                logger.error(
                    f"Failed to create agent in Pranthora: {response.status_code} - {error_detail}"
                )
                raise Exception(f"Pranthora API error: {response.status_code} - {error_detail}")

        except Exception as e:
            logger.error(f"Error creating agent in Pranthora: {e}")
            raise

    async def update_agent(self, agent_id: str, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing agent in Pranthora backend.

        Args:
            agent_id: Pranthora agent ID
            agent_data: Updated agent configuration data

        Returns:
            Updated agent response
        """
        try:
            # Prepare the update request
            update_data: Dict[str, Any] = {}

            # Agent fields
            if any(
                key in agent_data for key in ["name", "description", "is_active", "system_prompt"]
            ):
                update_data["agent"] = {}
                for field in ["name", "description", "is_active"]:
                    if field in agent_data:
                        update_data["agent"][field] = agent_data[field]

            # Model config fields
            if "system_prompt" in agent_data or "temperature" in agent_data:
                # Fetch default model provider ID from Pranthora so we never send an invalid ID
                default_provider_id = await self._get_default_model_provider_id()
                update_data["agent_model_config"] = {
                    "model_provider_id": default_provider_id,
                    "system_prompt": agent_data.get("system_prompt", ""),
                    "temperature": agent_data.get("temperature", 0.7),
                    "max_tokens": 4000,  # Keep default max tokens
                }

            if not update_data:
                raise ValueError("No valid update fields provided")

            url = f"{self.base_url}/api/v1/agents/{agent_id}"
            logger.info(f"Updating agent in Pranthora: {agent_id}")

            response = await self.client.put(
                url,
                params={"force_update": True},
                json=update_data,
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Successfully updated agent in Pranthora: {agent_id}")
                return result
            else:
                error_detail = response.text
                logger.error(
                    f"Failed to update agent in Pranthora: {response.status_code} - {error_detail}"
                )
                raise Exception(f"Pranthora API error: {response.status_code} - {error_detail}")

        except Exception as e:
            logger.error(f"Error updating agent in Pranthora: {e}")
            raise

    async def _get_default_model_provider_id(self) -> str:
        """
        Fetch a default model_provider_id from Pranthora.

        We call /api/v1/providers/model and prefer the one marked is_default,
        falling back to the first provider in the list.
        """
        try:
            url = f"{self.base_url}/api/v1/providers/model"
            logger.debug("Fetching default model provider from Pranthora")
            response = await self.client.get(url)
            if response.status_code != 200:
                raise Exception(f"provider list HTTP {response.status_code}: {response.text}")

            data = response.json()
            if not isinstance(data, list) or not data:
                raise Exception("No model providers returned from Pranthora")

            default_provider = next((p for p in data if p.get("is_default")), None)
            provider = default_provider or data[0]
            provider_id = provider.get("id")
            if not provider_id:
                raise Exception(f"Model provider object missing 'id': {provider}")
            return provider_id
        except Exception as e:
            logger.error(f"Failed to fetch default model provider from Pranthora: {e}")
            # Let caller see a clear failure; frontend will surface this
            raise

    async def get_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Get an agent from Pranthora backend.

        Args:
            agent_id: Pranthora agent ID

        Returns:
            Agent data
        """
        try:
            url = f"{self.base_url}/api/v1/agents/{agent_id}"
            logger.debug(f"Fetching agent from Pranthora: {agent_id}")

            response = await self.client.get(url)

            if response.status_code == 200:
                result = response.json()
                return result
            elif response.status_code == 404:
                raise Exception(f"Agent not found: {agent_id}")
            else:
                error_detail = response.text
                logger.error(
                    f"Failed to get agent from Pranthora: {response.status_code} - {error_detail}"
                )
                raise Exception(f"Pranthora API error: {response.status_code} - {error_detail}")

        except Exception as e:
            logger.error(f"Error getting agent from Pranthora: {e}")
            raise

    async def delete_agent(self, agent_id: str) -> bool:
        """
        Delete an agent from Pranthora backend.

        Args:
            agent_id: Pranthora agent ID

        Returns:
            Success status
        """
        try:
            url = f"{self.base_url}/api/v1/agents/{agent_id}?force_delete=true"

            response = await self.client.delete(url)

            if response.status_code in [200, 204]:
                logger.info(f"Successfully deleted agent from Pranthora: {agent_id}")
                return True
            else:
                error_detail = response.text
                logger.error(
                    f"Failed to delete agent from Pranthora: {response.status_code} - {error_detail}"
                )
                raise Exception(f"Pranthora API error: {response.status_code} - {error_detail}")

        except Exception as e:
            logger.error(f"Error deleting agent from Pranthora: {e}")
            raise

    async def get_call_logs(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get call logs/session transcripts from Pranthora backend by request ID.

        The request_id is the same as the x-pranthora-callid header value
        used when starting the test case execution.

        Args:
            request_id: The request ID (x-pranthora-callid)

        Returns:
            Call session data including transcripts, or None if not found
        """
        try:
            url = f"{self.base_url}/api/v1/call-analytics/call-logs/{request_id}"
            logger.info(f"📞 Fetching call logs from Pranthora for request_id: {request_id}, URL: {url}")

            response = await self.client.get(url)
            logger.info(f"📞 Pranthora API response status: {response.status_code} for request_id: {request_id}")

            if response.status_code == 200:
                result = response.json()
                transcript_count = len(result.get("call_transcript", [])) if result.get("call_transcript") else 0
                logger.info(f"✅ Successfully fetched call logs for request_id: {request_id}, transcript messages: {transcript_count}")
                return result
            elif response.status_code == 404:
                logger.warning(f"⚠️ Call logs not found for request_id: {request_id}")
                return None
            else:
                error_detail = response.text
                logger.error(
                    f"❌ Failed to get call logs from Pranthora: {response.status_code} - {error_detail}"
                )
                raise Exception(f"Pranthora API error: {response.status_code} - {error_detail}")

        except Exception as e:
            logger.error(f"❌ Error getting call logs from Pranthora for request_id {request_id}: {e}", exc_info=True)
            raise

    async def map_agent_to_phone_number(self, agent_id: str, phone_numbers: List[str]) -> Dict[str, Any]:
        """
        Map a Pranthora agent to a list of phone numbers.

        This wraps the pranthora_backend /phone/map_agent_to_phone_number endpoint.
        """
        try:
            url = f"{self.base_url}/api/v1/phone/map_agent_to_phone_number"
            payload = {
                "agent_id": agent_id,
                "phone_numbers": phone_numbers,
            }
            logger.info(f"Mapping agent {agent_id} to {len(phone_numbers)} phone numbers via Pranthora")
            response = await self.client.post(url, json=payload)
            if response.status_code in (200, 201):
                return response.json()
            error_detail = response.text
            logger.error(
                f"Failed to map agent to phone numbers in Pranthora: {response.status_code} - {error_detail}"
            )
            raise Exception(f"Pranthora API error: {response.status_code} - {error_detail}")
        except Exception as e:
            logger.error(f"Error mapping agent to phone numbers in Pranthora: {e}")
            raise

    async def check_agent_phone_numbers_mapings(self, agent_id: str, phone_numbers: List[str]) -> Dict[str, Any]:
        """
        Check if the given phone numbers are mapped to the specified agent.

        This wraps the pranthora_backend /phone/check_agent_phone_numbers_mapings endpoint.
        """
        try:
            url = f"{self.base_url}/api/v1/phone/check_agent_phone_numbers_mapings"
            payload = {
                "agent_id": agent_id,
                "phone_numbers": phone_numbers,
            }
            logger.info(f"Checking phone number mappings for agent {agent_id} via Pranthora")
            response = await self.client.post(url, json=payload)
            if response.status_code == 200:
                return response.json()
            error_detail = response.text
            logger.error(
                f"Failed to check phone number mappings in Pranthora: {response.status_code} - {error_detail}"
            )
            raise Exception(f"Pranthora API error: {response.status_code} - {error_detail}")
        except Exception as e:
            logger.error(f"Error checking agent phone number mappings in Pranthora: {e}")
            raise

    async def initiate_phone_call(self, target_phone_number: str, pranthora_agent_id: str) -> Dict[str, Any]:
        """
        Initiate an outbound phone call via Pranthora SDK /calls endpoint.

        Args:
            target_phone_number: E.164 phone number of the target agent.
            pranthora_agent_id: Pranthora agent ID for the user agent (sent as agent_id query param).
        """
        try:
            url = f"{self.base_url}/calls"
            params = {
                "phoneNumber": target_phone_number,
                "agent_id": pranthora_agent_id,
            }
            logger.info(
                f"Initiating phone call via Pranthora: target={target_phone_number}, agent_id={pranthora_agent_id}"
            )
            response = await self.client.post(url, params=params)
            if response.status_code in (200, 201):
                return response.json()
            error_detail = response.text
            logger.error(
                f"Failed to initiate phone call via Pranthora: {response.status_code} - {error_detail}"
            )
            raise Exception(f"Pranthora API error: {response.status_code} - {error_detail}")
        except Exception as e:
            logger.error(f"Error initiating phone call via Pranthora: {e}")
            raise

    async def end_phone_call(self, call_sid: str, from_phone_number: str) -> None:
        """
        End an ongoing phone call via Pranthora SDK /calls/end endpoint.

        Args:
            call_sid: Twilio Call SID to end.
            from_phone_number: Twilio phone number used as the caller ID.
        """
        try:
            url = f"{self.base_url}/calls/end"
            params = {
                "call_sid": call_sid,
                "from_phone_number": from_phone_number,
            }
            logger.info(f"Ending phone call via Pranthora: call_sid={call_sid}, from={from_phone_number}")
            response = await self.client.post(url, params=params)
            if response.status_code not in (200, 204):
                error_detail = response.text
                logger.error(
                    f"Failed to end phone call via Pranthora: {response.status_code} - {error_detail}"
                )
                raise Exception(f"Pranthora API error: {response.status_code} - {error_detail}")
        except Exception as e:
            logger.error(f"Error ending phone call via Pranthora: {e}")
            raise
