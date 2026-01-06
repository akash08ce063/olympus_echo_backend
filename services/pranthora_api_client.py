"""
Pranthora API Client for Voice Testing Platform.

This module provides a client to interact with the Pranthora backend API
for creating, updating, and managing agents.
"""

import json
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
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=30.0
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def create_agent(self, agent_data: Dict[str, Any], request_id: Optional[str] = None) -> Dict[str, Any]:
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
                "temperature": agent_data.get("temperature", 0.7)
            }

            url = f"{self.base_url}/api/v1/agents/simple"
            logger.info(f"Creating agent in Pranthora: {agent_data.get('name')}")

            # Prepare headers
            headers = {}
            if request_id:
                headers["x-pranthora-callid"] = request_id

            response = await self.client.post(
                url,
                json=request_data,
                headers=headers
            )

            if response.status_code == 201:
                result = response.json()
                logger.info(f"Successfully created agent in Pranthora: {result.get('agent', {}).get('id')}")
                return result
            else:
                error_detail = response.text
                logger.error(f"Failed to create agent in Pranthora: {response.status_code} - {error_detail}")
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
            update_data = {}

            # Agent fields
            if any(key in agent_data for key in ["name", "description", "is_active", "system_prompt"]):
                update_data["agent"] = {}
                for field in ["name", "description", "is_active"]:
                    if field in agent_data:
                        update_data["agent"][field] = agent_data[field]

            # Model config fields
            if "system_prompt" in agent_data or "temperature" in agent_data:
                update_data["agent_model_config"] = {
                    "model_provider_id": "openai",
                    "system_prompt": agent_data.get("system_prompt", ""),
                    "temperature": agent_data.get("temperature", 0.7),
                    "max_tokens": 4000  # Keep default max tokens
                }

            if not update_data:
                raise ValueError("No valid update fields provided")

            url = f"{self.base_url}/api/v1/agents/{agent_id}"
            logger.info(f"Updating agent in Pranthora: {agent_id}")

            response = await self.client.put(
                url,
                json=update_data
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Successfully updated agent in Pranthora: {agent_id}")
                return result
            else:
                error_detail = response.text
                logger.error(f"Failed to update agent in Pranthora: {response.status_code} - {error_detail}")
                raise Exception(f"Pranthora API error: {response.status_code} - {error_detail}")

        except Exception as e:
            logger.error(f"Error updating agent in Pranthora: {e}")
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
                logger.error(f"Failed to get agent from Pranthora: {response.status_code} - {error_detail}")
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
                logger.error(f"Failed to delete agent from Pranthora: {response.status_code} - {error_detail}")
                raise Exception(f"Pranthora API error: {response.status_code} - {error_detail}")

        except Exception as e:
            logger.error(f"Error deleting agent from Pranthora: {e}")
            raise
