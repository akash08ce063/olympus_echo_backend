"""
Test suite database models based on SQL schema.

This module contains Pydantic models for all database entities
used in the voice testing platform.
"""

from datetime import datetime
from typing import List, Optional, Any, Dict
from uuid import UUID

from pydantic import BaseModel, Field


class TestSuiteBase(BaseModel):
    """Base model for test suites."""

    user_id: Optional[UUID] = Field(None, description="User ID who owns this test suite (can also be provided as query parameter)")
    name: str = Field(..., min_length=1, max_length=255, description="Test suite name")
    description: Optional[str] = Field(None, description="Optional description")
    target_agent_id: Optional[UUID] = Field(None, description="ID of the target agent")
    user_agent_id: Optional[UUID] = Field(None, description="ID of the user agent")


class TestSuiteCreate(TestSuiteBase):
    """Model for creating a new test suite."""
    pass


class TestSuiteCreateRequest(BaseModel):
    """Request model for creating test suites (user_id can be in query parameter or body)."""
    user_id: Optional[UUID] = Field(None, description="User ID (can also be provided as query parameter)")
    name: str = Field(..., min_length=1, max_length=255, description="Test suite name")
    description: Optional[str] = Field(None, description="Optional description")
    target_agent_id: Optional[UUID] = Field(None, description="ID of the target agent")
    user_agent_id: Optional[UUID] = Field(None, description="ID of the user agent")


class TestSuiteUpdate(BaseModel):
    """Model for updating a test suite."""

    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Test suite name")
    description: Optional[str] = Field(None, description="Optional description")
    target_agent_id: Optional[UUID] = Field(None, description="ID of the target agent")
    user_agent_id: Optional[UUID] = Field(None, description="ID of the user agent")


class TestSuite(TestSuiteBase):
    """Complete test suite model."""

    id: UUID = Field(..., description="Unique test suite ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class TargetAgentBase(BaseModel):
    """Base model for target agents."""

    user_id: UUID = Field(..., description="User ID who owns this target agent")
    name: str = Field(..., min_length=1, max_length=255, description="Target agent name")
    websocket_url: str = Field(..., min_length=1, max_length=500, description="WebSocket URL")
    sample_rate: int = Field(16000, ge=1000, le=48000, description="Audio sample rate in Hz")
    encoding: str = Field("pcm_s16le", description="Audio encoding (e.g., pcm_s16le, opus, pcm_mulaw)")


class TargetAgentCreate(TargetAgentBase):
    """Model for creating a new target agent."""
    pass


class TargetAgentUpdate(BaseModel):
    """Model for updating a target agent."""

    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Target agent name")
    websocket_url: Optional[str] = Field(None, min_length=1, max_length=500, description="WebSocket URL")
    sample_rate: Optional[int] = Field(None, ge=1000, le=48000, description="Audio sample rate in Hz")
    encoding: Optional[str] = Field(None, description="Audio encoding")


class TargetAgent(TargetAgentBase):
    """Complete target agent model."""

    id: UUID = Field(..., description="Unique target agent ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class UserAgentBase(BaseModel):
    """Base model for user agents."""

    user_id: UUID = Field(..., description="User ID who owns this user agent")
    name: str = Field(..., min_length=1, max_length=255, description="User agent name")
    system_prompt: str = Field(..., description="System prompt for the agent")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature setting for the AI model")
    pranthora_agent_id: Optional[str] = Field(None, description="Corresponding agent ID in Pranthora backend")


class UserAgentCreate(UserAgentBase):
    """Model for creating a new user agent."""
    pass


class UserAgentUpdate(BaseModel):
    """Model for updating a user agent."""

    name: Optional[str] = Field(None, min_length=1, max_length=255, description="User agent name")
    system_prompt: Optional[str] = Field(None, description="System prompt for the agent")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature setting for the AI model")


class UserAgent(UserAgentBase):
    """Complete user agent model."""

    id: UUID = Field(..., description="Unique user agent ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    evaluation_criteria: Optional[List[Dict[str, Any]]] = Field(None, description="Array of evaluation criteria")
    agent_model_config: Optional[Dict[str, Any]] = Field(None, description="Model configuration for the agent")


class TestCaseBase(BaseModel):
    """Base model for test cases."""

    name: str = Field(..., min_length=1, max_length=255, description="Test case name")
    goals: List[Dict[str, Any]] = Field(..., description="Array of test goals/prompts")
    evaluation_criteria: List[Dict[str, Any]] = Field(..., description="Array of evaluation criteria")
    timeout_seconds: int = Field(300, ge=1, description="Timeout in seconds")
    order_index: int = Field(0, ge=0, description="Order index for sorting")
    is_active: bool = Field(True, description="Whether this test case is active")
    attempts: int = Field(1, ge=1, description="Number of times this test case should retry on failure")
    default_concurrent_calls: int = Field(1, ge=1, description="Default number of concurrent calls for this test case")


class TestCaseCreate(TestCaseBase):
    """Model for creating a new test case."""

    test_suite_id: UUID = Field(..., description="ID of the test suite this case belongs to")


class TestCaseUpdate(BaseModel):
    """Model for updating a test case."""

    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Test case name")
    goals: Optional[List[Dict[str, Any]]] = Field(None, description="Array of test goals/prompts")
    evaluation_criteria: Optional[List[Dict[str, Any]]] = Field(None, description="Array of evaluation criteria")
    timeout_seconds: Optional[int] = Field(None, ge=1, description="Timeout in seconds")
    order_index: Optional[int] = Field(None, ge=0, description="Order index for sorting")
    is_active: Optional[bool] = Field(None, description="Whether this test case is active")
    attempts: Optional[int] = Field(None, ge=1, description="Number of times this test case should retry on failure")
    default_concurrent_calls: Optional[int] = Field(None, ge=1, description="Default number of concurrent calls for this test case")


class TestCase(TestCaseBase):
    """Complete test case model."""

    id: UUID = Field(..., description="Unique test case ID")
    test_suite_id: UUID = Field(..., description="ID of the test suite this case belongs to")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    status: Optional[str] = Field(None, description="Current status of the test case (running, completed, failed, pending)")


class TestRunHistoryBase(BaseModel):
    """Base model for test run history."""

    test_suite_id: Optional[UUID] = Field(None, description="ID of the test suite that was run (can be null if suite was deleted)")
    user_id: UUID = Field(..., description="User ID who ran the test")
    status: str = Field(
        "running",
        description="Test run status (running, completed, failed, cancelled)"
    )
    total_test_cases: int = Field(0, ge=0, description="Total number of test cases")
    passed_count: int = Field(0, ge=0, description="Number of passed test cases")
    failed_count: int = Field(0, ge=0, description="Number of failed test cases")
    alert_count: int = Field(0, ge=0, description="Number of alerts generated")


class TestRunHistory(TestRunHistoryBase):
    """Complete test run history model."""

    id: UUID = Field(..., description="Unique test run ID")
    started_at: datetime = Field(..., description="When the test run started")
    completed_at: Optional[datetime] = Field(None, description="When the test run completed")


class TestCaseResultBase(BaseModel):
    """Base model for test case results."""

    test_run_id: UUID = Field(..., description="ID of the test run this result belongs to")
    test_case_id: UUID = Field(..., description="ID of the test case that was executed")
    test_suite_id: UUID = Field(..., description="ID of the test suite this result belongs to")
    status: str = Field(..., description="Result status (pass, fail, alert)")
    recording_file_id: Optional[UUID] = Field(None, description="Supabase storage file ID")
    conversation_logs: Optional[List[Dict[str, Any]]] = Field(None, description="Array of conversation turns")
    evaluation_result: Optional[Dict[str, Any]] = Field(None, description="Evaluation from user agent")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class TestCaseResult(TestCaseResultBase):
    """Complete test case result model."""

    id: UUID = Field(..., description="Unique test case result ID")
    started_at: datetime = Field(..., description="When the test case started")
    completed_at: Optional[datetime] = Field(None, description="When the test case completed")


class TestAlertBase(BaseModel):
    """Base model for test alerts."""

    test_case_result_id: UUID = Field(..., description="ID of the test case result this alert belongs to")
    alert_type: str = Field(..., description="Type of alert")
    severity: str = Field(..., description="Alert severity (low, medium, high)")
    message: str = Field(..., description="Alert message")


class TestAlert(TestAlertBase):
    """Complete test alert model."""

    id: UUID = Field(..., description="Unique alert ID")
    created_at: datetime = Field(..., description="When the alert was created")


# Response models for API endpoints
class TestSuiteWithRelations(TestSuite):
    """Test suite with related entities."""

    target_agent: Optional[TargetAgent] = Field(None, description="Target agent details")
    user_agent: Optional[UserAgent] = Field(None, description="User agent details")
    test_cases: List[TestCase] = Field(default_factory=list, description="Test cases in this suite")
    suite_status: Optional[str] = Field(None, description="Overall suite status from latest test run")


class TestRunWithResults(TestRunHistory):
    """Test run with results."""

    test_case_results: List[TestCaseResult] = Field(default_factory=list, description="Results of individual test cases")


class TestCaseResultWithAlerts(TestCaseResult):
    """Test case result with alerts."""

    alerts: List[TestAlert] = Field(default_factory=list, description="Alerts generated for this test case")
